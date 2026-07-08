import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet.core import multiclass_nms
from mmdet.models.builder import HEADS
from mmdet.models.losses import accuracy
from mmdet.models.roi_heads.bbox_heads import ConvFCBBoxHead


@HEADS.register_module()
class LCCBoxHead(ConvFCBBoxHead):

    def __init__(self,
                 num_novel_classes=5,
                 use_dropout=False,
                 dropout_ratio=0.8,
                 base_score_temperature=1.0,
                 novel_score_temperature=1.0,
                 novel_bg_score_power=1.0,
                 novel_prob_score_power=1.0,
                 use_visual_primitives=False,
                 primitive_dim=8,
                 primitive_hidden_dim=128,
                 use_semantic_aux=False,
                 semantic_dim=32,
                 semantic_logit_scale=20.0,
                 semantic_align_weight=0.1,
                 semantic_margin_weight=0.1,
                 semantic_gamma=0.5,
                 semantic_topk=3,
                 class_names=None,
                 semantic_embeddings=None,
                 use_vl_prototypes=False,
                 use_adaptive_prototype_alpha=True,
                 fixed_prototype_alpha=0.5,
                 prototype_momentum=0.9,
                 prototype_count_temperature=16.0,
                 prototype_logit_scale=10.0,
                 prototype_calibration_weight=1.0,
                 prototype_fusion_weight=0.5,
                 prototype_align_weight=0.05,
                 prototype_sep_weight=0.05,
                 prototype_sep_use_semantic_weight=False,
                 prototype_warmup_iters=0,
                 prototype_margin=0.1,
                 bg_suppress_weight=0.5,
                 ocs_loss_weight=0.0,
                 ocs_score_threshold=0.25,
                 ocs_warmup_iters=0,
                 use_ocs_aux_branch=False,
                 ocs_aux_detach=True,
                 prototype_loss_on_novel_only=True,
                 use_hallucinator=False,
                 hallucinator_noise_dim=128,
                 hallucinator_weight=0.1,
                 imprint_momentum=0.9,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_novel_classes = num_novel_classes
        self.use_dropout = use_dropout
        self.dropout_ratio = dropout_ratio
        self.base_score_temperature = max(float(base_score_temperature), 1e-6)
        self.novel_score_temperature = max(float(novel_score_temperature), 1e-6)
        self.novel_bg_score_power = max(float(novel_bg_score_power), 1e-6)
        self.novel_prob_score_power = max(float(novel_prob_score_power), 1e-6)
        self.use_visual_primitives = bool(use_visual_primitives)
        self.primitive_dim = int(primitive_dim)
        self.primitive_hidden_dim = int(primitive_hidden_dim)
        self.use_semantic_aux = bool(use_semantic_aux)
        self.use_vl_prototypes = bool(use_vl_prototypes)
        self.use_prototype_space = self.use_semantic_aux or self.use_vl_prototypes
        self.semantic_dim = int(semantic_dim)
        self.semantic_logit_scale = float(semantic_logit_scale)
        self.semantic_align_weight = float(semantic_align_weight)
        self.semantic_margin_weight = float(semantic_margin_weight)
        self.semantic_gamma = float(semantic_gamma)
        self.semantic_topk = int(semantic_topk)
        self.class_names = tuple(class_names) if class_names is not None else None
        self.use_adaptive_prototype_alpha = bool(use_adaptive_prototype_alpha)
        self.fixed_prototype_alpha = float(fixed_prototype_alpha)
        self.prototype_momentum = float(prototype_momentum)
        self.prototype_count_temperature = max(float(prototype_count_temperature), 1e-6)
        self.prototype_logit_scale = float(prototype_logit_scale)
        self.prototype_calibration_weight = float(prototype_calibration_weight)
        self.prototype_fusion_weight = float(prototype_fusion_weight)
        self.prototype_align_weight = float(prototype_align_weight)
        self.prototype_sep_weight = float(prototype_sep_weight)
        self.prototype_sep_use_semantic_weight = bool(
            prototype_sep_use_semantic_weight)
        self.prototype_warmup_iters = max(int(prototype_warmup_iters), 0)
        self.prototype_margin = float(prototype_margin)
        self.bg_suppress_weight = float(bg_suppress_weight)
        self.ocs_loss_weight = float(ocs_loss_weight)
        self.ocs_score_threshold = float(ocs_score_threshold)
        self.ocs_warmup_iters = max(int(ocs_warmup_iters), 0)
        self.use_ocs_aux_branch = bool(use_ocs_aux_branch)
        self.ocs_aux_detach = bool(ocs_aux_detach)
        self.prototype_loss_on_novel_only = bool(prototype_loss_on_novel_only)
        # Iteration counter used to ramp prototype-related weights from 0 -> 1
        # over the first `prototype_warmup_iters` training steps. Stored as a
        # buffer so it is preserved across checkpoints.
        self.register_buffer('proto_step',
                             torch.zeros(1, dtype=torch.long),
                             persistent=False)

        self.fc_cls_novel = torch.nn.Linear(self.cls_last_dim,
                                            self.num_novel_classes + 1)
        if self.use_ocs_aux_branch:
            self.ocs_context_cls = torch.nn.Linear(self.cls_last_dim,
                                                   self.num_novel_classes + 1)
        else:
            self.ocs_context_cls = None

        # === Novel Classifier Enhancement (NCE) ===
        # Feature hallucinator: generates diverse novel-class features from
        # a single prototype by learning a residual perturbation conditioned
        # on random noise. Trained jointly with the detector.
        self.use_hallucinator = bool(use_hallucinator)
        self.hallucinator_noise_dim = int(hallucinator_noise_dim)
        self.hallucinator_weight = float(hallucinator_weight)
        self.imprint_momentum = float(imprint_momentum)

        if self.use_hallucinator:
            # Hallucinator: noise + prototype -> residual feature
            self.hallucinator = torch.nn.Sequential(
                torch.nn.Linear(self.cls_last_dim + self.hallucinator_noise_dim,
                                self.cls_last_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(self.cls_last_dim, self.cls_last_dim))
            # Running mean of novel class features for weight imprinting
            self.register_buffer(
                'novel_class_prototypes',
                torch.zeros(self.num_novel_classes, self.cls_last_dim))
            self.register_buffer(
                'novel_class_counts',
                torch.zeros(self.num_novel_classes))
        if self.use_visual_primitives:
            self.primitive_encoder = torch.nn.Sequential(
                torch.nn.Linear(self.primitive_dim, self.primitive_hidden_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(self.primitive_hidden_dim, self.cls_last_dim),
                torch.nn.ReLU(inplace=True))

        if self.use_prototype_space:
            semantic_table = self._init_semantic_embeddings(
                class_names=self.class_names,
                semantic_embeddings=semantic_embeddings,
                semantic_dim=self.semantic_dim)
            self.semantic_projector = torch.nn.Linear(self.cls_last_dim,
                                                      self.semantic_dim)
            self.register_buffer('semantic_embeddings', semantic_table)
            self.register_buffer(
                'semantic_margin',
                self._build_semantic_margin_matrix(semantic_table,
                                                   self.semantic_gamma,
                                                   self.semantic_topk))
            self.register_buffer('text_prototypes', semantic_table.clone())
            self.register_buffer('visual_prototypes',
                                 semantic_table.new_zeros(semantic_table.shape))
            self.register_buffer('prototype_counts',
                                 semantic_table.new_zeros(semantic_table.size(0)))
        else:
            self.semantic_projector = None
            self.register_buffer('semantic_embeddings', torch.zeros(0, 0))
            self.register_buffer('semantic_margin', torch.zeros(0, 0))
            self.register_buffer('text_prototypes', torch.zeros(0, 0))
            self.register_buffer('visual_prototypes', torch.zeros(0, 0))
            self.register_buffer('prototype_counts', torch.zeros(0))

    @staticmethod
    def _default_attr_vocab():
        return (
            'air', 'airport', 'water', 'port', 'road', 'rail', 'bridge',
            'overpass', 'service', 'toll', 'sports', 'ball_field', 'court',
            'track', 'stadium', 'industrial', 'storage', 'chimney', 'energy',
            'vehicle', 'ship', 'train', 'infrastructure', 'large_area',
            'open_area', 'vertical', 'transport', 'manmade', 'static',
            'dynamic', 'land', 'waterfront')

    @classmethod
    def _default_attr_map(cls):
        return {
            'airplane': {
                'air': 1.0,
                'transport': 1.0,
                'dynamic': 1.0,
                'manmade': 0.8,
            },
            'airport': {
                'air': 1.0,
                'airport': 1.0,
                'transport': 1.0,
                'infrastructure': 1.0,
                'large_area': 0.9,
                'open_area': 0.7,
                'land': 0.7,
                'static': 0.8,
                'manmade': 0.8,
            },
            'dam': {
                'water': 1.0,
                'infrastructure': 1.0,
                'large_area': 0.8,
                'manmade': 0.8,
                'static': 0.8,
                'land': 0.5,
            },
            'Expressway-Service-area': {
                'road': 1.0,
                'service': 1.0,
                'transport': 0.9,
                'infrastructure': 0.9,
                'large_area': 0.6,
                'land': 0.8,
                'manmade': 0.8,
                'static': 0.7,
            },
            'Expressway-toll-station': {
                'road': 1.0,
                'toll': 1.0,
                'transport': 0.9,
                'infrastructure': 0.9,
                'land': 0.8,
                'manmade': 0.8,
                'static': 0.8,
            },
            'golffield': {
                'sports': 1.0,
                'ball_field': 0.7,
                'open_area': 1.0,
                'large_area': 0.9,
                'land': 0.9,
                'static': 0.8,
            },
            'groundtrackfield': {
                'sports': 1.0,
                'track': 1.0,
                'open_area': 0.9,
                'large_area': 0.7,
                'land': 0.8,
                'static': 0.8,
            },
            'ground-track-field': {
                'sports': 1.0,
                'track': 1.0,
                'open_area': 0.9,
                'large_area': 0.7,
                'land': 0.8,
                'static': 0.8,
            },
            'harbor': {
                'water': 1.0,
                'port': 1.0,
                'ship': 0.8,
                'infrastructure': 0.7,
                'waterfront': 1.0,
                'manmade': 0.6,
                'static': 0.8,
            },
            'overpass': {
                'road': 0.9,
                'bridge': 0.8,
                'overpass': 1.0,
                'transport': 0.8,
                'infrastructure': 0.9,
                'land': 0.8,
                'manmade': 0.8,
                'static': 0.8,
            },
            'stadium': {
                'sports': 1.0,
                'stadium': 1.0,
                'large_area': 0.8,
                'open_area': 0.5,
                'land': 0.7,
                'manmade': 0.8,
                'static': 0.8,
            },
            'storagetank': {
                'industrial': 0.8,
                'storage': 1.0,
                'energy': 0.7,
                'manmade': 0.8,
                'static': 0.9,
                'vertical': 0.4,
            },
            'storage-tank': {
                'industrial': 0.8,
                'storage': 1.0,
                'energy': 0.7,
                'manmade': 0.8,
                'static': 0.9,
                'vertical': 0.4,
            },
            'tenniscourt': {
                'sports': 1.0,
                'court': 1.0,
                'open_area': 0.7,
                'land': 0.8,
                'manmade': 0.8,
                'static': 0.8,
            },
            'tennis-court': {
                'sports': 1.0,
                'court': 1.0,
                'open_area': 0.7,
                'land': 0.8,
                'manmade': 0.8,
                'static': 0.8,
            },
            'trainstation': {
                'rail': 1.0,
                'train': 1.0,
                'transport': 1.0,
                'infrastructure': 0.9,
                'land': 0.8,
                'manmade': 0.8,
                'static': 0.8,
            },
            'vehicle': {
                'vehicle': 1.0,
                'transport': 1.0,
                'dynamic': 0.9,
                'land': 0.8,
                'manmade': 0.7,
            },
            'windmill': {
                'energy': 1.0,
                'vertical': 1.0,
                'manmade': 0.8,
                'static': 0.9,
                'land': 0.8,
                'open_area': 0.6,
            },
            'baseballfield': {
                'sports': 1.0,
                'ball_field': 1.0,
                'open_area': 0.9,
                'large_area': 0.8,
                'land': 0.8,
                'static': 0.8,
            },
            'baseball-diamond': {
                'sports': 1.0,
                'ball_field': 1.0,
                'open_area': 0.9,
                'large_area': 0.8,
                'land': 0.8,
                'static': 0.8,
            },
            'basketballcourt': {
                'sports': 1.0,
                'court': 1.0,
                'open_area': 0.6,
                'land': 0.8,
                'manmade': 0.8,
                'static': 0.8,
            },
            'basketball-court': {
                'sports': 1.0,
                'court': 1.0,
                'open_area': 0.6,
                'land': 0.8,
                'manmade': 0.8,
                'static': 0.8,
            },
            'bridge': {
                'bridge': 1.0,
                'road': 0.6,
                'waterfront': 0.5,
                'transport': 0.7,
                'infrastructure': 1.0,
                'manmade': 0.8,
                'static': 0.8,
            },
            'chimney': {
                'chimney': 1.0,
                'industrial': 0.9,
                'vertical': 1.0,
                'manmade': 0.8,
                'static': 0.9,
            },
            'ship': {
                'ship': 1.0,
                'water': 1.0,
                'transport': 0.8,
                'dynamic': 0.8,
                'waterfront': 0.8,
                'manmade': 0.7,
            },
        }

    @classmethod
    def _build_handcrafted_semantic_embeddings(cls, class_names, semantic_dim):
        vocab = cls._default_attr_vocab()
        attr_to_idx = {name: idx for idx, name in enumerate(vocab)}
        attr_map = cls._default_attr_map()
        embeddings = []
        for class_name in class_names:
            vec = torch.zeros(len(vocab), dtype=torch.float32)
            for attr_name, value in attr_map.get(class_name, {}).items():
                if attr_name in attr_to_idx:
                    vec[attr_to_idx[attr_name]] = float(value)
            if torch.sum(torch.abs(vec)) == 0:
                vec[-1] = 1.0
            embeddings.append(vec)
        semantic = torch.stack(embeddings, dim=0)
        if semantic.size(1) > semantic_dim:
            semantic = semantic[:, :semantic_dim]
        elif semantic.size(1) < semantic_dim:
            pad = semantic.new_zeros(semantic.size(0),
                                     semantic_dim - semantic.size(1))
            semantic = torch.cat([semantic, pad], dim=1)
        return F.normalize(semantic, p=2, dim=-1)

    def _init_semantic_embeddings(self,
                                  class_names=None,
                                  semantic_embeddings=None,
                                  semantic_dim=32):
        total_fg_classes = self.num_classes + self.num_novel_classes
        if semantic_embeddings is not None:
            semantic = torch.tensor(semantic_embeddings, dtype=torch.float32)
        elif class_names is not None:
            semantic = self._build_handcrafted_semantic_embeddings(
                class_names, semantic_dim)
        else:
            raise ValueError('Semantic auxiliary loss requires class_names '
                             'or semantic_embeddings.')

        if semantic.dim() != 2:
            raise ValueError('semantic_embeddings must be a 2D matrix.')
        if semantic.size(0) != total_fg_classes:
            raise ValueError(f'semantic_embeddings rows={semantic.size(0)} '
                             f'but expected {total_fg_classes}.')
        if semantic.size(1) > semantic_dim:
            semantic = semantic[:, :semantic_dim]
        elif semantic.size(1) < semantic_dim:
            pad = semantic.new_zeros(semantic.size(0),
                                     semantic_dim - semantic.size(1))
            semantic = torch.cat([semantic, pad], dim=1)
        semantic = F.normalize(semantic, p=2, dim=-1)
        return semantic

    @staticmethod
    def _build_semantic_margin_matrix(semantic_embeddings, gamma, topk):
        sim = semantic_embeddings @ semantic_embeddings.t()
        sim.fill_diagonal_(0.0)
        margin = torch.where(sim - gamma > 0, sim, torch.zeros_like(sim))
        if topk > 0 and topk < margin.size(1):
            topk_vals, topk_inds = torch.topk(margin, k=topk, dim=1)
            filtered = torch.zeros_like(margin)
            filtered.scatter_(1, topk_inds, topk_vals)
            margin = filtered
        return margin

    def _prototype_class_mask(self, labels, label_weights):
        total_fg_classes = self.num_classes + self.num_novel_classes
        fg_mask = ((labels >= 0) & (labels < total_fg_classes)
                   & (label_weights > 0))
        if self.prototype_loss_on_novel_only:
            fg_mask = fg_mask & (labels >= self.num_classes)
        return fg_mask

    def _prototype_warmup_ramp(self, dtype=torch.float32, device=None):
        """Smooth ramp for prototype-related terms over the first
        ``prototype_warmup_iters`` training iterations.

        Returns a scalar tensor in [0, 1]. Used to suppress prototype
        calibration and background-suppression early in training, when the
        EMA visual prototypes have not yet converged to a stable state.

        Outside training, or once the warmup window has elapsed, the ramp is
        always 1.0 so inference is unaffected.
        """
        if device is None:
            device = self.proto_step.device
        if (not self.training) or self.prototype_warmup_iters <= 0:
            return torch.ones((), dtype=dtype, device=device)
        step = float(self.proto_step.item())
        ratio = min(max(step / float(self.prototype_warmup_iters), 0.0), 1.0)
        return torch.tensor(ratio, dtype=dtype, device=device)

    def _compute_prototype_alpha(self, targets, dtype):
        if targets.numel() == 0:
            return self.prototype_counts.new_zeros((0, 1), dtype=dtype)
        if not self.use_adaptive_prototype_alpha:
            return self.prototype_counts.new_full(
                (targets.size(0), 1), self.fixed_prototype_alpha, dtype=dtype)
        counts = self.prototype_counts[targets].to(dtype=dtype)
        alpha = counts / (counts + self.prototype_count_temperature)
        alpha = alpha.clamp(0.0, 1.0).unsqueeze(1)
        return alpha

    def _fused_prototypes_for_targets(self, targets, dtype):
        text_proto = self.text_prototypes[targets].to(dtype=dtype)
        visual_proto = self.visual_prototypes[targets].to(dtype=dtype)
        alpha = self._compute_prototype_alpha(targets, dtype)
        fused = alpha * visual_proto + (1.0 - alpha) * text_proto
        fused = F.normalize(fused, p=2, dim=-1)
        return fused, alpha

    def _semantic_aux_losses(self, semantic_feat, labels, label_weights):
        if (not self.use_semantic_aux) or semantic_feat is None:
            return {}
        total_fg_classes = self.num_classes + self.num_novel_classes
        fg_mask = ((labels >= 0) & (labels < total_fg_classes)
                   & (label_weights > 0))
        if not fg_mask.any():
            zero = semantic_feat.sum() * 0.0
            return dict(loss_sem_align=zero, loss_sem_margin=zero)

        feat = semantic_feat[fg_mask]
        targets = labels[fg_mask]
        semantic_table = self.semantic_embeddings.to(dtype=feat.dtype)
        logits = self.semantic_logit_scale * feat.matmul(semantic_table.t())
        loss_align = F.cross_entropy(logits, targets)

        semantic_margin = self.semantic_margin[targets].to(dtype=logits.dtype)
        margin_logits = logits + semantic_margin
        loss_margin = F.cross_entropy(margin_logits, targets)

        return dict(
            loss_sem_align=self.semantic_align_weight * loss_align,
            loss_sem_margin=self.semantic_margin_weight * loss_margin)

    @torch.no_grad()
    def _update_visual_prototypes(self, semantic_feat, labels, label_weights):
        if (not self.use_vl_prototypes) or semantic_feat is None:
            return
        fg_mask = self._prototype_class_mask(labels, label_weights)
        if not fg_mask.any():
            return
        feat = semantic_feat[fg_mask].detach()
        feat = F.normalize(feat, p=2, dim=-1)
        targets = labels[fg_mask]
        unique_targets = torch.unique(targets)
        for cls_idx in unique_targets.tolist():
            cls_mask = targets == cls_idx
            cls_feat = feat[cls_mask].mean(dim=0)
            cls_feat = F.normalize(cls_feat.unsqueeze(0), p=2, dim=-1).squeeze(0)
            if self.prototype_counts[cls_idx] <= 0:
                updated = cls_feat
            else:
                updated = self.prototype_momentum * self.visual_prototypes[cls_idx] + \
                    (1.0 - self.prototype_momentum) * cls_feat
                updated = F.normalize(updated.unsqueeze(0), p=2, dim=-1).squeeze(0)
            self.visual_prototypes[cls_idx].copy_(updated)
            self.prototype_counts[cls_idx] += float(cls_mask.sum().item())

    def _prototype_aux_losses(self, semantic_feat, labels, label_weights):
        if (not self.use_vl_prototypes) or semantic_feat is None:
            return {}
        fg_mask = self._prototype_class_mask(labels, label_weights)
        if not fg_mask.any():
            zero = semantic_feat.sum() * 0.0
            return dict(loss_proto_align=zero, loss_proto_sep=zero)

        feat = semantic_feat[fg_mask]
        feat = F.normalize(feat, p=2, dim=-1)
        targets = labels[fg_mask]
        fused_targets, _ = self._fused_prototypes_for_targets(targets,
                                                              feat.dtype)
        align_sims = torch.sum(feat * fused_targets, dim=-1)
        loss_align = (1.0 - align_sims).mean()

        fused_all = self.visual_prototypes.to(dtype=feat.dtype)
        fused_all = fused_all.clone()
        if fused_all.numel() == 0:
            zero = semantic_feat.sum() * 0.0
            return dict(loss_proto_align=zero, loss_proto_sep=zero)
        active = self.prototype_counts > 0
        if active.any():
            alpha_all = self._compute_prototype_alpha(
                torch.arange(self.prototype_counts.size(0),
                             device=self.prototype_counts.device), feat.dtype)
            text_proto = self.text_prototypes.to(dtype=feat.dtype)
            fused_all = alpha_all * self.visual_prototypes.to(dtype=feat.dtype) + \
                (1.0 - alpha_all) * text_proto
            fused_all = F.normalize(fused_all, p=2, dim=-1)
        else:
            fused_all = F.normalize(self.text_prototypes.to(dtype=feat.dtype),
                                    p=2,
                                    dim=-1)

        sims = feat.matmul(fused_all.t())
        pos = torch.gather(sims, 1, targets.unsqueeze(1)).squeeze(1)
        neg_mask = torch.ones_like(sims, dtype=torch.bool)
        neg_mask.scatter_(1, targets.unsqueeze(1), False)
        neg = sims.masked_fill(~neg_mask, -1.0)
        hardest_neg = neg.max(dim=1).values
        # Confusion-aware adaptive margin:
        # for each fg sample, look at the text-prototype similarity between
        # its target class and every other class. Classes whose text prototype
        # is close to the target's text prototype are "confusable" and get a
        # larger margin contribution. The base margin is preserved when the
        # `semantic_margin` row is all-zero.
        margin = torch.full_like(pos, self.prototype_margin)
        if self.prototype_sep_use_semantic_weight and \
                self.semantic_margin.numel() > 0:
            margin_row = self.semantic_margin[targets].to(dtype=pos.dtype)
            row_max = margin_row.max(dim=1).values
            # rescale to [0, 1] then add as a fractional bonus to the margin
            margin = margin + self.prototype_margin * row_max.clamp(0.0, 1.0)
        loss_sep = F.relu(margin + hardest_neg - pos).mean()

        ramp = self._prototype_warmup_ramp(dtype=feat.dtype, device=feat.device)
        return dict(
            loss_proto_align=self.prototype_align_weight * ramp * loss_align,
            loss_proto_sep=self.prototype_sep_weight * ramp * loss_sep)

    def _ooc_context_suppression_loss(self, cls_score_novel, labels,
                                      label_weights, loss_name='loss_ocs'):
        """Suppress novel-class activations on background RoIs.

        Negative RoIs are treated as local context-only regions. OCS penalizes
        novel-branch probabilities above a threshold during training and leaves
        inference unchanged.
        """
        if self.ocs_loss_weight <= 0 or cls_score_novel is None:
            return {}
        bg_class_ind = self.num_classes + self.num_novel_classes
        bg_mask = (labels == bg_class_ind) & (label_weights > 0)
        if not bg_mask.any():
            zero = cls_score_novel.sum() * 0.0
            return {loss_name: zero}
        novel_probs = F.softmax(cls_score_novel[bg_mask], dim=-1)[:, :self.num_novel_classes]
        max_novel_prob = novel_probs.max(dim=1).values
        excess = F.relu(max_novel_prob - self.ocs_score_threshold)
        loss = excess.pow(2).mean()
        if self.ocs_warmup_iters > 0 and self.training:
            step = float(self.proto_step.item())
            ratio = min(max(step / float(self.ocs_warmup_iters), 0.0), 1.0)
            ramp = torch.tensor(
                ratio, dtype=cls_score_novel.dtype, device=cls_score_novel.device)
        else:
            ramp = cls_score_novel.new_tensor(1.0)
        return {loss_name: self.ocs_loss_weight * ramp * loss}

    def _prototype_distill_loss(self, semantic_feat, cls_score_novel,
                                labels, label_weights):
        """Prototype-Guided Classifier Distillation (PGCD) loss.

        Instead of modifying inference-time scores (which injects noise from
        unstable prototypes), this loss guides the novel classifier to produce
        predictions that are structurally consistent with the prototype
        similarity space during training.

        For each novel-class foreground sample:
          - Compute prototype-based soft target: softmax(feat @ proto^T / tau)
          - Compute classifier prediction: softmax(cls_score_novel[:, :N] / tau)
          - Minimize KL divergence between them

        This teaches fc_cls_novel to respect the inter-class structure encoded
        in the text/visual prototypes without altering inference logic.
        """
        if (not self.use_vl_prototypes) or semantic_feat is None:
            return {}
        if cls_score_novel is None:
            return {}

        total_fg = self.num_classes + self.num_novel_classes
        fg_mask = ((labels >= self.num_classes) & (labels < total_fg)
                   & (label_weights > 0))
        if not fg_mask.any():
            zero = semantic_feat.sum() * 0.0
            return dict(loss_proto_distill=zero)

        feat = F.normalize(semantic_feat[fg_mask], p=2, dim=-1)
        novel_logits = cls_score_novel[fg_mask][:, :self.num_novel_classes]

        # Build prototype-based soft targets
        novel_cls_idx = torch.arange(
            self.num_classes,
            self.num_classes + self.num_novel_classes,
            device=feat.device, dtype=torch.long)
        # Use text prototypes directly (stable, from CLIP) rather than
        # potentially unstable fused prototypes
        text_proto = F.normalize(
            self.text_prototypes[novel_cls_idx].to(dtype=feat.dtype), p=2, dim=-1)
        proto_logits = feat.matmul(text_proto.t()) * self.prototype_logit_scale

        # Soft targets from prototype space
        proto_soft = F.softmax(proto_logits, dim=-1)
        # Classifier log-probs
        cls_log_probs = F.log_softmax(novel_logits / self.novel_score_temperature,
                                      dim=-1)

        # KL(proto_soft || cls_probs) - teaches classifier to match prototype structure
        loss_distill = F.kl_div(cls_log_probs, proto_soft, reduction='batchmean')

        ramp = self._prototype_warmup_ramp(dtype=feat.dtype, device=feat.device)
        return dict(
            loss_proto_distill=self.prototype_calibration_weight * ramp * loss_distill)

    @torch.no_grad()
    def _update_novel_prototypes(self, x_cls, labels, label_weights):
        """Update running mean of novel class features for weight imprinting."""
        if not self.use_hallucinator:
            return
        total_fg = self.num_classes + self.num_novel_classes
        fg_mask = ((labels >= self.num_classes) & (labels < total_fg)
                   & (label_weights > 0))
        if not fg_mask.any():
            return
        feat = x_cls[fg_mask].detach()
        targets = labels[fg_mask] - self.num_classes  # 0-indexed novel class
        for cls_idx in torch.unique(targets).tolist():
            cls_mask = targets == cls_idx
            cls_feat = feat[cls_mask].mean(dim=0)
            if self.novel_class_counts[cls_idx] <= 0:
                self.novel_class_prototypes[cls_idx] = cls_feat
            else:
                self.novel_class_prototypes[cls_idx] = (
                    self.imprint_momentum * self.novel_class_prototypes[cls_idx]
                    + (1.0 - self.imprint_momentum) * cls_feat)
            self.novel_class_counts[cls_idx] += float(cls_mask.sum().item())

    def _hallucinator_loss(self, x_cls, labels, label_weights):
        """Feature hallucination loss for novel class enhancement.

        Generates hallucinated novel features by adding learned residuals to
        novel prototypes, then trains the novel classifier to correctly
        classify these hallucinated features. This effectively augments the
        feature space with diverse novel-class representations.
        """
        if not self.use_hallucinator:
            return {}
        # Only activate after we have seen some novel samples
        if self.novel_class_counts.sum() <= 0:
            zero = x_cls.sum() * 0.0
            return dict(loss_hallucinate=zero)

        active_mask = self.novel_class_counts > 0
        if not active_mask.any():
            zero = x_cls.sum() * 0.0
            return dict(loss_hallucinate=zero)

        active_idx = torch.where(active_mask)[0]
        num_active = active_idx.size(0)
        # Generate K hallucinated features per active novel class
        K = 4
        device = x_cls.device
        dtype = x_cls.dtype

        protos = self.novel_class_prototypes[active_idx].to(dtype=dtype)
        # Repeat each prototype K times
        protos_rep = protos.unsqueeze(1).expand(-1, K, -1).reshape(-1, self.cls_last_dim)
        # Random noise
        noise = torch.randn(num_active * K, self.hallucinator_noise_dim,
                            device=device, dtype=dtype)
        # Hallucinator input: concat(prototype, noise)
        hall_input = torch.cat([protos_rep, noise], dim=-1)
        # Generate residual
        residual = self.hallucinator(hall_input)
        # Hallucinated features = prototype + residual
        hall_feat = protos_rep + 0.1 * residual  # small residual scale

        # Classify hallucinated features with fc_cls_novel
        hall_logits = self.fc_cls_novel(hall_feat)[:, :self.num_novel_classes]
        # Target labels
        hall_targets = active_idx.unsqueeze(1).expand(-1, K).reshape(-1)

        loss_hall = F.cross_entropy(hall_logits, hall_targets)

        ramp = self._prototype_warmup_ramp(dtype=dtype, device=device)
        return dict(loss_hallucinate=self.hallucinator_weight * ramp * loss_hall)

    @torch.no_grad()
    def _imprint_novel_weights(self):
        """Periodically imprint fc_cls_novel weights from novel prototypes.

        This soft-imprints the classifier weights toward the running mean of
        novel features, preventing the classifier from drifting too far from
        the actual feature distribution when training data is scarce.
        """
        if not self.use_hallucinator:
            return
        if not self.training:
            return
        active_mask = self.novel_class_counts > 0
        if not active_mask.any():
            return
        # Soft imprint: blend current weights with normalized prototypes
        active_idx = torch.where(active_mask)[0]
        protos = self.novel_class_prototypes[active_idx]
        protos_norm = F.normalize(protos, p=2, dim=-1)
        current_w = self.fc_cls_novel.weight.data[active_idx]
        current_norm = current_w.norm(dim=1, keepdim=True).clamp_min(1e-6)
        # Scale prototypes to match current weight magnitude
        target_w = protos_norm * current_norm
        # Soft blend (very gentle, 1% per step)
        self.fc_cls_novel.weight.data[active_idx] = (
            0.99 * current_w + 0.01 * target_w)

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, cfg):
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Args:
            pos_bboxes (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_bboxes (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains gt_boxes for
                all positive samples, has shape (num_pos, 4),
                the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains gt_labels for
                all positive samples, has shape (num_pos, ).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals
            in a single image. Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all
                  proposals, has shape (num_proposals, 4), the
                  last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all
                  proposals, has shape (num_proposals, 4).
        """
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes + self.num_novel_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def _prototype_calibrated_scores(self, cls_score_base, cls_score_novel,
                                     semantic_feat):
        """Merge base and novel branch scores into a unified probability vector.

        This method implements the standard LCC scoring logic without any
        inference-time prototype calibration. Prototype-related losses
        (align, sep, distill) are applied during training only and do not
        alter the inference scoring pipeline.
        """
        if cls_score_base is None or cls_score_novel is None:
            return None
        base_scores = F.softmax(
            cls_score_base / self.base_score_temperature, dim=-1)
        base_fg = base_scores[:, :self.num_classes]
        bg_score = base_scores[:, self.num_classes].clamp_min(1e-12)
        if self.novel_bg_score_power != 1.0:
            bg_score = bg_score.pow(self.novel_bg_score_power)

        novel_scores = F.softmax(
            cls_score_novel / self.novel_score_temperature, dim=-1)
        if self.novel_prob_score_power != 1.0:
            novel_scores = novel_scores.clamp_min(1e-12).pow(
                self.novel_prob_score_power)

        return torch.cat([base_fg, bg_score[:, None] * novel_scores], dim=1)

    def forward(self, x, roi_primitives=None):
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        if self.use_visual_primitives and roi_primitives is not None:
            primitive_embed = self.primitive_encoder(
                roi_primitives.to(dtype=x_cls.dtype))
            x_cls = x_cls + primitive_embed
            x_reg = x_reg + primitive_embed

        if self.use_dropout and self.training:
            x_cls = F.dropout(x_cls, p=self.dropout_ratio)

        cls_score_base = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred_base = self.fc_reg(x_reg) if self.with_reg else None
        cls_score_novel = self.fc_cls_novel(x_cls) if self.with_cls else None
        ocs_aux_score = None
        if self.with_cls and self.use_ocs_aux_branch and self.ocs_context_cls is not None:
            ocs_feat = x_cls.detach() if self.ocs_aux_detach else x_cls
            ocs_aux_score = self.ocs_context_cls(ocs_feat)
        self._cached_ocs_aux_score = ocs_aux_score
        # Store x_cls for hallucinator loss (accessed in loss() method)
        if self.use_hallucinator and self.training:
            self._cached_x_cls = x_cls
        semantic_feat = None
        if self.use_prototype_space:
            semantic_feat = F.normalize(
                self.semantic_projector(x_cls), p=2, dim=-1)

        return cls_score_base, bbox_pred_base, cls_score_novel, semantic_feat

    def _merge_branch_scores(self,
                             cls_score,
                             cls_score_novel,
                             semantic_feat=None):
        if cls_score is None or cls_score_novel is None:
            return None
        return self._prototype_calibrated_scores(cls_score, cls_score_novel,
                                                 semantic_feat)

    @force_fp32(apply_to=('cls_score', 'cls_score_novel', 'bbox_pred',
                          'semantic_feat'))
    def loss(self,
             cls_score,
             cls_score_novel,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None,
             semantic_feat=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    cls_score_novel,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    cls_score_temp = self._merge_branch_scores(
                        cls_score, cls_score_novel, semantic_feat)
                    losses['acc'] = accuracy(cls_score_temp, labels)
                losses.update(
                    self._semantic_aux_losses(semantic_feat, labels,
                                              label_weights))
                losses.update(
                    self._prototype_aux_losses(semantic_feat, labels,
                                               label_weights))
                losses.update(
                    self._prototype_distill_loss(semantic_feat, cls_score_novel,
                                                labels, label_weights))
                ocs_aux_score = getattr(self, '_cached_ocs_aux_score', None)
                if self.use_ocs_aux_branch and ocs_aux_score is not None:
                    losses.update(
                        self._ooc_context_suppression_loss(
                            ocs_aux_score, labels, label_weights,
                            loss_name='loss_ocs_aux'))
                    self._cached_ocs_aux_score = None
                else:
                    losses.update(
                        self._ooc_context_suppression_loss(
                            cls_score_novel, labels, label_weights))
                # Hallucinator uses cached x_cls from forward pass
                x_cls_feat = getattr(self, '_cached_x_cls', None)
                if x_cls_feat is not None:
                    losses.update(
                        self._hallucinator_loss(x_cls_feat, labels, label_weights))
                    self._update_novel_prototypes(x_cls_feat, labels, label_weights)
                    self._imprint_novel_weights()
                    self._cached_x_cls = None  # clear cache
                self._update_visual_prototypes(semantic_feat, labels,
                                               label_weights)
                # Advance prototype warmup counter
                if self.use_vl_prototypes and self.prototype_warmup_iters > 0:
                    self.proto_step += 1
        if bbox_pred is not None:
            bg_class_ind = self.num_classes + self.num_novel_classes
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   cls_score_novel,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None,
                   semantic_feat=None):
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = self._merge_branch_scores(cls_score, cls_score_novel,
                                               semantic_feat)
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            return det_bboxes, det_labels
