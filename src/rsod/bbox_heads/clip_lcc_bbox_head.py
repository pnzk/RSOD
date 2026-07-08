"""CLIP-enhanced LCC Bbox Head.

Adds CLIP text embeddings as frozen class prototypes for auxiliary
contrastive alignment during training. At inference, CLIP similarity
provides an additional scoring signal blended with LCC scores.

Key: the CLIP model is frozen and only the projection layer is trained.
This prevents catastrophic forgetting and keeps training stable.
"""
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the optional OpenAI CLIP source path.
# Set CLIP_ROOT to a local clone of https://github.com/openai/CLIP when needed.
_clip_root = os.environ.get('CLIP_ROOT')
if _clip_root and os.path.isdir(_clip_root):
    sys.path.insert(0, _clip_root)

from mmdet.models.builder import HEADS
from .lcc_bbox_head import LCCBoxHead


@HEADS.register_module()
class CLIPLCCBoxHead(LCCBoxHead):
    """LCC Box Head with CLIP text prototype alignment.

    Adds:
    1. Frozen CLIP text embeddings as class prototypes
    2. A learnable projection from RoI features to CLIP space
    3. Contrastive alignment loss during training
    4. CLIP-based score calibration at inference

    Args:
        clip_model_path (str): Path to CLIP ViT-B/32 weights.
        clip_proj_dim (int): CLIP embedding dimension (512 for ViT-B/32).
        clip_align_weight (float): Weight for CLIP alignment loss.
        clip_score_weight (float): Weight for CLIP score at inference.
        clip_temperature (float): Temperature for CLIP similarity.
        class_prompts (list[str]): Text prompts for each class.
    """

    def __init__(self,
                 clip_model_path=None,
                 clip_proj_dim=512,
                 clip_align_weight=0.5,
                 clip_score_weight=0.2,
                 clip_temperature=10.0,
                 class_prompts=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.clip_proj_dim = clip_proj_dim
        self.clip_align_weight = clip_align_weight
        self.clip_score_weight = clip_score_weight
        self.clip_temperature = clip_temperature

        if clip_model_path is None:
            clip_model_path = os.environ.get('CLIP_MODEL_PATH', 'ViT-B/32')

        # Learnable projection: RoI features (1024) -> CLIP space (512)
        self.clip_projector = nn.Sequential(
            nn.Linear(self.cls_last_dim, clip_proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(clip_proj_dim, clip_proj_dim),
        )

        # Load CLIP text embeddings (frozen)
        self._load_clip_text_embeddings(clip_model_path, class_prompts)

    def _load_clip_text_embeddings(self, model_path, class_prompts):
        """Load CLIP and pre-compute frozen text embeddings."""
        try:
            import clip as clip_module
            device = 'cpu'  # Load on CPU, will move to GPU with model
            model, _ = clip_module.load(model_path, device=device, jit=False)

            if class_prompts is None:
                class_prompts = self._default_dior_prompts()

            tokens = clip_module.tokenize(class_prompts)
            with torch.no_grad():
                text_features = model.encode_text(tokens)
                text_features = F.normalize(text_features.float(), p=2, dim=-1)

            # Register as buffer (frozen, moves with model to GPU)
            self.register_buffer('clip_text_prototypes', text_features)
            print(f'[CLIP-LCC] Loaded {text_features.shape[0]} text prototypes')
            del model  # Free CLIP model memory
        except Exception as e:
            print(f'[CLIP-LCC] Failed to load CLIP: {e}')
            # Fallback: random prototypes (will not help but won't crash)
            num_cls = self.num_classes + self.num_novel_classes
            self.register_buffer(
                'clip_text_prototypes',
                torch.randn(num_cls, self.clip_proj_dim))

    @staticmethod
    def _default_dior_prompts():
        """Default DIOR split1 prompts (15 base + 5 novel)."""
        return [
            'a satellite photo of airplanes on the ground',
            'a satellite photo of an airport with runways',
            'a satellite photo of a dam across a river',
            'a satellite photo of an expressway service area',
            'a satellite photo of an expressway toll station',
            'a satellite photo of a golf course',
            'a satellite photo of a ground track field',
            'a satellite photo of a harbor with docks',
            'a satellite photo of a road overpass',
            'a satellite photo of a sports stadium',
            'a satellite photo of storage tanks',
            'a satellite photo of tennis courts',
            'a satellite photo of a train station',
            'a satellite photo of vehicles on a road',
            'a satellite photo of wind turbines',
            'a satellite photo of a baseball field',
            'a satellite photo of a basketball court',
            'a satellite photo of a bridge over water',
            'a satellite photo of an industrial chimney',
            'a satellite photo of ships in water',
        ]

    def forward(self, x, roi_primitives=None):
        """Forward with bbox_feats caching for CLIP loss and inference scoring."""
        # Cache raw bbox_feats for both training (loss) and inference (score blending)
        self._cached_bbox_feats = x.clone() if x.dim() <= 2 else x.flatten(1).clone()

        # Standard LCC forward
        cls_score, bbox_pred, cls_score_novel, semantic_feat = \
            super().forward(x, roi_primitives=roi_primitives)

        return cls_score, bbox_pred, cls_score_novel, semantic_feat

    def _get_clip_features(self, x):
        """Project RoI features to CLIP space.

        Args:
            x: raw RoI features [N, C, H, W] or features after shared_fcs [N, D]
        """
        if x.dim() > 2:
            x = x.flatten(1)
        # Run through shared_fcs (same as in forward)
        for fc in self.shared_fcs:
            x = self.relu(fc(x))
        # Project to CLIP space
        clip_feat = self.clip_projector(x)
        clip_feat = F.normalize(clip_feat, p=2, dim=-1)
        return clip_feat

    def loss(self, cls_score, cls_score_novel, bbox_pred, rois,
             labels, label_weights, bbox_targets, bbox_weights,
             reduction_override=None, semantic_feat=None,
             bbox_feats=None):
        """Loss with CLIP alignment added."""
        # Standard LCC loss
        losses = super().loss(
            cls_score, cls_score_novel, bbox_pred, rois,
            labels, label_weights, bbox_targets, bbox_weights,
            reduction_override=reduction_override,
            semantic_feat=semantic_feat)

        # CLIP alignment loss 鈥?use cached bbox_feats from forward
        if self.clip_align_weight > 0:
            cached_feats = getattr(self, '_cached_bbox_feats', None)
            if cached_feats is not None:
                clip_loss = self._clip_alignment_loss(
                    cached_feats, labels, label_weights)
                losses['loss_clip_align'] = clip_loss
                self._cached_bbox_feats = None

        return losses

    def _clip_alignment_loss(self, bbox_feats, labels, label_weights):
        """Contrastive alignment between RoI features and CLIP text prototypes.

        Pulls features toward their class's text prototype and pushes away
        from other classes' prototypes.
        """
        num_total = self.num_classes + self.num_novel_classes
        fg_mask = (labels >= 0) & (labels < num_total) & (label_weights > 0)
        if not fg_mask.any():
            return bbox_feats.new_zeros(1)

        # Get CLIP-projected features for foreground proposals
        clip_feat = self._get_clip_features(bbox_feats[fg_mask])
        fg_labels = labels[fg_mask]

        # Compute similarity with all text prototypes
        text_protos = self.clip_text_prototypes.to(dtype=clip_feat.dtype)
        sim = clip_feat @ text_protos.t() * self.clip_temperature  # [N, C]

        # Cross-entropy loss (align features with their class prototype)
        loss = F.cross_entropy(sim, fg_labels)
        return self.clip_align_weight * loss

    def get_bboxes(self, rois, cls_score, cls_score_novel, bbox_pred,
                   img_shape, scale_factor, rescale=False, cfg=None,
                   semantic_feat=None, bbox_feats=None):
        """Get bboxes with CLIP score blending at inference."""
        # Standard LCC score merging
        scores = self._merge_branch_scores(cls_score, cls_score_novel, semantic_feat)

        # Blend with CLIP similarity using cached features
        if not self.training and self.clip_score_weight > 0:
            cached_feats = getattr(self, '_cached_bbox_feats', None)
            if cached_feats is not None:
                clip_feat = self._get_clip_features(cached_feats)
                text_protos = self.clip_text_prototypes.to(dtype=clip_feat.dtype)
                sim = clip_feat @ text_protos.t() * self.clip_temperature
                clip_probs = F.softmax(sim, dim=-1)

                # Blend: only for foreground classes
                num_fg = scores.size(1) - 1
                num_clip = clip_probs.size(1)
                if num_clip <= num_fg:
                    w = self.clip_score_weight
                    scores[:, :num_clip] = (1 - w) * scores[:, :num_clip] + w * clip_probs

        # Decode bboxes
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
            from mmdet.core import multiclass_nms
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)
            return det_bboxes, det_labels

