"""Remote-sensing feature adapter RoI head.

The adapter refines RoI features before they enter the LCC/RVLP-LCC bbox
head. It keeps the detector interface unchanged while moving part of the
innovation from score calibration into feature extraction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import bbox2roi
from mmdet.models.builder import HEADS

from .lcc_roi_head import LCCRoIHead


class ObjectContextGate(nn.Module):
    """Object-aware context calibration gate for adapted RoI features."""

    def __init__(self,
                 channels=256,
                 hidden_channels=128,
                 primitive_dim=8,
                 dilation=2,
                 dropout=0.0):
        super().__init__()
        self.object_branch = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32 if hidden_channels % 32 == 0 else 16,
                         hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, bias=False),
            nn.GroupNorm(32 if channels % 32 == 0 else 16, channels))
        self.context_branch = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False),
            nn.GroupNorm(32 if hidden_channels % 32 == 0 else 16,
                         hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, bias=False),
            nn.GroupNorm(32 if channels % 32 == 0 else 16, channels))
        gate_hidden = max(hidden_channels, 32)
        self.gate_mlp = nn.Sequential(
            nn.Linear(channels * 2 + primitive_dim, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(gate_hidden, channels),
            nn.Sigmoid())

    def forward(self, x, roi_primitives=None):
        obj = self.object_branch(x)
        ctx = self.context_branch(x)
        obj_vec = obj.mean(dim=(2, 3))
        ctx_vec = ctx.mean(dim=(2, 3))
        if roi_primitives is None:
            primitives = obj_vec.new_zeros((obj_vec.size(0), 8))
        else:
            primitives = roi_primitives.to(
                dtype=obj_vec.dtype, device=obj_vec.device)
        gate = self.gate_mlp(torch.cat([obj_vec, ctx_vec, primitives], dim=1))
        gate_map = gate.view(gate.size(0), gate.size(1), 1, 1)
        return obj + gate_map * ctx, gate


class RSFeatureAdapter(nn.Module):
    """Lightweight spatial-channel adapter for remote-sensing RoI features."""

    def __init__(self,
                 channels=256,
                 hidden_channels=128,
                 spatial_kernel=3,
                 dropout=0.0,
                 residual_scale_init=0.2,
                 use_channel_gate=True,
                 use_spatial_gate=True):
        super().__init__()
        padding = spatial_kernel // 2
        self.use_channel_gate = bool(use_channel_gate)
        self.use_spatial_gate = bool(use_spatial_gate)

        self.local_context = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=spatial_kernel,
                padding=padding,
                groups=channels,
                bias=False),
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(32 if hidden_channels % 32 == 0 else 16,
                         hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(float(dropout)),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(32 if channels % 32 == 0 else 16, channels))

        if self.use_channel_gate:
            gate_hidden = max(hidden_channels, 32)
            self.channel_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, gate_hidden, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(gate_hidden, channels, kernel_size=1),
                nn.Sigmoid())
        else:
            self.channel_gate = None

        if self.use_spatial_gate:
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=padding),
                nn.Sigmoid())
        else:
            self.spatial_gate = None

        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init)))

    def forward(self, x):
        residual = self.local_context(x)
        if self.channel_gate is not None:
            residual = residual * self.channel_gate(x)
        if self.spatial_gate is not None:
            avg_map = x.mean(dim=1, keepdim=True)
            max_map = x.max(dim=1, keepdim=True).values
            residual = residual * self.spatial_gate(
                torch.cat([avg_map, max_map], dim=1))
        return x + self.residual_scale.tanh() * residual


@HEADS.register_module()
class RSFAdapterRoIHead(LCCRoIHead):
    """LCC-compatible RoI head with a trainable RoI feature adapter."""

    def __init__(self,
                 adapter_channels=256,
                 adapter_hidden_channels=128,
                 adapter_spatial_kernel=3,
                 adapter_dropout=0.0,
                 adapter_residual_scale_init=0.2,
                 adapter_use_channel_gate=True,
                 adapter_use_spatial_gate=True,
                 adapter_consistency_weight=0.01,
                 use_ocg=False,
                 ocg_hidden_channels=128,
                 ocg_dilation=2,
                 ocg_dropout=0.0,
                 ocg_gate_loss_weight=0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.adapter_consistency_weight = float(adapter_consistency_weight)
        self.use_ocg = bool(use_ocg)
        self.ocg_gate_loss_weight = float(ocg_gate_loss_weight)
        self.ocg_gate = ObjectContextGate(
            channels=int(adapter_channels),
            hidden_channels=int(ocg_hidden_channels),
            primitive_dim=8,
            dilation=int(ocg_dilation),
            dropout=float(ocg_dropout)) if self.use_ocg else None
        self.rsf_adapter = RSFeatureAdapter(
            channels=int(adapter_channels),
            hidden_channels=int(adapter_hidden_channels),
            spatial_kernel=int(adapter_spatial_kernel),
            dropout=float(adapter_dropout),
            residual_scale_init=float(adapter_residual_scale_init),
            use_channel_gate=adapter_use_channel_gate,
            use_spatial_gate=adapter_use_spatial_gate)

    def _adapter_consistency_loss(self, raw_feats, adapted_feats):
        if self.adapter_consistency_weight <= 0:
            return None
        raw_vec = F.normalize(raw_feats.mean(dim=(2, 3)).detach(), dim=-1)
        adapted_vec = F.normalize(adapted_feats.mean(dim=(2, 3)), dim=-1)
        return F.mse_loss(adapted_vec, raw_vec) * self.adapter_consistency_weight

    def _ocg_gate_loss(self, gate, labels, label_weights):
        if self.ocg_gate_loss_weight <= 0 or gate is None:
            return None
        bg_class_ind = self.bbox_head.num_classes + self.bbox_head.num_novel_classes
        bg_mask = (labels == bg_class_ind) & (label_weights > 0)
        if not bg_mask.any():
            return gate.sum() * 0.0
        return gate[bg_mask].mean() * self.ocg_gate_loss_weight

    def _bbox_forward(self, x, rois, img_metas=None):
        raw_bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            raw_bbox_feats = self.shared_head(raw_bbox_feats)

        bbox_feats = self.rsf_adapter(raw_bbox_feats)
        roi_primitives = self._build_roi_primitives(rois, img_metas)
        ocg_gate = None
        if self.ocg_gate is not None:
            bbox_feats, ocg_gate = self.ocg_gate(bbox_feats, roi_primitives)
        cls_score, bbox_pred, cls_score_novel, semantic_feat = self.bbox_head(
            bbox_feats, roi_primitives=roi_primitives)

        return dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            bbox_feats=bbox_feats,
            raw_bbox_feats=raw_bbox_feats,
            cls_score_novel=cls_score_novel,
            semantic_feat=semantic_feat,
            loss_rsf_adapter=self._adapter_consistency_loss(
                raw_bbox_feats, bbox_feats),
            ocg_gate=ocg_gate,
            rois=rois)

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois, img_metas)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(
            bbox_results['cls_score'],
            bbox_results['cls_score_novel'],
            bbox_results['bbox_pred'],
            rois,
            *bbox_targets,
            semantic_feat=bbox_results.get('semantic_feat', None))

        if bbox_results.get('loss_rsf_adapter', None) is not None:
            loss_bbox['loss_rsf_adapter'] = bbox_results['loss_rsf_adapter']
        loss_ocg_gate = self._ocg_gate_loss(
            bbox_results.get('ocg_gate', None), bbox_targets[0],
            bbox_targets[1])
        if loss_ocg_gate is not None:
            loss_bbox['loss_ocg_gate'] = loss_ocg_gate

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results
