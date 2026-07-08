# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from .contrastive_bbox_head import ContrastiveBBoxHead
from .cosine_sim_bbox_head import CosineSimBBoxHead
from .dgc_bbox_head import DGCBoxHead
from .gfsdet_kd_bbox_head import DisKDBBoxHead, IncreaseBBoxHead, KDBBoxHead
from .lcc_bbox_head import LCCBoxHead
from .pgam_bbox_head import PGAMBBoxHead
from .gcp_bbox_head import GCPBoxHead
from .clip_lcc_bbox_head import CLIPLCCBoxHead
from .hppf_bbox_head import HPPFBoxHead
from .meta_bbox_head import MetaBBoxHead
from .multi_relation_bbox_head import MultiRelationBBoxHead
from .two_branch_bbox_head import TwoBranchBBoxHead

__all__ = [
    'CosineSimBBoxHead', 'ContrastiveBBoxHead', 'MultiRelationBBoxHead',
    'MetaBBoxHead', 'TwoBranchBBoxHead', 'LCCBoxHead', 'GCPBoxHead',
    'DGCBoxHead', 'CLIPLCCBoxHead', 'PGAMBBoxHead', 'HPPFBoxHead',
    'IncreaseBBoxHead', 'KDBBoxHead', 'DisKDBBoxHead'
]

try:
    from .max_bbox_head import MAEBBoxHead

    __all__.append('MAEBBoxHead')
except ModuleNotFoundError as exc:
    warnings.warn(f'Optional MAE bbox head was skipped during import: {exc}')
