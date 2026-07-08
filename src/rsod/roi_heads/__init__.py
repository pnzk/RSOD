# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from .bbox_heads import (ContrastiveBBoxHead, CosineSimBBoxHead,
                         MultiRelationBBoxHead)
from .contrastive_roi_head import ContrastiveRoIHead
from .fsdetview_roi_head import FSDetViewRoIHead
from .icpe_roi_head import ICPERoIHead
from .imted_roi_head import imTEDRoIHead
from .lcc_roi_head import LCCRoIHead
from .mmfe_roi_head import MMFERoIHead
from .hppf_roi_head import HPPFRoIHead
from .dgc_roi_head import DGCRoIHead
from .rsf_adapter_roi_head import RSFAdapterRoIHead
from .rsf_adapter_v2_roi_head import RSFAdapterV2RoIHead
from .gcp_roi_head import GCPRoIHead
from .defrcn_lcc_roi_head import DeFRCNLCCRoIHead
from .clip_calibrated_roi_head import CLIPCalibratedRoIHead
from .meta_rcnn_roi_head import MetaRCNNRoIHead
from .multi_relation_roi_head import MultiRelationRoIHead
from .shared_heads import MetaRCNNResLayer
from .simple_meta_roi_head import SimpleMetaRoIHead
from .two_branch_roi_head import TwoBranchRoIHead

__all__ = [
    'CosineSimBBoxHead', 'ContrastiveBBoxHead', 'MultiRelationBBoxHead',
    'ContrastiveRoIHead', 'MultiRelationRoIHead', 'FSDetViewRoIHead',
    'MetaRCNNRoIHead', 'MetaRCNNResLayer', 'TwoBranchRoIHead',
    'SimpleMetaRoIHead', 'ICPERoIHead', 'imTEDRoIHead',
    'LCCRoIHead', 'MMFERoIHead', 'HPPFRoIHead', 'GCPRoIHead', 'DGCRoIHead',
    'RSFAdapterRoIHead', 'RSFAdapterV2RoIHead', 'DeFRCNLCCRoIHead'
]

try:
    from .hcrn_roi_head import HCRNRoIHead
    from .hcrn_roi_head_misc import HCRNRoIHeadV2, HCRNRoIHeadV3

    __all__ += ['HCRNRoIHead', 'HCRNRoIHeadV2', 'HCRNRoIHeadV3']
except ModuleNotFoundError as exc:
    warnings.warn(f'Optional HCRN roi heads were skipped during import: {exc}')

try:
    from .roi_extractors import MultiRoIExtractor

    __all__.append('MultiRoIExtractor')
except ModuleNotFoundError as exc:
    warnings.warn(
        f'Optional multi-level roi extractor was skipped during import: {exc}')
