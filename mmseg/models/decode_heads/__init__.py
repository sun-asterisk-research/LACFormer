# Copyright (c) OpenMMLab. All rights reserved.
from .ann_head import ANNHead
from .apc_head import APCHead
from .aspp_head import ASPPHead
from .cc_head import CCHead
from .da_head import DAHead
from .dm_head import DMHead
from .dnl_head import DNLHead
from .dpt_head import DPTHead
from .ema_head import EMAHead
from .enc_head import EncHead
from .fcn_head import FCNHead
from .fpn_head import FPNHead
from .gc_head import GCHead
from .isa_head import ISAHead
from .knet_head import IterativeDecodeHead, KernelUpdateHead, KernelUpdator
from .lraspp_head import LRASPPHead
from .nl_head import NLHead
from .ocr_head import OCRHead
from .point_head import PointHead
from .psa_head import PSAHead
from .psp_head import PSPHead
from .segformer_head import SegformerHead
from .segmenter_mask_head import SegmenterMaskTransformerHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .sep_fcn_head import DepthwiseSeparableFCNHead
from .setr_mla_head import SETRMLAHead
from .setr_up_head import SETRUPHead
from .stdc_head import STDCHead
from .uper_head import UPerHead
from .uper_LRC_head import UperLRCHead
from .mlp_GeSE import MLPGeSEHead, MLPGeSEHead_v2, MLPSEHead
from .mlp_la_head import MLPLAHead
from .mlp_la_ra_head import MLPLARAHead
from .mlp_osa_head import MLP_OSAHead, MLP_OSAHead_v2, MLP_OSAHead_v3, MLP_OSAHead_v4
from .mlp_slow_head import MLPSLowHead
from .mlp_slowcat_head import MLPSLowCatHead
from .mlp_slowcatse_head import MLPSLowCatSEHead
from .mlp_slowcatese_head import MLPSLowCatESEHead
from .mlp_slowcat_la_head import MLPSLowCatLAHead
from .ssformer_head import SSFormerHead
from .uper_headv3 import UPerHeadV3
from .rcfpn_head import RPFNHead
from .drp_head import DRPHead
__all__ = [
    'FCNHead', 'PSPHead', 'ASPPHead', 'PSAHead', 'NLHead', 'GCHead', 'CCHead',
    'UPerHead', 'DepthwiseSeparableASPPHead', 'ANNHead', 'DAHead', 'OCRHead',
    'EncHead', 'DepthwiseSeparableFCNHead', 'FPNHead', 'EMAHead', 'DNLHead',
    'PointHead', 'APCHead', 'DMHead', 'LRASPPHead', 'SETRUPHead',
    'SETRMLAHead', 'DPTHead', 'SETRMLAHead', 'SegmenterMaskTransformerHead',
    'SegformerHead', 'ISAHead', 'STDCHead', 'IterativeDecodeHead',
    'KernelUpdateHead', 'KernelUpdator', 'UperLRCHead',
    'MLPGeSEHead', 'MLPGeSEHead_v2', 'MLPSEHead', 'MLPLAHead',
    'MLPLARAHead', 'MLP_OSAHead', 'MLPSLowHead', 'MLPSLowCatHead',
    'MLPSLowCatSEHead', 'MLPSLowCatESEHead', 'MLPSLowCatLAHead', 'RPFNHead',
    'MLP_OSAHead_v2', 'SSFormerHead', 'MLP_OSAHead_v3', 'MLP_OSAHead_v4', 'UPerHeadV3', 'DRPHead'
]
