import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import BaseModule
from .. import builder
from mmseg.core import add_prefix
from mmseg.ops import resize
from ..builder import SEGMENTORS


@SEGMENTORS.register_module()
class SunSegmentor(BaseModule):
    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 pretrained=None,
                 init_cfg=None,
                 train_cfg=dict(),
                 test_cfg=dict(mode='whole')):
        super().__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

    @property
    def with_neck(self):
        """bool: whether the segmentor has neck"""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_auxiliary_head(self):
        """bool: whether the segmentor has auxiliary head"""
        return hasattr(self,
                       'auxiliary_head') and self.auxiliary_head is not None

    @property
    def with_decode_head(self):
        """bool: whether the segmentor has decode head"""
        return hasattr(self, 'decode_head') and self.decode_head is not None

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def _forward_auxiliary_head(self, img):
        aux_logits = []
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                aux_logit = aux_head(img)
                aux_logits.append(aux_logit)
        else:
            aux_logit = self.auxiliary_head(img)
            aux_logits.append(aux_logit)

        return aux_logits

    def forward(self, img):
        outs = []
        x = self.extract_feat(img)
        out = self.decode_head(x)
        if isinstance(out, list):
            for _out in out:
                outs.append(resize(
                    input=_out,
                    size=img.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners
                ))
            return outs if self.training else outs[0]
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        outs.append(out)
        if self.with_auxiliary_head and self.training:
            aux_outs = self._forward_auxiliary_head(x)
            for aux_out in aux_outs:
                outs.append(
                    resize(
                        input=aux_out,
                        size=img.shape[2:],
                        mode='bilinear',
                        align_corners=self.align_corners
                    )
                )
        return outs if self.training else outs[0] # return list of tensors if training else tensor of main head