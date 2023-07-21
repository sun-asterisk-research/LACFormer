# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM
from mmseg.models.utils import LayerAttention


class AdaptiveWeightSum(nn.Module):
    def __init__(self, in_channels, groups, la_down_rate=8):
        super().__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.avg_attention = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels // la_down_rate, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.in_channels // la_down_rate,
                self.groups,
                1
            ),
            nn.Sigmoid()
        )

        self.max_attention = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels // la_down_rate, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.in_channels // la_down_rate,
                self.groups,
                1
            ),
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        b, c, h, w = x1.shape

        avg_feat_1 = nn.AdaptiveAvgPool2d(1)(x1)
        avg_feat_2 = nn.AdaptiveAvgPool2d(1)(x2)

        max_feat_1 = nn.AdaptiveMaxPool2d(1)(x1)
        max_feat_2 = nn.AdaptiveMaxPool2d(1)(x2)

        max_feat = torch.cat([max_feat_1, max_feat_2], dim=1)
        avg_feat = torch.cat([avg_feat_1, avg_feat_2], dim=1)

        avg_weight = self.avg_attention(avg_feat)
        max_weight = self.max_attention(max_feat)
        weight = self.sigmoid(avg_weight + max_weight)
        weight = weight.view(b, self.groups, 1, 1, 1)

        out = weight[:, 0] * x1 + weight[:, 1] * x2

        return out


@HEADS.register_module()
class UperLRCHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(UperLRCHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                kernel_size=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, inplace=False)

            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                kernel_size=3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck_1 = ConvModule(
            len(self.in_channels) * self.channels * 2,
            len(self.in_channels) * self.channels,
            kernel_size=3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        self.fpn_bottleneck_2 = ConvModule(
            len(self.in_channels) * self.channels,
            int(len(self.in_channels) * self.channels * 0.5),
            kernel_size=3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        self.fpn_bottleneck_3 = ConvModule(
            int(len(self.in_channels) * self.channels * 0.5),
            int(len(self.in_channels) * self.channels * 0.25),
            kernel_size=3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        self.fpn_bottleneck_edge1 = ConvModule(
            len(self.in_channels) * self.channels,
            1,
            kernel_size=3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        self.layer_attn = LayerAttention(
            self.channels * len(self.in_channels),
            groups=len(self.in_channels)
        )

        self.sum = AdaptiveWeightSum(
            self.channels * len(self.in_channels) * 2,
            groups=len(self.in_channels)
        )

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))
        return self.psp_forward(inputs)
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)

        ## semantic attention
        fpn_outs_semantic = self.layer_attn(fpn_outs)

        ## edge attention
        fpn_outs_edge = self.fpn_bottleneck_edge1(fpn_outs)
        fpn_outs_edge = -1 * (torch.sigmoid(fpn_outs_edge)) + 1
        fpn_outs_edge = fpn_outs_edge.expand(-1, self.channels * 4, -1, -1).mul(fpn_outs)

        feats = torch.cat([fpn_outs_edge, fpn_outs_semantic], dim=1)
        # feats = fpn_outs_edge + fpn_outs_semantic
        # feats = self.sum(fpn_outs_edge, fpn_outs_semantic)
        feats = self.fpn_bottleneck_1(feats)
        feats = self.fpn_bottleneck_2(feats)
        feats = self.fpn_bottleneck_3(feats)

        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
