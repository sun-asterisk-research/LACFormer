import torch
import torch.nn as nn
from mmcv.cnn import ConvModule


class VisualAttention(nn.Module):
    def __init__(self,
                 channels,
                 with_bottleneck=False):
        super(VisualAttention, self).__init__()
        self.with_bottleneck = with_bottleneck
        # capture local feature
        self.local_attention = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,  # get local feature
            stride=1,
            padding=1,
            groups=channels
        )
        # capture long-range feature
        self.long_attention = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=7,
            stride=1,
            padding=9,
            dilation=3,
            groups=channels
        )
        if self.with_bottleneck:
            self.bottleneck = ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                padding=0
            )

    def forward(self, x):
        _x = x.clone()

        # get local feature
        local_feat = self.local_attention(_x)
        local_bridge = local_feat + _x

        # get long-range feature
        long_feat = self.long_attention(local_bridge)
        long_bridge = long_feat + local_bridge

        attention_weight = x + long_bridge

        return x * attention_weight


class VisualAttentionv2(nn.Module):
    def __init__(self,
                 max_pool=False):
        super(VisualAttentionv2, self).__init__()

        self.max_pool = max_pool
        # capture local feature
        self.local_attention = ConvModule(
            in_channels=2 if self.max_pool else 1,
            out_channels=1,
            kernel_size=3,  # get local feature
            stride=1,
            padding=1
        )
        # capture long-range feature
        self.long_attention = ConvModule(
            in_channels=1,
            out_channels=1,
            kernel_size=7,
            stride=1,
            padding=9,
            dilation=3
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_feat = torch.mean(x, dim=1, keepdim=True)
        if self.max_pool:
            max_feat, _ = torch.max(x, dim=1, keepdim=True)
        feat = avg_feat if not self.max_pool else torch.cat([avg_feat, max_feat], dim=1)

        # get local feature
        local_feat = self.local_attention(feat)
        local_bridge = local_feat + feat

        # get long-range feature
        long_feat = self.long_attention(local_bridge)
        long_bridge = long_feat + local_bridge

        attention_weight = self.sigmoid(x + long_bridge)

        return x * attention_weight
