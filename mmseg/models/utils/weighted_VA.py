import torch
import torch.nn as nn
from mmcv.cnn import ConvModule


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


class wVisualAttention(nn.Module):
    def __init__(self,
                 channels,
                 with_bottleneck=False):
        super(wVisualAttention, self).__init__()
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
        self.adaptivesum_local = AdaptiveWeightSum(in_channels=channels, group=2)
        self.adaptivesum_long = AdaptiveWeightSum(in_channels=channels, group=2)
        self.adaptivesum_total = AdaptiveWeightSum(in_channels=channels, group=2)

    def forward(self, x):
        _x = x.clone()

        # get local feature
        local_feat = self.local_attention(_x)
        local_bridge = self.adaptivesum_local(local_feat, _x)

        # get long-range feature
        long_feat = self.long_attention(local_bridge)
        long_bridge = self.adaptivesum_long(long_feat, local_bridge)

        attention_weight = self.adaptivesum_total(x, long_bridge)

        return x * attention_weight