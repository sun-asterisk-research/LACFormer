from torch import nn


class LayerAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 groups,
                 la_down_rate=8,
                 act='sigmoid'):
        super(LayerAttention, self).__init__()
        assert act in ['sigmoid', 'softmax'], "Activation for LayerAttention must be 'sigmoid' or 'softmax'"
        self.in_channels = in_channels
        self.groups = groups
        self.layer_attention = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels // la_down_rate, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.in_channels // la_down_rate,
                self.groups,
                1
            ),
            nn.Sigmoid() if act == 'sigmoid' else nn.Softmax(dim=1) # channel softmax
        )

    def forward(self, x):
        b, c, h, w = x.shape

        avg_feat = nn.AdaptiveAvgPool2d(1)(x)           # average pooling like every fucking attention do
        weight = self.layer_attention(avg_feat)         # make weight of shape (b, groups, 1, 1)

        x = x.view(b, self.groups, c // self.groups, h, w)
        weight = weight.view(b, self.groups, 1, 1, 1)
        _x = x.clone()
        for group in range(self.groups):
            _x[:, group] = x[:, group] * weight[:, group]

        _x = _x.view(b, c, h, w)

        return _x


class EfficientLayerAttn(nn.Module):
    def __init__(self,
                 in_channels,
                 groups,
                 act='sigmoid'):
        super(EfficientLayerAttn, self).__init__()
        assert act in ['sigmoid', 'softmax'], "Activation for LayerAttention must be 'sigmoid' or 'softmax'"
        self.in_channels = in_channels
        self.groups = groups
        self.layer_attention = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.groups,
                1
            ),
            nn.Sigmoid() if act == 'sigmoid' else nn.Softmax(dim=1) # channel softmax
        )

    def forward(self, x):
        b, c, h, w = x.shape

        avg_feat = nn.AdaptiveAvgPool2d(1)(x)           # average pooling like every fucking attention do
        weight = self.layer_attention(avg_feat)         # make weight of shape (b, groups, 1, 1)

        x = x.view(b, self.groups, c // self.groups, h, w)
        weight = weight.view(b, self.groups, 1, 1, 1)
        _x = x.clone()
        for group in range(self.groups):
            _x[:, group] = x[:, group] * weight[:, group]

        _x = _x.view(b, c, h, w)

        return _x