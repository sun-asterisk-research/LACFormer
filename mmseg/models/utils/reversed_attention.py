import torch
import torch.nn as nn


class ReversedAttention(nn.Module):
    def __init__(self):
        super(ReversedAttention, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape

        max_feat, _ = torch.max(x, dim=1, keepdim=True)
        out_feat = -1 * torch.sigmoid(max_feat) + 1
        out_feat = out_feat.expand([-1, c, -1, -1])

        feat = out_feat * x

        return feat
