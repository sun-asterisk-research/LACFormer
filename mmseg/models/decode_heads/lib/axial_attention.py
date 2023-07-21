# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:17:13 2021

@author: angelou
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv
from .attention import self_attn
import math


class AA_kernel(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AA_kernel, self).__init__()
        self.conv0 = Conv(in_channel, out_channel, kernel_size=1,stride=1)
        self.conv1 = Conv(out_channel, out_channel, kernel_size=3,stride=1)
        self.Hattn = self_attn(out_channel, mode='h')
        self.Wattn = self_attn(out_channel, mode='w')

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        Hx = self.Hattn(x)
        Wx = self.Wattn(Hx)

        return Wx
    


class PAA_kernel(nn.Module):
    def __init__(self, in_channel, out_channel, receptive_size=3):
        super(PAA_kernel, self).__init__()
        self.conv0 = Conv(in_channel, out_channel, 1)
        self.conv1 = Conv(out_channel, out_channel, kernel_size=(1, receptive_size))
        self.conv2 = Conv(out_channel, out_channel, kernel_size=(receptive_size, 1))
        self.conv3 = Conv(out_channel, out_channel, 3, dilation=receptive_size)
        self.Hattn = self_attn(out_channel, mode='h')
        self.Wattn = self_attn(out_channel, mode='w')

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        Hx = self.Hattn(x)
        Wx = self.Wattn(x)

        x = self.conv3(Hx + Wx)
        return x

class PAA_e(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PAA_e, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = Conv(in_channel, out_channel, 1)
        self.branch1 = PAA_kernel(in_channel, out_channel, 3)
        self.branch2 = PAA_kernel(in_channel, out_channel, 5)
        self.branch3 = PAA_kernel(in_channel, out_channel, 7)

        self.conv_cat = Conv(4 * out_channel, out_channel, 3)
        self.conv_res = Conv(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))

        return x