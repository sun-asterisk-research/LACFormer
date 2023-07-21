from turtle import forward
import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
from .conv import Conv, BNPReLU


class FeatureEnhanceModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        
        self.conv1 = Conv(in_channels, in_channels, kernel_size=3, dilation=1, bn=True, relu=True)
        self.conv2 = Conv(in_channels, in_channels, kernel_size=3, dilation=3, bn=True, relu=True)
        self.conv3 = Conv(in_channels, in_channels, kernel_size=3, dilation=5, bn=True, relu=True)
        
        self.fusion = Conv(in_channels * 3, out_channels, kernel_size=1, bn=True, relu=True)
        
    def forward(self, inp):
        inp1 = self.conv1(inp)
        inp2 = self.conv2(inp)
        inp3 = self.conv3(inp)
        
        output = self.fusion(torch.cat([inp1, inp2, inp3], dim=1))
        
        return output

class FeatureEnhanceModule2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.bn_relu_1 = BNPReLU(in_channels)
        
        self.conv = Conv(in_channels, out_channels, kernel_size=3, dilation=1, bn=True, relu=True)
        
        self.conv1 = Conv(out_channels, out_channels, kernel_size=3, dilation=3, bn=True, relu=True)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, dilation=5, bn=True, relu=True)
        self.conv3 = Conv(out_channels, out_channels, kernel_size=3, dilation=7, bn=True, relu=True)
        
        self.fusion = Conv(out_channels * 3, out_channels, kernel_size=1, bn=True, relu=True)
        
    def forward(self, inp):
        inp = self.conv(inp)
        
        inp1 = self.conv1(inp)
        inp2 = self.conv2(inp)
        inp3 = self.conv3(inp)
        
        out1 = inp1
        out2 = out1 + inp2
        out3 = out2 + inp3
        
        output = self.fusion(torch.cat([out1, out2, out3], dim=1))
        
        return output + inp