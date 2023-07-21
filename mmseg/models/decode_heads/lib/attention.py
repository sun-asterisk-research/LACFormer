# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:15:44 2021

@author: angelou
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv
import math
from mmcv.cnn import ConvModule
class self_attn(nn.Module):
    def __init__(self, in_channels, mode='hw'):
        super(self_attn, self).__init__()

        self.mode = mode

        self.query_conv = Conv(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.key_conv = Conv(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.value_conv = Conv(in_channels, in_channels, kernel_size=(1, 1))

        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)
        projected_query = self.query_conv(x).reshape(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).reshape(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.sigmoid(attention_map)
        projected_value = self.value_conv(x).reshape(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out
    




class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=9):
        super(SpatialAttention, self).__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        pad = (self.kernel_size-1)//2  # Padding on one side for stride 1

        self.grp1_conv1k = nn.Conv2d(self.in_channels, self.in_channels//2, (1, self.kernel_size), padding=(0, pad))
        self.grp1_bn1 = nn.BatchNorm2d(self.in_channels//2)
        self.grp1_convk1 = nn.Conv2d(self.in_channels//2, 1, (self.kernel_size, 1), padding=(pad, 0))
        self.grp1_bn2 = nn.BatchNorm2d(1)

        self.grp2_convk1 = nn.Conv2d(self.in_channels, self.in_channels//2, (self.kernel_size, 1), padding=(pad, 0))
        self.grp2_bn1 = nn.BatchNorm2d(self.in_channels//2)
        self.grp2_conv1k = nn.Conv2d(self.in_channels//2, 1, (1, self.kernel_size), padding=(0, pad))
        self.grp2_bn2 = nn.BatchNorm2d(1)

    def forward(self, input_):
        # Generate Group 1 Features
        grp1_feats = self.grp1_conv1k(input_)
        grp1_feats = F.relu(self.grp1_bn1(grp1_feats))
        grp1_feats = self.grp1_convk1(grp1_feats)
        grp1_feats = F.relu(self.grp1_bn2(grp1_feats))

        # Generate Group 2 features
        grp2_feats = self.grp2_convk1(input_)
        grp2_feats = F.relu(self.grp2_bn1(grp2_feats))
        grp2_feats = self.grp2_conv1k(grp2_feats)
        grp2_feats = F.relu(self.grp2_bn2(grp2_feats))

        added_feats = torch.sigmoid(torch.add(grp1_feats, grp2_feats))
        added_feats = added_feats.expand_as(input_).clone()

        return added_feats


class ChannelwiseAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelwiseAttention, self).__init__()

        self.in_channels = in_channels

        self.linear_1 = nn.Linear(self.in_channels, self.in_channels//4)
        self.linear_2 = nn.Linear(self.in_channels//4, self.in_channels)

    def forward(self, input_):
        n_b, n_c, h, w = input_.size()

        feats = F.adaptive_avg_pool2d(input_, (1, 1)).view((n_b, n_c))
        feats = F.relu(self.linear_1(feats))
        feats = torch.sigmoid(self.linear_2(feats))
        
        # Activity regularizer
        ca_act_reg = torch.mean(feats)

        feats = feats.view((n_b, n_c, 1, 1))
        feats = feats.expand_as(input_).clone()

        return feats


class LayerAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 groups, la_down_rate=8):
        super(LayerAttention, self).__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.layer_attention = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels // la_down_rate, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.in_channels // la_down_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.in_channels // la_down_rate,
                self.groups,
                kernel_size=3, padding=1
            ),
            nn.Sigmoid()
        )
        
        # self.la_conv = ConvModule(self.in_channels, self.in_channels, kernel_size=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)


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
        # _x = self.la_conv(_x)

        return _x


        
class ReverseAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
    def forward(self, input, mul_op):
        background = -1*(torch.sigmoid(input)) + 1
        attn = mul_op * background
        return attn
    
class BoundaryAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, input, mul_op):
        
        score = torch.sigmoid(input)
        dist = torch.abs(score - 0.5)
        boundary_att = 1 - (dist / 0.5)

        attn = mul_op * boundary_att
        return attn

class EfficientSELayer(nn.Module):
    def __init__(self,
                 channels,
                 conv_cfg=None):
        super(EfficientSELayer, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.attn_weight = nn.Sequential(
            ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                conv_cfg=conv_cfg,
                act_cfg=None
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.attn_weight(out)

        return x * out

class SequentialPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax_channel=nn.Softmax(1)
        self.softmax_spatial=nn.Softmax(-1)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        b, c, h, w = x.size()

        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x) #bs,c//2,h,w
        channel_wq=self.ch_wq(x) #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax_channel(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x

        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(channel_out) #bs,c//2,h,w
        spatial_wq=self.sp_wq(channel_out) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wq=self.softmax_spatial(spatial_wq)
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*channel_out
        return spatial_out

class BAM(nn.Module):
    def __init__(self,in_channels):
        super(BAM, self).__init__()
        
        self.boundary_conv=nn.Sequential(
            nn.Conv2d(in_channels, in_channels//3,3,1,1),
            nn.BatchNorm2d(in_channels//3),
            nn.ReLU(inplace=True),
            nn.Dropout2d())
        self.foregound_conv=nn.Sequential(
            nn.Conv2d(in_channels, in_channels//3,3,1, 1),
            nn.BatchNorm2d(in_channels//3),
            nn.ReLU(inplace=True),
            nn.Dropout2d())
        self.background_conv=nn.Sequential(
            nn.Conv2d(in_channels, in_channels//3,3,1, 1),
            nn.BatchNorm2d(in_channels//3),
            nn.ReLU(inplace=True),
            nn.Dropout2d())
        self.out_conv=nn.Sequential(
            nn.Conv2d((in_channels//3)*3, in_channels,3,1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d())
        self.fpn_bottleneck = ConvModule(
            in_channels, 1 ,kernel_size=1, padding=0)
        self.selayer=SELayer((in_channels//3)*3)

    def forward(self, x):
        residual = x
        pred = self.fpn_bottleneck(x)
        score = torch.sigmoid(pred)
        
        #boundary
        dist = torch.abs(score - 0.5)
        boundary_att = 1 - (dist / 0.5)
        boundary_x = x * boundary_att
        
        #foregound
        foregound_att= score
        foregound_att=torch.clip(foregound_att-boundary_att,0,1)
        foregound_x= x*foregound_att

        #background
        background_att=1-score
        background_att=torch.clip(background_att-boundary_att,0,1)
        background_x= x*background_att

        foregound_x= foregound_x 
        background_x= background_x 
        boundary_x= boundary_x  

        foregound_xx=self.foregound_conv(foregound_x)
        background_xx=self.background_conv(background_x)
        boundary_xx=self.boundary_conv(boundary_x)

        out=torch.cat([foregound_xx,background_xx,boundary_xx], dim=1) 
        out=self.selayer(out)
        out=self.out_conv(out)+residual
        return out


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(
            dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(
            dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        # attn = self.conv3(attn)

        return attn * u


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 9, stride=1, padding=12, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn_1 = self.conv0(x)
        attn_2 = self.conv_spatial(x)
        attn = attn_1 + attn_2
        attn = self.conv1(attn)
        return u * attn
    


