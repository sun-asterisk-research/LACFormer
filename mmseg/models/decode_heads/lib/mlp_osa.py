import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from .attention import *

from mmseg.ops import resize
from mmseg.models.utils import SELayer
from .axial_attention import AA_kernel 
class MLP_OSA(nn.Module):
    def __init__(self,
                 interpolate_mode='bilinear',
                 ops='cat', in_channels=None, channels=None,
                 **kwargs):
        super().__init__()

        self.interpolate_mode = interpolate_mode
        assert ops in ['cat', 'add']
        self.ops = ops
        self.in_channels = in_channels
        self.channels = channels
        num_inputs = len(self.in_channels)

        # self.convs = nn.ModuleList()
        # for i in range(num_inputs):
        #     convs = nn.Sequential(nn.Conv2d(self.channels, self.channels, 3, padding=1, bias=False), nn.ReLU())
        #     self.convs.append(convs)

        self.linear_projections = nn.ModuleList()
        for i in range(num_inputs - 1):
            self.linear_projections.append(
                nn.Sequential(
                    SELayer(self.channels * 2),
                    ConvModule(
                    in_channels=self.channels * 2 if self.ops == 'cat' else self.channels,
                    out_channels=self.channels, norm_cfg=None,
                    kernel_size=1, 
                    padding=0),
                    
                )
            )
        
        self.aa_module = AA_kernel(self.channels, self.channels)
        self.fusion_conv = ConvModule(
            in_channels=self.channels * 4,
            out_channels=self.channels,
            kernel_size=1, padding=0,
            norm_cfg=None)
        
        self.layer_attn = LayerAttention(
            self.channels * 4,
            groups=4, la_down_rate=8
        )

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        _inputs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            _inputs.append(
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=False))

        # slow concatenate
        _out = torch.empty(
            _inputs[0].shape
        )
        outs = [_inputs[-1]]
        for idx in range(len(_inputs) - 1, 0, -1):
            linear_prj = self.linear_projections[idx - 1]
            # cat first 2 from _inputs
            if idx == len(_inputs) - 1:
                x1 = _inputs[idx]
                x2 = _inputs[idx - 1]
            # if not first 2 then cat from prev outs and _inputs
            else:
                x1 = _out
                x2 = _inputs[idx - 1]
            if self.ops == 'cat':
                x = torch.cat([x1, x2], dim=1)
            else:
                x = x1 + x2
            _out = linear_prj(x)
            outs.append(_out)
            
        # out = torch.cat(outs, dim=1)
        # out = self.layer_attn(out)
        # out = self.fusion_conv(out)
        # aa_atten = self.aa_module(out)
        # # out  = outs[-1]  + aa_atten
        # out  = out  + aa_atten
        # out += outs[-1]
        
        # outs.append(out)

        return _out