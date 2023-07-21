import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize


@HEADS.register_module()
class SSFormerHead(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        num_inputs = len(self.in_channels)
        self.interpolate_mode = interpolate_mode
        assert num_inputs == len(self.in_index)

        self.local_emphasises = nn.ModuleList()
        for i in range(num_inputs):
            # 2 convs with ReLU
            local_emphasis = nn.Sequential(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=3,
                    padding=1,
                    bias=False
                ),
                ConvModule(
                    in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=3,
                    padding=1,
                    bias=False
                )
            )
            self.local_emphasises.append(local_emphasis)

        self.linear_projections = nn.ModuleList()
        for i in range(num_inputs - 1):
            self.linear_projections.append(
                ConvModule(
                    in_channels=self.channels * 2,
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        _inputs = []
        # local emphasis
        for idx in range(len(inputs)):
            x = inputs[idx]
            local_emphasis = self.local_emphasises[idx]
            _inputs.append(
                resize(
                    input=local_emphasis(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        # Stepwise Feature Aggregation
        out = torch.empty(
            _inputs[0].shape
        )
        for idx in range(len(_inputs) - 1, 0, -1):
            linear_prj = self.linear_projections[idx - 1]
            # cat first 2 from _inputs
            if idx == len(_inputs) - 1:
                x1 = _inputs[idx]
                x2 = _inputs[idx - 1]
            # if not first 2 then cat from prev outs and _inputs
            else:
                x1 = out
                x2 = _inputs[idx - 1]
            x = torch.cat([x1, x2], dim=1)
            out = linear_prj(x)

        out = self.cls_seg(out)

        return out