import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class DepthwiseConvBlock(nn.Module):
    """
    Depthwise seperable convolution. 
    
    
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
        super(DepthwiseConvBlock,self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
                               padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                   stride=1, padding=0, dilation=1, groups=1, bias=False)
        
        
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU()
        
    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)
    
class ConvBlock(nn.Module):
    """
    Convolution block with Batch Normalization and ReLU activation.
    
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
        super(ConvBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return self.act(x)

class BiFPNBlock(nn.Module):
    """
    Bi-directional Feature Pyramid Network
    """
    def __init__(self, feature_size=64, epsilon=0.0001):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon
        
        self.p3_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p4_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p5_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p6_td = DepthwiseConvBlock(feature_size, feature_size)
        
        self.p4_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p5_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p6_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p7_out = DepthwiseConvBlock(feature_size, feature_size)
        
        # TODO: Init weights
        self.w1 = nn.Parameter(torch.Tensor(2, 3))
        self.w1_relu = nn.ReLU()
        self.w2 = nn.Parameter(torch.Tensor(3, 3))
        self.w2_relu = nn.ReLU()
    
    def forward(self, inputs):
        p3_x, p4_x, p5_x, p6_x = inputs
        # Calculate Top-Down Pathway
        w1 = self.w1_relu(self.w1)
        w1 /= torch.sum(w1, dim=0) + self.epsilon
        w2 = self.w2_relu(self.w2)
        w2 /= torch.sum(w2, dim=0) + self.epsilon
        
        p6_td = p6_x
        p5_td = self.p5_td(w1[0, 0] * p5_x + w1[1, 0] * F.interpolate(p6_td, scale_factor=2))
        p4_td = self.p4_td(w1[0, 1] * p4_x + w1[1, 1] * F.interpolate(p5_td, scale_factor=2))
        p3_td = self.p3_td(w1[0, 2] * p3_x + w1[1, 2] * F.interpolate(p4_td, scale_factor=2))
        
        # Calculate Bottom-Up Pathway
        p3_out = p3_td
        p4_out = self.p4_out(w2[0, 0] * p4_x + w2[1, 0] * p4_td + w2[2, 0] * nn.Upsample(scale_factor=0.5)(p3_out))
        p5_out = self.p5_out(w2[0, 1] * p5_x + w2[1, 1] * p5_td + w2[2, 1] * nn.Upsample(scale_factor=0.5)(p4_out))
        p6_out = self.p6_out(w2[0, 2] * p6_x + w2[1, 2] * p6_td + w2[2, 2] * nn.Upsample(scale_factor=0.5)(p5_out))

        return [p3_out, p4_out, p5_out, p6_out]
    
class BiFPN(nn.Module):
    def __init__(self, size, feature_size=64, num_layers=1, epsilon=0.0001):
        super(BiFPN, self).__init__()
        self.p3 = nn.Conv2d(size[0], feature_size, kernel_size=1, stride=1, padding=0)
        self.p4 = nn.Conv2d(size[1], feature_size, kernel_size=1, stride=1, padding=0)
        self.p5 = nn.Conv2d(size[2], feature_size, kernel_size=1, stride=1, padding=0)
        
        # p6 is obtained via a 3x3 stride-2 conv on C5
        self.p6 = nn.Conv2d(size[3], feature_size, kernel_size=1, stride=1, padding=0)



        self.bifpn = BiFPNBlock(feature_size)
        self.post_conv = nn.ModuleList()
        for i in range(4):
            self.post_conv.append(
                nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
            )
    def forward(self, inputs):
        c3, c4, c5, c6 = inputs

        # Calculate the input column of BiFPN
        p3_x = self.p3(c3)        
        p4_x = self.p4(c4)
        p5_x = self.p5(c5)
        p6_x = self.p6(c6)
        features = self.bifpn([p3_x, p4_x, p5_x, p6_x])
        for i in range(4):
            features[i] = self.post_conv[i](features[i])
        return features
    


# bifpn = BiFPN([64, 128, 320, 512], 128)

# test1 = torch.rand((2, 64, 128, 128))
# test2 = torch.rand((2, 128, 64, 64))
# test3 = torch.rand((2, 320, 32, 32))
# test4 = torch.rand((2, 512, 16, 16))

# bifpn_out = bifpn([test1, test2, test3, test4])
# for i in bifpn_out:
#     print(i.shape)