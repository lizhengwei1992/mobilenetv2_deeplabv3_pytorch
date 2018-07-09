'''
    MobileNet_v2
    these codes brow from https://github.com/tonylins/pytorch-mobilenet-v2
    but little different from original architecture :
    +-------------------------------------------+-------------------------+
    |                                               output stride
    +===========================================+=========================+                                            
    |       original Mobile_Net_V2              |          32             | 
    +-------------------------------------------+-------------------------+
    |   self.interverted_residual_setting = [   |                         |
    |       # t, c, n, s                        |                         |
    |       [1, 16, 1, 1],                      |  pw -> dw -> pw-linear  |
    |       [6, 24, 2, 2],                      |                         |
    |       [6, 32, 3, 2],                      |                         |
    |       [6, 64, 4, 2],                      |       stride = 2        |
    |       [6, 96, 3, 1],                      |                         |
    |       [6, 160, 3, 2],                     |       stride = 2        |
    |       [6, 320, 1, 1],                     |                         |
    |   ]                                       |                         |
    +-------------------------------------------+-------------------------+
    |    mobile_net_v2 in deeplab_v3+            |          8             |
    +-------------------------------------------+-------------------------+
    |   self.interverted_residual_setting = [   |                         |
    |       # t, c, n, s                        |                         |
    |       [1, 16, 1, 1],                      |    dw -> pw-linear      |
    |       [6, 24, 2, 2],                      |                         |
    |       [6, 32, 3, 2],                      |                         |
    |       [6, 64, 4, 1],                      |       stride = 1        |
    |       [6, 96, 3, 1],                      |                         |
    |       [6, 160, 3, 1],                     |       stride = 1        |
    |       [6, 320, 1, 1],                     |                         |
    |   ]                                       |                         |
    +-------------------------------------------+-------------------------+

    Notation! I throw away last layers.
    if input is       3   x 513 x 513
    then, feature map 320 x 65  x 65

Author: Zhengwei Li
Data: July 1 2018
'''
import torch
import torch.nn as nn
import math

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            # [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2], 
            [6, 64, 4, 1],     # difference [6, 64, 4, 2]
            [6, 96, 3, 1],
            [6, 160, 3, 1],    # difference [6, 64, 4, 2]
            [6, 320, 1, 1],
        ]

        # building first layer
        self.features = [conv_bn(3, 32, 2)]

        # head bock different form original mobilenetv2
        block_head = nn.Sequential(
            # dw
            nn.Conv2d(32, 32, 3, 1, 1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(32, 16, 1, 1, 0, bias=False),
            nn.BatchNorm2d(16),
        )
        self.features.append(block_head)

        # building inverted residual blocks
        input_channel = int(16)
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel

        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        x = self.features(x)

        return x

if __name__ == "__main__":
    model = MobileNetV2()
    x = torch.randn(1,3,512,512)
    y = model(x)
    print(y.size())

