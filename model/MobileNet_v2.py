'''
    MobileNet_v2_OS_32
    these codes brow from https://github.com/tonylins/pytorch-mobilenet-v2
    but little different from original architecture :
    +-------------------------------------------+-------------------------+
    |                                               output stride
    +===========================================+=========================+                                            
    |       original MobileNet_v2_OS_32         |          32             | 
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
    |          MobileNet_v2_OS_8                |          8              |
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
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
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

#-------------------------------------------------------------------------------------------------
# MobileNet_v2_os_32
#--------------------
class MobileNet_v2_os_32(nn.Module):
    def __init__(self, nInputChannels=3):
        super(MobileNet_v2_os_32, self).__init__()
        # 1/2
        # 256 x 256
        self.head_conv = conv_bn(nInputChannels, 32, 2)

        # 1/2
        # 256 x 256
        self.block_1 = InvertedResidual(32, 16, 1, 1)
        # 1/4 128 x 128
        self.block_2 = nn.Sequential( 
            InvertedResidual(16, 24, 2, 6),
            InvertedResidual(24, 24, 1, 6)
            )
        # 1/8 64 x 64
        self.block_3 = nn.Sequential( 
            InvertedResidual(24, 32, 2, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 32, 1, 6)
            )
        # 1/16 32 x 32
        self.block_4 = nn.Sequential( 
            InvertedResidual(32, 64, 2, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6)            
            )
        # 1/16 32 x 32
        self.block_5 = nn.Sequential( 
            InvertedResidual(64, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6)          
            )
        # 1/32 16 x 16
        self.block_6 = nn.Sequential( 
            InvertedResidual(96, 160, 2, 6),
            InvertedResidual(160, 160, 1, 6),
            InvertedResidual(160, 160, 1, 6)          
            )
        # 1/32 16 x 16
        self.block_7 = InvertedResidual(160, 320, 1, 6)


    def forward(self, x):
        x = self.head_conv(x)

        x = self.block_1(x)
        x = self.block_2(x)
        low_level_feat = x

        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        x = self.block_7(x)

        return x, low_level_feat



#-------------------------------------------------------------------------------------------------
# MobileNet_v2_os_8
#--------------------
class MobileNet_v2_os_8(nn.Module):
    def __init__(self, nInputChannels=3):
        super(MobileNet_v2_os_8, self).__init__()

        # 1/2 256 x 256
        self.head_conv = conv_bn(nInputChannels, 32, 2)
        # 1/2 256 x 256
        # head bock different form original mobilenetv2
        self.block_1 = nn.Sequential(
            # dw
            nn.Conv2d(32, 32, 3, 1, 1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(32, 16, 1, 1, 0, bias=False),
            nn.BatchNorm2d(16),
        )
        # 1/4 128 x 128
        self.block_2 = nn.Sequential( 
            InvertedResidual(16, 24, 2, 6),
            InvertedResidual(24, 24, 1, 6)
            )
        # 1/8 64 x 64
        self.block_3 = nn.Sequential( 
            InvertedResidual(24, 32, 2, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 32, 1, 6)
            )
        # 1/8 64 x 64
        self.block_4 = nn.Sequential( 
            InvertedResidual(32, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6)            
            )
        # 1/8 64 x 64
        self.block_5 = nn.Sequential( 
            InvertedResidual(64, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6)          
            )
        # 1/8 64 x 64
        self.block_6 = nn.Sequential( 
            InvertedResidual(96, 160, 1, 6),
            InvertedResidual(160, 160, 1, 6),
            InvertedResidual(160, 160, 1, 6)          
            )
        # 1/8 64 x 64
        self.block_7 = InvertedResidual(160, 320, 1, 6)

    def forward(self, x):
        x = self.head_conv(x)

        x = self.block_1(x)
        x = self.block_2(x)
        # 1/4 128 x 128
        low_level_feat = x

        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        x = self.block_7(x)

        return x, low_level_feat



if __name__ == "__main__":
    model = MobileNet_v2_OS_8()
    x = torch.randn(1,3,960,720)
    y, low_level_feat = model(x)
    print(y.size())

