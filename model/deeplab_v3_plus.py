'''
    deeplab_v3+ :

        "Encoder-Decoder with Atrous Separable Convolution for Semantic Image
        Segmentation"
        Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam.
        (https://arxiv.org/abs/1802.02611)

    according to [mobilenetv2_coco_voc_trainaug / mobilenetv2_coco_voc_trainval]
    https://github.com/lizhengwei1992/models/tree/master/research/deeplab
    we use MobileNet_v2 as feature exstractor

    These codes are motified frome https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/networks/deeplab_xception.py

Author: Zhengwei Li
Data: July 1 2018
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# mobilenet_v2
from model.MobileNet_v2 import MobileNet_v2_os_32, MobileNet_v2_os_8
import pdb


INPUT_SIZE = 512


# filter size : 1x1 conv
# dimension   : 65 x 65 x 320 --> 65 x 65 x 256
class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes):
        super(ASPP_module, self).__init__()

        self.aspp0 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1,
                                            stride=1, padding=0, dilation=1, bias=False),
                    nn.BatchNorm2d(planes))
        self.aspp1 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
                                            stride=1, padding=6, dilation=6, bias=False),
                    nn.BatchNorm2d(planes))        
        self.aspp2 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
                                            stride=1, padding=12, dilation=12, bias=False),
                    nn.BatchNorm2d(planes))
        self.aspp3 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
                                            stride=1, padding=18, dilation=18, bias=False),
                    nn.BatchNorm2d(planes))

    def forward(self, x):
        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)

        return torch.cat((x0, x1, x2, x3), dim=1)

#-------------------------------------------------------------------------------------------------
# DeepLabv_v3_plus_mv2_os_32
# feature exstractor : MobileNet_v2_os_32
#-----------------------------------------
class DeepLabv_v3_plus_mv2_os_32(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=1):

        super(DeepLabv_v3_plus_mv2_os_32, self).__init__()

        # mobilenetv2 feature 
        self.mobilenet_features = MobileNet_v2_os_32(nInputChannels)
        # ASPP
        self.aspp = ASPP_module(320, 256)
        # global pooling
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                        nn.Conv2d(320, 256, 1, stride=1, bias=False))

        self.conv = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn = nn.BatchNorm2d(256)

        # low_level_features to 48 channels
        self.conv2 = nn.Conv2d(24, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)
        
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))
 
        # init weights
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x : 1/32 16 x 16
        x, low_level_features = self.mobilenet_features(x)
        # NOTE ! !
        # 1/32 16 x 16 too small, up first and then ASPP
        x = F.upsample(x, scale_factor=4, mode='bilinear', align_corners=True)

        x_aspp = self.aspp(x)
        x_ = self.global_avg_pool(x)
        x_ = F.upsample(x_, size=x.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x_aspp, x_), dim=1)

        x = self.conv(x)
        x = self.bn(x)
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)

        # 1/4 128 x 128
        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)


        x = self.last_conv(x)
        x = F.upsample(x, scale_factor=4, mode='bilinear', align_corners=True)

        return x

#-------------------------------------------------------------------------------------------------
# DeepLabv_v3_plus_mv2_os_8
# feature exstractor : MobileNet_v2_os_8
#-----------------------------------------
class DeepLabv_v3_plus_mv2_os_8(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=1):

        super(DeepLabv_v3_plus_mv2_os_8, self).__init__()

        # mobilenetv2 feature 
        self.mobilenet_features = MobileNet_v2_os_8(nInputChannels)

        # ASPP
        self.aspp = ASPP_module(320, 256)
        # global pooling
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                        nn.Conv2d(320, 256, 1, stride=1, bias=False))

        self.conv = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn = nn.BatchNorm2d(256)

        # low_level_features to 48 channels
        self.conv2 = nn.Conv2d(24, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)
        
        self.last_conv = nn.Sequential(nn.Conv2d(304, n_classes, kernel_size=1, stride=1))
 
        # init weights
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # x : 1/8 64 x 64
        x, low_level_features = self.mobilenet_features(x)

        x_aspp = self.aspp(x)
        x_ = self.global_avg_pool(x)
        x_ = F.upsample(x_, size=x.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x_aspp, x_), dim=1)

        x = self.conv(x)
        x = self.bn(x)
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)

        # 1/4 128 x 128
        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)

        x = self.last_conv(x)
        x = F.upsample(x, scale_factor=4, mode='bilinear', align_corners=True)

        return x