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
from model.MobileNet_v2 import MobileNetV2
import pdb


INPUT_SIZE = 512

# only aspp0
# filter size : 1x1 conv
# dimension   : 65 x 65 x 320 --> 65 x 65 x 256
class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes):
        super(ASPP_module, self).__init__()
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=1,
                                            stride=1, padding=0, dilation=1)
        self.batch_norm = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.batch_norm(x)

        return x


class DeepLab_v3_plus(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=21):

        super(DeepLab_v3_plus, self).__init__()

        # mobilenetv2 feature 
        self.mobilenet_features = MobileNetV2(nInputChannels)

        # ASPP
        # only aspp0 !
        self.aspp0 = ASPP_module(320, 256)


        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                        nn.Conv2d(320, 256, 1, stride=1, bias=False))

        self.conv = nn.Conv2d(512, 256, 1, bias=False)
        self.bn = nn.BatchNorm2d(256)
	
        self.identity = nn.ReLU(inplace=True)
        self.last_conv = nn.Conv2d(256, n_classes, kernel_size=1, stride=1, bias=True)

        # init weights
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                # torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.mobilenet_features(x)
        x0 = self.aspp0(x)
        x1 = self.global_avg_pool(x)
        x1 = F.upsample(x1, size=x.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x0, x1), dim=1)

        x = self.conv(x)
        x = self.bn(x)

        x = self.identity(x)

        x = self.last_conv(x)
        x = F.upsample(x, scale_factor=8, mode='bilinear', align_corners=True)

        return x

