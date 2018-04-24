import torch.nn as nn
import math
import numpy as np
import torch
from torch.autograd import Variable


def conv_bn(inp, oup, stride):
    # convolution layer with batchnorm
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def mix_conv(inp):
    # used to mix input 3 channels
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, 1, 1, bias=False),
        nn.BatchNorm2d(inp)
    )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        # nn.ReLU(inplace=True),
    )

def conv_1x1_bn(inp, oup):
    # 1x1 convolution layer with batchnorm
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


# calculate parameters:
# inp * expand_ratio * inp + inp * expand_ratio * 3 * 3
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

class Hourglass2(nn.Module):
    def __init__(self, inp, oup, inner_size):
        super(Hourglass2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            conv_dw(inp, inner_size, 1),
            conv_dw(inner_size, inner_size, 1),
            conv_dw(inner_size, inner_size, 1),
        )

        self.conv2 = nn.Sequential(
            conv_dw(inp, inner_size, 1),
            conv_dw(inner_size, inner_size, 1),
            conv_dw(inner_size, oup, 1)
        )

        self.conv3 = nn.Sequential(
            conv_dw(inner_size, inner_size, 1),
            conv_dw(inner_size, inner_size, 1),
            conv_dw(inner_size, oup, 1)
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            conv_dw(inner_size, inner_size, 1),
            conv_dw(inner_size, inner_size, 1),
            conv_dw(inner_size, inner_size, 1),
            conv_dw(inner_size, oup, 1),
            conv_dw(oup, oup, 1),
            nn.Upsample(scale_factor=2)
        )

        self.conv5 = nn.Sequential(
            conv_dw(oup, oup, 1),
            nn.Upsample(scale_factor=2)
        )


    def forward(self, x):
        x1 = self.conv5(self.conv4(self.conv1(x)))
        x2 = self.conv5(self.conv3(self.conv1(x)))
        return self.conv2(x) + x1 + x2


class HeatmapMobileNetV2(nn.Module):
    def __init__(self, image_channel=5, n_class=32, input_size=224, width_mult=1.):
        super(HeatmapMobileNetV2, self).__init__()
        # begin heatmap-mobilenetV2
        self.features = [mix_conv(image_channel)]
        # to get kernel = 7
        self.features.append(conv_dw(image_channel, 8, 1))
        self.features.append(conv_dw(8, 16, 1))
        # maxpool
        self.features.append(nn.MaxPool2d(2, stride=2))
        self.features.append(Hourglass2(16, 16, 16))
        self.features.append(Hourglass2(16, 16, 16))

        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [2, 32, 2, 1],
            [2, 32, 3, 1],
            [2, 64, 2, 1],
            [2, 96, 2, 1],
        ]

        input_channel = 16
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel


        self.features.append(Hourglass2(96, 128, 32))

        # self.features.append(conv_dw(256, 256, 1))
        # self.features.append(conv_dw(256, 128, 1))
        # self.features.append(nn.Conv2d(128, 1, 1, 1))

        self.features.append(conv_dw(128, 128, 1))
        self.features.append(conv_dw(128, 64, 1))
        self.features.append(nn.Conv2d(64, 16, 1, 1))
        self.features = nn.Sequential(*self.features)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def count_para(parameters):
    total_size = 0
    for para in parameters:
        cur_size = 1
        parasize = para.size()
        for i in range(len(parasize)):
            cur_size *= parasize[i]
        total_size += cur_size
    # float 4 bytes
    return total_size * 4 * 1.0 / 1024 / 1024


# image = np.random.rand(10, 5, 224, 224)
# image = torch.from_numpy(image).float()
# # print(image.shape)
# net = HeatmapMobileNetV2()
#
# print("para size is", count_para(net.parameters()))
# out = net(Variable(image))
# print(out.size())


