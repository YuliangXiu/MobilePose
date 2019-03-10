'''
File: network.py
Project: MobilePose-PyTorch
File Created: Thursday, 7th March 2019 6:33:57 pm
Author: Yuliang Xiu (yuliangxiu@sjtu.edu.cn)
-----
Last Modified: Monday, 11th March 2019 12:50:40 am
Modified By: Yuliang Xiu (yuliangxiu@sjtu.edu.cn>)
-----
Copyright 2018 - 2019 Shanghai Jiao Tong University, Machine Vision and Intelligence Group
'''



from networks import *
from networks.senet import se_resnet
import torch.nn as nn
import dsntnn

class CoordRegressionNetwork(nn.Module):
    def __init__(self, n_locations, backbone):
        super(CoordRegressionNetwork, self).__init__()

        if backbone == "unet":
            self.resnet = UNet()
            self.outsize = 64
        elif backbone == "resnet18":
            self.resnet = resnet.resnet18_ed(pretrained=False)
            self.outsize = 32
        elif backbone == "resnet34":
            self.resnet = resnet.resnet34_ed(pretrained=False)
            self.outsize = 512
        elif backbone == "resnet50":
            self.resnet = resnet.resnet50_ed(pretrained=False)
            self.outsize = 2048
        elif backbone == "senet18":
            self.resnet = se_resnet.senet18_ed(pretrained=False)
            self.outsize = 512
        elif backbone == "shufflenetv2":
            self.resnet = ShuffleNetV2.shufflenetv2_ed(width_mult=1.0)
            self.outsize = 32
        elif backbone == "mobilenetv2":
            self.resnet = MobileNetV2.mobilenetv2_ed(width_mult=1.0)
            self.outsize = 32
        elif backbone == "squeezenet":
            self.resnet = squeezenet1_1()
            self.outsize = 64

        self.hm_conv = nn.Conv2d(self.outsize, n_locations, kernel_size=1, bias=False)

    def forward(self, images):
        # 1. Run the images through our Resnet
        resnet_out = self.resnet(images)
        # 2. Use a 1x1 conv to get one unnormalized heatmap per location
        unnormalized_heatmaps = self.hm_conv(resnet_out)
        # 3. Normalize the heatmaps
        heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)
        # 4. Calculate the coordinates
        coords = dsntnn.dsnt(heatmaps)

        return coords, heatmaps