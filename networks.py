'''
File: networks.py
Project: MobilePose
File Created: Thursday, 8th March 2018 2:59:28 pm
Author: Yuliang Xiu (yuliangxiu@sjtu.edu.cn)
-----
Last Modified: Thursday, 8th March 2018 3:01:29 pm
Modified By: Yuliang Xiu (yuliangxiu@sjtu.edu.cn>)
-----
Copyright 2018 - 2018 Shanghai Jiao Tong University, Machine Vision and Intelligence Group
'''


from torchvision import models
import torch.nn as nn
from mobilenetv2 import *
import dsntnn

def get_graph_path(model_name):
    return {
        'resnet': './models/resnet_227x227.t7',
        'mobilenet': './models/mobilenet_224x224.t7',
    }[model_name]

def model_wh(model_name):
    # get the input image size from the model name
    if 'resnet' in model_name.split('_')[0]:
        width, height = 227, 227
    else:
        width, height = 224, 224 
    return int(width), int(height)

class Net(nn.Module):
    
    def __init__(self, layers):
        super(Net, self).__init__()
        if layers == 18:
            model = models.resnet18(pretrained=True)
        elif layers == 34:
            model = models.resnet34(pretrained=True)
        # change the first layer to recieve five channel image
        model.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3,bias=True)
        # change the last layer to output 32 coordinates
        # model.fc=nn.Linear(512,32)
        # remove final two layers(fc, avepool)
        model = nn.Sequential(*(list(model.children())[:-2]))
        for param in model.parameters():
            param.requires_grad = True
        self.resnet = model
        
    def forward(self, x):
       
        pose_out = self.resnet(x)
        return pose_out

class CoordRegressionNetwork(nn.Module):
    def __init__(self, n_locations, layers):
        super(CoordRegressionNetwork, self).__init__()
        self.resnet = Net(layers)
        self.hm_conv = nn.Conv2d(512, n_locations, kernel_size=1, bias=False)

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