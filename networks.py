'''
File: networks.py
Project: DeepPose
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

def get_graph_path(model_name):
    return {
        'resnet': './models/demo/resnet18_227x227.t7',
        'mobilenet': './models/demo/mobilenetv2_224x224.t7',
    }[model_name]

def model_wh(model_name):
    width, height = model_name.split('_')[-1].split('x')
    return int(width), int(height.split(".")[0])

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        model = models.resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3,bias=True)
        model.fc=nn.Linear(512,32)
        for param in model.parameters():
            param.requires_grad = True
        self.resnet = model.cuda()
        
    def forward(self, x):
       
        pose_out = self.resnet(x)
        return pose_out