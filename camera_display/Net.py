import torch
import torch.nn as nn
from torchvision import models


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = True
        model.fc=nn.Linear(512,32)
        self.resnet = model

    def forward(self, x):
        pose_out = self.resnet(x)
        return pose_out
