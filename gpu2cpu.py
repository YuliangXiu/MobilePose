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

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# os.environ["CUDA_VISIBLE_DEVICES"]=""
# os.environ["CUDA_LAUNCH_BLOCKING"]="1"
# torch.cuda.set_device(1) 
torch.backends.cudnn.enabled = True
print(torch.cuda.device_count())
gpus = [0,1]


checkpoint_path = "./scale2-checkpoint100.t7"
net = torch.load(checkpoint_path, map_location={'cuda:1' : 'cuda:1', 'cuda:0': 'cuda:0'}).cuda(device_id=gpus[0])
# net = torch.load(checkpoint_path)
net_cpu = net.cpu()
torch.save(net_cpu, "./scale2-checkpoint-cpu-100.t7")
torch.save(net_cpu.state_dict(), "./scale2-checkpoint-cpu-100-state.t7")
