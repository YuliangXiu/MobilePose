from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils, models
from torch.autograd import Variable
import os
from skimage import io, transform
import numpy as np
import csv
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from dataloader import *

os.environ["CUDA_VISIBLE_DEVICES"]="1"
torch.backends.cudnn.enabled = True
print("GPU : %d"%(torch.cuda.device_count()))

name = "yh"
ROOT_DIR = "/home/yuliang/code/deeppose_tf/datasets/mpii"
PATH_PREFIX = '/home/yuliang/code/DeepPose-pytorch/models/{}/'.format(name)
batchsize = 200
inputsize = 230

train_dataset = PoseDataset(csv_file=os.path.join(ROOT_DIR,'train_joints.csv'),
                                  transform=transforms.Compose([
                                            #    Augmentation(),
                                               Rescale((inputsize,inputsize)),
                                               Expansion(),
                                               ToTensor()
                                           ]))
train_dataloader = DataLoader(train_dataset, batch_size=batchsize,
                        shuffle=False, num_workers = 10)

test_dataset = PoseDataset(csv_file=os.path.join(ROOT_DIR,'test_joints.csv'),
                                  transform=transforms.Compose([
                                               Rescale((inputsize,inputsize)),
                                               Expansion(),
                                               ToTensor()
                                           ]))
test_dataloader = DataLoader(test_dataset, batch_size=batchsize,
                        shuffle=False, num_workers = 10)

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

net = Net().cuda()
gpus = [0,1]
net = torch.load('models/yh/final-noaug-adam.t7').cuda(device_id=gpus[0])
criterion = nn.MSELoss().cuda()
# optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0005, momentum=0.9)
# optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-06, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=1e-05, betas=(0.9, 0.999), eps=1e-08)
# optimizer = optim.SGD(net.parameters(), lr=1e-06, momentum=0.9)


def mse_loss(input, target):
    return torch.sum(torch.pow(input - target,2)) / input.nelement()

train_loss_all = []
valid_loss_all = []

minloss = 302.0

for epoch in range(1000):  # loop over the dataset multiple times
    
    train_loss_epoch = []
    for i, data in enumerate(train_dataloader):
        images, poses = data['image'], data['pose']
        images, poses = Variable(images.cuda()), Variable(poses.cuda())
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, poses)
        loss.backward()
        optimizer.step()

        train_loss_epoch.append(loss.data[0])

    if epoch%5==0:
        valid_loss_epoch = []
        for i_batch, sample_batched in enumerate(test_dataloader):

            net_forward = net
            images = sample_batched['image'].cuda()
            poses = sample_batched['pose'].cuda()
            outputs = net_forward(Variable(images, volatile=True))
            valid_loss_epoch.append(mse_loss(outputs.data,poses))

        if np.mean(np.array(valid_loss_epoch)) < minloss:
            minloss = np.mean(np.array(valid_loss_epoch))
            checkpoint_file = PATH_PREFIX + 'final-noaug-adam.t7'.format(epoch)
            torch.save(net, checkpoint_file)
            print('==> checkpoint model saving to %s'%checkpoint_file)

        print('[epoch %d] train loss: %.8f, valid loss: %.8f' %
          (epoch + 1, np.mean(np.array(train_loss_epoch)), np.mean(np.array(valid_loss_epoch))))
        with open(PATH_PREFIX+"resnet-log.txt", 'a+') as file_output:
            file_output.write('[epoch %d] train loss: %.8f, valid loss: %.8f\n' %
              (epoch + 1, np.mean(np.array(train_loss_epoch)), np.mean(np.array(valid_loss_epoch))))
            file_output.flush() 
            
print('Finished Training')