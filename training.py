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
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import warnings
warnings.filterwarnings('ignore')

from dataloader import *
from image-util import *
from nets import *
from mobilenetv2 import *


os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# torch.cuda.set_device(1) 
torch.backends.cudnn.enabled = True
print(torch.cuda.device_count())
gpus = [1,2]


ROOT_DIR = "/home/yuliang/code/deeppose_tf/datasets/mpii"

train_dataset = PoseDataset(csv_file=os.path.join(ROOT_DIR,'train_joints.csv'),
                                  transform=transforms.Compose([
                                               Rescale((227,227)),
                                               Expansion(),
                                               ToTensor()
                                           ]))
train_dataloader = DataLoader(train_dataset, batch_size=256,
                        shuffle=False, num_workers = 10)

test_dataset = PoseDataset(csv_file=os.path.join(ROOT_DIR,'test_joints.csv'),
                                  transform=transforms.Compose([
                                               Rescale((227,227)),
                                               Expansion(),
                                               ToTensor()
                                           ]))
test_dataloader = DataLoader(test_dataset, batch_size=256,
                        shuffle=False, num_workers = 10)



# net = torch.load('checkpoint20.t7').cuda(device_id=gpus[1])
net = Net()
criterion = nn.MSELoss().cuda()
# optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001, momentum=0.9)
optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)


def mse_loss(input, target):
    return torch.sum(torch.pow(input - target,2)) / input.nelement()

train_loss_all = []
valid_loss_all = []

for epoch in tqdm_notebook(range(1000)):  # loop over the dataset multiple times
    
    train_loss_epoch = []
    for i, data in enumerate(train_dataloader):
        if i % 20 == 0:
            print ("i=%d in train_dataloader" % i)
        # get the inputs
        images, poses = data['image'], data['pose']
        # wrap them in Variable
        images, poses = Variable(images.cuda()), Variable(poses.cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(images)
        loss = criterion(outputs, poses)
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss_epoch.append(loss.data[0])
    if epoch%10==0:
        PATH_PREFIX = '/disk3/yinghong/data/mobile-model/'
        checkpoint_file = PATH_PREFIX + 'orig-checkpoint{}.t7'.format(epoch)
#         checkpoint_file = 'checkpoint{}.t7'.format(epoch)
        torch.save(net, checkpoint_file)
        valid_loss_epoch = []
        for i_batch, sample_batched in enumerate(test_dataloader):

            net_forward = torch.load(checkpoint_file).cuda()
            images = sample_batched['image'].cuda()
            poses = sample_batched['pose'].cuda()
            outputs = net_forward(Variable(images, volatile=True))
            valid_loss_epoch.append(mse_loss(outputs.data,poses))
        print('[epoch %d] train loss: %.8f, valid loss: %.8f' %
          (epoch + 1, sum(train_loss_epoch)/(71*256), sum(valid_loss_epoch)/(8*256)))
        print('==> checkpoint model saving to %s'%checkpoint_file)
        train_loss_all.append(sum(train_loss_epoch)/(71*256))
        valid_loss_all.append(sum(valid_loss_epoch)/(8*256))
            

print('Finished Training')



useCuda = True

version = "mobilenetv2-l2"
epoch_start = 0
if useCuda:
    net = MobileNetV2().cuda()
    # switch to train mode
    net.train()
else:
    net = MobileNetV2()
    # switch to train mode
    net.train()
    
if useCuda:
    criterion = nn.MSELoss().cuda()
    # criterion = nn.SmoothL1Loss().cuda()
else:
    criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss().cuda()
    
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001, momentum=0.9)
# optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)


from tqdm import tqdm_notebook

def mse_loss(input, target):
    return torch.sum(torch.pow(input - target,2)) / input.nelement()

train_loss_all = []
valid_loss_all = []


for epoch in tqdm_notebook(range(epoch_start,1000)):  # loop over the dataset multiple times
    
    train_loss_epoch = []
    for i, data in enumerate(train_dataloader):
        # get the inputs
        images, poses = data['image'], data['pose']
        # wrap them in Variable
        if useCuda:
            images, poses = Variable(images.cuda()), Variable(poses.cuda())
        else:
            images, poses = Variable(images), Variable(poses)
#         print("input",images.size())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(images)
        loss = criterion(outputs, poses)
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss_epoch.append(loss.data[0])
    if epoch%5==0:
        checkpoint_file = './checkpoint/checkpoint-%s-epoch%d.t7'%(version,epoch)
        torch.save(net, checkpoint_file)
        valid_loss_epoch = []
        for i_batch, sample_batched in enumerate(test_dataloader):
            if useCuda:
                net_forward = torch.load(checkpoint_file).cuda()
                images = sample_batched['image'].cuda()
                poses = sample_batched['pose'].cuda()
            else:
                net_forward = torch.load(checkpoint_file)
                images = sample_batched['image']
                poses = sample_batched['pose']
            outputs = net_forward(Variable(images, volatile=True))
            valid_loss_epoch.append(mse_loss(outputs.data,poses))
        print('[epoch %d] train loss: %.8f, valid loss: %.8f' %
          (epoch + 1, sum(train_loss_epoch)/(71*256), sum(valid_loss_epoch)/(8*256)))
        with open('./checkpoint/ckpoint-%s-data'%(version), 'a+') as file_output:
            file_output.write('[epoch %d] train loss: %.8f, valid loss: %.8f\n' %
              (epoch + 1, sum(train_loss_epoch)/(71*256), sum(valid_loss_epoch)/(8*256)))
            file_output.flush() 
        print('==> checkpoint model saving to %s'%checkpoint_file)
        train_loss_all.append(sum(train_loss_epoch)/(71*256))
        valid_loss_all.append(sum(valid_loss_epoch)/(8*256))
#         break
#     break
            

print('Finished Training')