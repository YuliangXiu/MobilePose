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
# import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from dataloader import *
from mobilenetv2 import *


os.environ["CUDA_VISIBLE_DEVICES"]="0"
# torch.cuda.set_device(1) 
torch.backends.cudnn.enabled = True
print("GPU : %d"%(torch.cuda.device_count()))


ROOT_DIR = "/home/yuliang/code/deeppose_tf/datasets/mpii"

batch_size = 128
input_size = 224

train_dataset = PoseDataset(csv_file=os.path.join(ROOT_DIR,'train_joints.csv'),
                                  transform=transforms.Compose([
                                               Rescale((input_size,input_size)),
                                               Expansion(),
                                               ToTensor()
                                           ]))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=False, num_workers = 10)

test_dataset = PoseDataset(csv_file=os.path.join(ROOT_DIR,'test_joints.csv'),
                                  transform=transforms.Compose([
                                               Rescale((input_size,input_size)),
                                               Expansion(),
                                               ToTensor()
                                           ]))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                        shuffle=False, num_workers = 10)


useCuda = True

version = "mobilenetv2"
if useCuda:
    net = MobileNetV2().cuda()
    criterion = nn.MSELoss().cuda()
    # switch to train mode
    net.train()
else:
    net = MobileNetV2()
    criterion = nn.MSELoss()
    # switch to train mode
    net.train()
    
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001, momentum=0.9)

def mse_loss(input, target):
    return torch.sum(torch.pow(input - target,2)) / input.nelement()

train_loss_all = []
valid_loss_all = []


for epoch in tqdm(range(200)):  # loop over the dataset multiple times
    
    train_loss_epoch = []
    for i, data in enumerate(train_dataloader):
        images, poses = data['image'], data['pose']
        if useCuda:
            images, poses = Variable(images.cuda()), Variable(poses.cuda())
        else:
            images, poses = Variable(images), Variable(poses)

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, poses)
        loss.backward()
        optimizer.step()

        train_loss_epoch.append(loss.data[0])

    if epoch%10==0:
        PATH_PREFIX = '/home/yuliang/code/DeepPose-pytorch/models/czx/'
        checkpoint_file = PATH_PREFIX + 'checkpoint{}.t7'.format(epoch)
        torch.save(net, checkpoint_file)
        print('==> checkpoint model saving to %s'%checkpoint_file)

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
          (epoch + 1, np.mean(np.array(train_loss_epoch)), np.mean(np.array(valid_loss_epoch))))
        
        with open(PATH_PREFIX+"mobile-log.txt", 'a+') as file_output:
            file_output.write('[epoch %d] train loss: %.8f, valid loss: %.8f\n' %
              (epoch + 1, np.mean(np.array(train_loss_epoch)), np.mean(np.array(valid_loss_epoch))))
            file_output.flush() 

print('Finished Training')