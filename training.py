# coding: utf-8

'''
File: training.py
Project: DeepPose
File Created: Thursday, 8th March 2018 2:50:11 pm
Author: Yuliang Xiu (yuliangxiu@sjtu.edu.cn)
-----
Last Modified: Thursday, 8th March 2018 2:50:51 pm
Modified By: Yuliang Xiu (yuliangxiu@sjtu.edu.cn>)
-----
Copyright 2018 - 2018 Shanghai Jiao Tong University, Machine Vision and Intelligence Group
'''

# remove warning
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
from networks import *
from dataloader import *
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='deeppose demo')
    parser.add_argument('--model', type=str, default="resnet")
    parser.add_argument('--gpu', type=str, default="0")
    args = parser.parse_args()
    modeltype = args.model

    # user defined parameters
    num_threads = 10

    if modeltype =='resnet':
        modelname = "final-aug.t7"
        pretrain = True
        batchsize = 256
        minloss = 316.52189376 #changed expand ratio
        # minloss = 272.49565467 #fixed expand ratio
        learning_rate = 1e-05
        net = Net().cuda()
        inputsize = 227
    elif modeltype == "mobilenet":
        modelname = "final-noaug.t7"
        pretrain = True
        batchsize = 128
        minloss = 337.44666895
        learning_rate = 1e-06
        net = MobileNetV2(image_channel=5).cuda()
        inputsize = 224

    # gpu setting
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    torch.backends.cudnn.enabled = True
    gpus = [0,1]
    print("GPU NUM: %d"%(torch.cuda.device_count()))

    logname = modeltype+'-log.txt'

    if pretrain:
        net = torch.load('./models/%s/%s'%(modeltype,modelname)).cuda(device_id=gpus[0])

    ROOT_DIR = "/home/yuliang/code/deeppose_tf/datasets/mpii"
    PATH_PREFIX = '/home/yuliang/code/DeepPose-pytorch/models/{}/'.format(modeltype)

    train_dataset = PoseDataset(csv_file=os.path.join(ROOT_DIR,'train_joints.csv'),
                                    transform=transforms.Compose([
                                                # Augmentation(),
                                                Rescale((inputsize,inputsize)),
                                                # Wrap((inputsize,inputsize)),
                                                Expansion(),
                                                ToTensor()
                                            ]))
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize,
                            shuffle=False, num_workers = num_threads)

    test_dataset = PoseDataset(csv_file=os.path.join(ROOT_DIR,'test_joints.csv'),
                                    transform=transforms.Compose([
                                                Rescale((inputsize,inputsize)),
                                                # Wrap((inputsize, inputsize)),
                                                Expansion(),
                                                ToTensor()
                                            ]))
    test_dataloader = DataLoader(test_dataset, batch_size=batchsize,
                            shuffle=False, num_workers = num_threads)


    criterion = nn.MSELoss().cuda()
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)


    def mse_loss(input, target):
        return torch.sum(torch.pow(input - target,2)) / input.nelement()

    train_loss_all = []
    valid_loss_all = []

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
                checkpoint_file = PATH_PREFIX + modelname
                torch.save(net, checkpoint_file)
                print('==> checkpoint model saving to %s'%checkpoint_file)

            print('[epoch %d] train loss: %.8f, valid loss: %.8f' %
            (epoch + 1, np.mean(np.array(train_loss_epoch)), np.mean(np.array(valid_loss_epoch))))
            with open(PATH_PREFIX+logname, 'a+') as file_output:
                file_output.write('[epoch %d] train loss: %.8f, valid loss: %.8f\n' %
                (epoch + 1, np.mean(np.array(train_loss_epoch)), np.mean(np.array(valid_loss_epoch))))
                file_output.flush() 
                
    print('Finished Training')