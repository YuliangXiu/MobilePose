# coding: utf-8

'''
File: training.py
Project: MobilePose
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

from networks import *
from dataloader import *
from mobilenetv2_heatmap import HeatmapMobileNetV2
import argparse
import alog

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_factory import DatasetFactory


# def one_hot(output, pose):
def pp(obj):
    alog.info(type(obj))
    alog.info(obj.shape)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MobilePose Demo')
    parser.add_argument('--model', type=str, default="mobilenet-heatmap")
    parser.add_argument('--gpu', type=str, default="1")
    parser.add_argument('--retrain', type=bool, default=True)
    args = parser.parse_args()
    modeltype = args.model

    # user defined parameters
    num_threads = 10

    if modeltype == 'resnet':
        modelname = "final-aug.t7"
        batchsize = 128
        minloss = 316.52189376  # changed expand ratio
        # minloss = 272.49565467 #fixed expand ratio
        learning_rate = 1e-05
        net = Net().cuda()
        inputsize = 224
    elif modeltype == "mobilenet":
        modelname = "final-aug.t7"
        batchsize = 128
        minloss = 396.84708708  # change expand ratio
        # minloss = 332.48316225 # fixed expand ratio
        learning_rate = 1e-06
        net = MobileNetV2(image_channel=5).cuda()
        inputsize = 224
    elif modeltype == "mobilenet-heatmap":
        modelname = "final-aug.t7"
        batchsize = 2
        minloss = 396.84708708  # change expand ratio
        # minloss = 332.48316225 # fixed expand ratio
        learning_rate = 1e-06
        net = HeatmapMobileNetV2().cuda()
        inputsize = 224

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.backends.cudnn.enabled = True
    print("GPU NUM: %d" % (torch.cuda.device_count()))

    logname = modeltype + '-log.txt'

    if not args.retrain:
        # load pretrain model
        # net = torch.load('./models/%s/%s'%(modeltype,modelname)).cuda()
        net = torch.load('./models/%s/%s' % (modeltype, modelname)).cuda()
    # alog.info(net)
    net = net.train()

    ROOT_DIR = "../deeppose_tf/datasets/mpii"  # root dir to the dataset
    PATH_PREFIX = './models/{}/'.format(modeltype)  # path to save the model

    tmp_modeltype = "resnet"
    train_dataset = DatasetFactory.get_train_dataset(tmp_modeltype, inputsize)
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize,
                                  shuffle=False, num_workers=num_threads)

    test_dataset = DatasetFactory.get_test_dataset(tmp_modeltype, inputsize)
    test_dataloader = DataLoader(test_dataset, batch_size=batchsize,
                                 shuffle=False, num_workers=num_threads)

    criterion = nn.MSELoss().cuda()
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, momentum=0.9)


    def mse_loss(input, target):
        return torch.sum(torch.pow(input - target, 2)) / input.nelement()


    train_loss_all = []
    valid_loss_all = []

    for epoch in range(1000):  # loop over the dataset multiple times

        train_loss_epoch = []
        for i, data in enumerate(train_dataloader):
            # training
            images, poses = data['image'], data['pose']
            guass_heatmap = data['guass_heatmap']

            images, poses = Variable(images.cuda()), Variable(poses.cuda())
            optimizer.zero_grad()


            outputs = net(images)
            output_heatmap = nn.UpsamplingBilinear2d((inputsize, inputsize))(outputs)
            guass_heatmap = Variable(guass_heatmap.cuda())

            loss = criterion(output_heatmap, guass_heatmap)
            loss.backward()
            optimizer.step()

            train_loss_epoch.append(loss.data[0])

        if epoch % 2 == 0:
            valid_loss_epoch = []
            for i_batch, sample_batched in enumerate(test_dataloader):
                # calculate the valid loss
                net_forward = net
                images = sample_batched['image'].cuda()
                poses = sample_batched['pose'].cuda()
                guass_heatmap = sample_batched['guass_heatmap'].cuda()

                outputs = net_forward(Variable(images, volatile=True))
                output_heatmap = nn.UpsamplingBilinear2d((inputsize, inputsize))(outputs)
                guass_heatmap = Variable(guass_heatmap.cuda())

                loss = criterion(output_heatmap, guass_heatmap)

                valid_loss_epoch.append(loss)

            if np.mean(np.array(valid_loss_epoch)) < minloss:
                # save the model
                minloss = np.mean(np.array(valid_loss_epoch))
                checkpoint_file = PATH_PREFIX + modelname
                torch.save(net, checkpoint_file)
                print('==> checkpoint model saving to %s' % checkpoint_file)

            print('[epoch %d] train loss: %.8f, valid loss: %.8f' %
                  (epoch + 1, np.mean(np.array(train_loss_epoch)),
                   np.mean(np.array(valid_loss_epoch))))
            # write the log of the training process
            with open(PATH_PREFIX + logname, 'a+') as file_output:
                file_output.write(
                    '[epoch %d] train loss: %.8f, valid loss: %.8f\n' %
                    (epoch + 1, np.mean(np.array(train_loss_epoch)),
                     np.mean(np.array(valid_loss_epoch))))
                file_output.flush()

    print('Finished Training')
