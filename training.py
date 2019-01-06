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
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_factory import DatasetFactory, ROOT_DIR
import os
import multiprocessing
from tqdm import tqdm


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MobilePose Demo')
    parser.add_argument('--model', type=str, default="resnet")
    parser.add_argument('--gpu', type=str, default="")
    parser.add_argument('--t7', type=str, default="")

    args = parser.parse_args()
    modeltype = args.model

    device = torch.device("cuda" if len(args.gpu)>0 else "cpu")

    # user defined parameters
    num_threads = multiprocessing.cpu_count()
    minloss = np.float("inf")

    if "resnet" in modeltype:

        learning_rate = 1e-5
        if "18" in modeltype:
            batchsize = 128 # 186
            net = CoordRegressionNetwork(n_locations=16, layers=18).to(device)
        elif "34" in modeltype:
            batchsize = 64
            net = CoordRegressionNetwork(n_locations=16, layers=34).to(device)
        inputsize = 224
        modelname = "%s_%d"%(modeltype,inputsize)

    elif modeltype == "mobilenet":

        batchsize = 128
        learning_rate = 1e-06
        net = MobileNetV2(image_channel=3).to(device)
        inputsize = 224
        modelname =  "%s_%d"%(modeltype,inputsize)


    # gpu setting
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    torch.backends.cudnn.enabled = True
    gpus = [0,1]
    print("GPU NUM: %d"%(torch.cuda.device_count()))


    logname = modeltype+'-log.txt'

    if args.t7 != "":
        # load pretrain model
        net = torch.load(args.t7).to(device)

    net = net.train()

    PATH_PREFIX = './models' # path to save the model

    train_dataset = DatasetFactory.get_train_dataset(modeltype, inputsize)
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize,
                            shuffle=True, num_workers = num_threads)


    test_dataset = DatasetFactory.get_test_dataset(modeltype, inputsize)
    test_dataloader = DataLoader(test_dataset, batch_size=batchsize,
                            shuffle=False, num_workers = num_threads)


    criterion = nn.MSELoss().to(device)
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    train_loss_all = []
    valid_loss_all = []

    for epoch in range(1000):  # loop over the dataset multiple times
        
        train_loss_epoch = []
        for i, data in enumerate(tqdm(train_dataloader)):
            # training
            images, poses = data['image'], data['pose']
            images, poses = images.to(device), poses.to(device)
            coords, heatmaps = net(images)

            # Per-location euclidean losses
            euc_losses = dsntnn.euclidean_losses(coords, poses)
            # Per-location regularization losses
            reg_losses = dsntnn.js_reg_losses(heatmaps, poses, sigma_t=1.0)
            # Combine losses into an overall loss
            loss = dsntnn.average_loss(euc_losses + reg_losses)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_epoch.append(loss.item())

        if epoch%2==0:

            valid_loss_epoch = []

            with torch.no_grad():  
                for i_batch, sample_batched in enumerate(tqdm(test_dataloader)):
                    # calculate the valid loss
                    images = sample_batched['image'].to(device)
                    poses = sample_batched['pose'].to(device)
                    coords, heatmaps = net(images)

                    # Per-location euclidean losses
                    euc_losses = dsntnn.euclidean_losses(coords, poses)
                    # Per-location regularization losses
                    reg_losses = dsntnn.js_reg_losses(heatmaps, poses, sigma_t=1.0)
                    # Combine losses into an overall loss
                    loss = dsntnn.average_loss(euc_losses + reg_losses)

                    valid_loss_epoch.append(loss.item())

            if np.mean(np.array(valid_loss_epoch)) < minloss:
                # save the model
                minloss = np.mean(np.array(valid_loss_epoch))
                checkpoint_file = "%s/%s_%.4f.t7"%(PATH_PREFIX, modelname, minloss)
                checkpoint_best_file = "%s/%s_best.t7"%(PATH_PREFIX, modelname)
                # torch.save(net, checkpoint_file)
                torch.save(net, checkpoint_best_file)
                print('==> checkpoint model saving to %s and %s'%(checkpoint_file, checkpoint_best_file))

            print('[epoch %d] train loss: %.8f, valid loss: %.8f' %
                (epoch + 1, np.mean(np.array(train_loss_epoch)), np.mean(np.array(valid_loss_epoch))))

            # write the log of the training process
            if not os.path.exists(PATH_PREFIX):
                os.makedirs(PATH_PREFIX)

            with open(os.path.join(PATH_PREFIX,logname), 'a+') as file_output:
                file_output.write('[epoch %d] train loss: %.8f, valid loss: %.8f\n' %
                (epoch + 1, np.mean(np.array(train_loss_epoch)), np.mean(np.array(valid_loss_epoch))))
                file_output.flush() 
                
    print('Finished Training')
