# coding: utf-8
'''
File: training.py
Project: MobilePose-PyTorch
File Created: Friday, 8th March 2019 6:53:13 pm
Author: Yuliang Xiu (yuliangxiu@sjtu.edu.cn)
-----
Last Modified: Monday, 11th March 2019 12:50:27 am
Modified By: Yuliang Xiu (yuliangxiu@sjtu.edu.cn>)
-----
Copyright 2018 - 2019 Shanghai Jiao Tong University, Machine Vision and Intelligence Group
'''


# remove warning
import warnings
warnings.filterwarnings('ignore')


from network import *
from dataloader import *
from networks import *
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import *
from dataset_factory import DatasetFactory, ROOT_DIR
import os
import multiprocessing
from tqdm import tqdm


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MobilePose Demo')
    parser.add_argument('--model', type=str, default="resnet")
    parser.add_argument('--gpu', type=str, default="")
    parser.add_argument('--inputsize', type=int, default=224)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--t7', type=str, default="")

    args = parser.parse_args()
    modeltype = args.model

    device = torch.device("cuda:0" if len(args.gpu)>1 else "cuda")

    # user defined parameters
    num_threads = int(multiprocessing.cpu_count()/2)
    minloss = np.float("inf")
    # minloss = 0.43162785

    # gpu setting
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    torch.backends.cudnn.enabled = True
    cudnn.benchmark = True

    print("GPU NUM: %d"%(torch.cuda.device_count()))
    net = CoordRegressionNetwork(n_locations=16, backbone=modeltype).to(device)
    net = torch.nn.DataParallel(net).to(device)

    learning_rate = args.lr
    batchsize = args.batchsize
    inputsize = args.inputsize
    modelname = "%s_%d"%(modeltype,inputsize)


    logname = modeltype+'-log.txt'

    if args.t7 != "":
        # load pretrain model
        pre_net = torch.load(args.t7)
        net.module.load_state_dict(pre_net)
        
        for param in list(net.parameters()):
            param.requires_grad = True

    net = net.train()

    PATH_PREFIX = './models' # path to save the model

    train_dataset = DatasetFactory.get_train_dataset(modeltype, inputsize)
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize,
                            shuffle=True, num_workers = num_threads)


    test_dataset = DatasetFactory.get_test_dataset(modeltype, inputsize)
    test_dataloader = DataLoader(test_dataset, batch_size=batchsize,
                            shuffle=False, num_workers = num_threads)


    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = optim.RMSprop(net.parameters(), lr=learning_rate)

    scheduler = StepLR(optimizer, step_size=80, gamma=0.5)


    train_loss_all = []
    valid_loss_all = []

    for epoch in range(1000):  # loop over the dataset multiple times
        
        train_loss_epoch = []
        train_loss_epoch_coords = []
        train_loss_epoch_hm = []

        scheduler.step()

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

            del data, images, poses, coords, heatmaps
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_epoch.append(loss.item())
            train_loss_epoch_coords.append(torch.mean(euc_losses).item())
            train_loss_epoch_hm.append(torch.mean(reg_losses).item())

        if epoch%2==0:

            valid_loss_epoch = []
            valid_loss_epoch_coords = []
            valid_loss_epoch_hm = []

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

                    del sample_batched, images, poses, coords, heatmaps

                    valid_loss_epoch.append(loss.item())
                    valid_loss_epoch_coords.append(torch.mean(euc_losses).item())
                    valid_loss_epoch_hm.append(torch.mean(reg_losses).item())

            if np.mean(np.array(valid_loss_epoch)) < minloss:
                # save the model
                minloss = np.mean(np.array(valid_loss_epoch))
                checkpoint_file = "%s/%s_%.4f.t7"%(PATH_PREFIX, modelname, minloss)
                checkpoint_best_file = "%s/%s_adam_best.t7"%(PATH_PREFIX, modelname)
                # torch.save(net, checkpoint_file)
                torch.save(net.module.state_dict(), checkpoint_best_file)
                print('==> checkpoint model saving to %s and %s'%(checkpoint_file, checkpoint_best_file))

            print('[epoch %d] train loss(coords): %.8f, train loss(hm): %.8f, train loss: %.8f,\n          valid loss(coords): %.8f, valid loss(hm): %.8f, valid loss: %.8f\n' %
                (epoch + 1, np.mean(np.array(train_loss_epoch_coords)), np.mean(np.array(train_loss_epoch_hm)), np.mean(np.array(train_loss_epoch)), 
                 np.mean(np.array(valid_loss_epoch_coords)), np.mean(np.array(valid_loss_epoch_hm)), np.mean(np.array(valid_loss_epoch))))

            # write the log of the training process
            if not os.path.exists(PATH_PREFIX):
                os.makedirs(PATH_PREFIX)

            with open(os.path.join(PATH_PREFIX,logname), 'a+') as file_output:
                file_output.write('[epoch %d] train loss(coords): %.8f, train loss(hm): %.8f, train loss: %.8f,\n          valid loss(coords): %.8f, valid loss(hm): %.8f, valid loss: %.8f\n' %
                (epoch + 1, np.mean(np.array(train_loss_epoch_coords)), np.mean(np.array(train_loss_epoch_hm)), np.mean(np.array(train_loss_epoch)), 
                 np.mean(np.array(valid_loss_epoch_coords)), np.mean(np.array(valid_loss_epoch_hm)), np.mean(np.array(valid_loss_epoch))))
                file_output.flush() 
                
    print('Finished Training')
