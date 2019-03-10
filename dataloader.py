'''
File: dataloader.py
Project: MobilePose-PyTorch
File Created: Tuesday, 15th January 2019 6:26:25 pm
Author: Yuliang Xiu (yuliangxiu@sjtu.edu.cn)
-----
Last Modified: Monday, 11th March 2019 12:51:19 am
Modified By: Yuliang Xiu (yuliangxiu@sjtu.edu.cn>)
-----
Copyright 2018 - 2019 Shanghai Jiao Tong University, Machine Vision and Intelligence Group
'''


import csv
import numpy as np
import os
from skimage import io, transform
import cv2

import torch
import alog
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils, models

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def crop_camera(image, ratio=0.15):
    height = image.shape[0]
    width = image.shape[1]
    mid_width = width / 2.0
    width_20 = width * ratio
    crop_img = image[0:int(height), int(mid_width - width_20):int(mid_width + width_20)]
    return crop_img

def display_pose( img, pose, ids):
    
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    pose  = pose.data.cpu().numpy()
    img = img.cpu().numpy().transpose(1,2,0)
    colors = ['g', 'g', 'g', 'g', 'g', 'g', 'm', 'm', 'r', 'r', 'y', 'y', 'y', 'y','y','y']
    pairs = [[8,9],[11,12],[11,10],[2,1],[1,0],[13,14],[14,15],[3,4],[4,5],[8,7],[7,6],[6,2],[6,3],[8,12],[8,13]]
    colors_skeleton = ['r', 'y', 'y', 'g', 'g', 'y', 'y', 'g', 'g', 'm', 'm', 'g', 'g', 'y','y']
    img = np.clip(img*std+mean, 0.0, 1.0)
    img_width, img_height,_ = img.shape
    pose = ((pose + 1)* np.array([img_width, img_height])-1)/2 # pose ~ [-1,1]

    plt.subplot(25,4,ids+1)
    ax = plt.gca()
    plt.imshow(img)
    for idx in range(len(colors)):
        plt.plot(pose[idx,0], pose[idx,1], marker='o', color=colors[idx])
    for idx in range(len(colors_skeleton)):
        plt.plot(pose[pairs[idx],0], pose[pairs[idx],1],color=colors_skeleton[idx])

    xmin = np.min(pose[:,0])
    ymin = np.min(pose[:,1])
    xmax = np.max(pose[:,0])
    ymax = np.max(pose[:,1])

    bndbox = np.array(expand_bbox(xmin, xmax, ymin, ymax, img_width, img_height))
    coords = (bndbox[0], bndbox[1]), bndbox[2]-bndbox[0]+1, bndbox[3]-bndbox[1]+1
    ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='yellow', linewidth=1))

def expand_bbox(left, right, top, bottom, img_width, img_height):
    width = right-left
    height = bottom-top
    ratio = 0.15
    new_left = np.clip(left-ratio*width,0,img_width)
    new_right = np.clip(right+ratio*width,0,img_width)
    new_top = np.clip(top-ratio*height,0,img_height)
    new_bottom = np.clip(bottom+ratio*height,0,img_height)

    return [int(new_left), int(new_top), int(new_right), int(new_bottom)]

# Rescale implementation of mobilenetV2

class Wrap(object):
    
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):

        image_, pose_ = sample['image']/256.0, sample['pose']
        h, w = image_.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image_, (new_w, new_h))
        pose = pose_.reshape([-1,2])/np.array([w,h])
        pose *= -1.0

        return {'image': image, 'pose': pose}


# Rescale implementation of Resnet18

class Rescale(object):


    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image_, pose_ = sample['image']/256.0, sample['pose']
        h, w = image_.shape[:2]
        im_scale = min(float(self.output_size[0]) / float(h), float(self.output_size[1]) / float(w))
        new_h = int(image_.shape[0] * im_scale)
        new_w = int(image_.shape[1] * im_scale)
        image = cv2.resize(image_, (new_w, new_h),
                    interpolation=cv2.INTER_LINEAR)
        left_pad = (self.output_size[1] - new_w) // 2
        right_pad = (self.output_size[1] - new_w) - left_pad
        top_pad = (self.output_size[0] - new_h) // 2
        bottom_pad = (self.output_size[0] - new_h) - top_pad
        mean=np.array([0.485, 0.456, 0.406])
        pad = ((top_pad, bottom_pad), (left_pad, right_pad))
        image = np.stack([np.pad(image[:,:,c], pad, mode='constant', constant_values=mean[c]) 
                        for c in range(3)], axis=2)
        pose = (pose_.reshape([-1,2])/np.array([w,h])*np.array([new_w,new_h]))
        pose += [left_pad, top_pad]
        pose = (pose * 2 + 1) / self.output_size - 1 # pose ~ [-1,1]

        return {'image': image, 'pose': pose}


class Expansion(object):
    
    def __call__(self, sample):
        image, pose = sample['image'], sample['pose']
        h, w = image.shape[:2]
        x = np.arange(0, h)
        y = np.arange(0, w) 
        x, y = np.meshgrid(x, y)
        x = x[:,:, np.newaxis]/h
        y = y[:,:, np.newaxis]/w
        image = np.concatenate((image, x, y), axis=2)
        
        return {'image': image,
                'pose': pose}
    
class ToTensor(object):

    def __call__(self, sample):
        image, pose = sample['image'], sample['pose']
		# todo: support heatmap
        # guass_heatmap = sample['guass_heatmap']
        h, w = image.shape[:2]

        mean=np.array([0.485, 0.456, 0.406])
        std=np.array([0.229, 0.224, 0.225])

        image[:,:,:3] = (image[:,:,:3]-mean)/(std)
        image = torch.from_numpy(image.transpose((2, 0, 1))).float()
        pose = torch.from_numpy(pose).float()

		# todo: support heatmap
	    # guass_heatmap = torch.from_numpy(guass_heatmap).float()
        return {'image': image,
                'pose': pose}
        #return {'image': image,
        #        'pose': pose,
        #        'guass_heatmap': guass_heatmap}

class PoseDataset(Dataset):

    def __init__(self, csv_file, transform):
        self.root = os.path.dirname(csv_file)
        with open(csv_file) as f:
            self.f_csv = list(csv.reader(f, delimiter='\t'))
        self.transform = transform

    def __len__(self):
        return len(self.f_csv)
        
    def __getitem__(self, idx):
        line = self.f_csv[idx][0].split(",")
        img_path = os.path.join(self.root,'images',line[0])
        image = io.imread(img_path)
        height, width = image.shape[0], image.shape[1]
        pose = np.array([float(item) for item in line[1:]]).reshape([-1,2])
        
        xmin = np.min(pose[:,0])
        ymin = np.min(pose[:,1])
        xmax = np.max(pose[:,0])
        ymax = np.max(pose[:,1])

        box = expand_bbox(xmin, xmax, ymin, ymax, width, height)
        image = image[box[1]:box[3],box[0]:box[2],:]
        pose = (pose-np.array([box[0],box[1]])).flatten()
        sample = {'image': image, 'pose':pose}
        if self.transform:
            sample = self.transform(sample)
        return sample


import imgaug as ia
from imgaug import augmenters as iaa
from scipy import misc
import copy
import random
from imgaug import parameters as iap

class Augmentation(object):
    
    def pose2keypoints(self, image, pose):
        keypoints = []
        for row in range(int(pose.shape[0])):
            x = pose[row,0]
            y = pose[row,1]
            keypoints.append(ia.Keypoint(x=x, y=y))
        return ia.KeypointsOnImage(keypoints, shape=image.shape)

    def keypoints2pose(self, keypoints_aug):
        one_person = []
        for kp_idx, keypoint in enumerate(keypoints_aug.keypoints):
            x_new, y_new = keypoint.x, keypoint.y
            one_person.append(np.array(x_new).astype(np.float32))
            one_person.append(np.array(y_new).astype(np.float32))
        return np.array(one_person).reshape([-1,2])

    def __call__(self, sample):
        image, pose= sample['image'], sample['pose'].reshape([-1,2])

        sometimes = lambda aug: iaa.Sometimes(0.3, aug)

        seq = iaa.Sequential(
            [
                # Apply the following augmenters to most images.

                sometimes(iaa.CropAndPad(percent=(-0.25, 0.25), pad_mode=["edge"], keep_size=False)),

                sometimes(iaa.Affine(
                    scale={"x": (0.75, 1.25), "y": (0.75, 1.25)},
                    translate_percent={"x": (-0.25, 0.25), "y": (-0.25, 0.25)},
                    rotate=(-45, 45),
                    shear=(-5, 5),
                    order=[0, 1],
                    cval=(0, 255),
                    mode=ia.ALL
                )),

                iaa.SomeOf((0, 3),
                    [
        
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 3.0)),
                            # iaa.AverageBlur(k=(2, 7)),
                            iaa.MedianBlur(k=(3, 11)),
                            iaa.MotionBlur(k=5,angle=[-45, 45])
                        ]),

                        iaa.OneOf([
                            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                            iaa.AdditivePoissonNoise(lam=(0,8), per_channel=True),
                        ]),

                        iaa.OneOf([
                            iaa.Add((-10, 10), per_channel=0.5),
                            iaa.Multiply((0.2, 1.2), per_channel=0.5),
                            iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                        ]),
                    ],
                    # do all of the above augmentations in random order
                    random_order=True
                )
            ],
            # do all of the above augmentations in random order
            random_order=True
        )

        # augmentation choices
        seq_det = seq.to_deterministic()

        image_aug = seq_det.augment_images([image])[0]
        keypoints_aug = seq_det.augment_keypoints([self.pose2keypoints(image,pose)])[0]

        return {'image': image_aug, 'pose': self.keypoints2pose(keypoints_aug)}

# TBD
class OneHot(object):
    def __call__(self, sample):
        image, pose = sample['image'], sample['pose']
        one_hot = torch.zeros_like(torch.from_numpy(image))
        return {'image': image, 'pose': pose}


class Guass(object):
    def __call__(self, sample):
        sigma = 5
        image, pose = sample['image'], sample['pose']
        # create meshgrid
        h, w = image.shape[:2]
        x = np.arange(0, h)
        y = np.arange(0, w)
        x, y = np.meshgrid(x, y)
        # declare guass
        guass_heatmap = np.zeros([len(pose) // 2, image.shape[0], image.shape[1]])
        xy_pose = np.reshape(pose,(-1, 2))
        guass_hrescale = h // 30
        guass_wrescale = w // 30

        for idx,(x0,y0) in enumerate(xy_pose):
            # alog.info(idx)
            guass_heatmap[idx] = np.exp(- (((x - x0) * 1.0 /  guass_hrescale) ** 2 + ((y - y0) * 1.0 / guass_wrescale) ** 2) / (2 * sigma ** 2))
        # alog.info(guass_heatmap.shape)
        return {'image': image, 'pose': pose, 'guass_heatmap': guass_heatmap}
