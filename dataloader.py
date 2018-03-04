import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils, models
from torch.autograd import Variable
import csv
import numpy as np
import os
from skimage import io, transform
import cv2


def expand_bbox(left, right, top, bottom, img_width, img_height):
    width = right-left
    height = bottom-top
    ratio = 0.1
    new_left = np.clip(left-ratio*width,0,img_width)
    new_right = np.clip(right+ratio*width,0,img_width)
    new_top = np.clip(top-ratio*height,0,img_height)
    new_bottom = np.clip(bottom+ratio*height,0,img_height)
    
    return [int(new_left), int(new_top), int(new_right), int(new_bottom)]

class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image_, pose_ = sample['image'], sample['pose']
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
        pose = pose.flatten()

        return {'image': image, 'pose': pose}

class Expansion(object):
    
    def __call__(self, sample):
        image, pose = sample['image'], sample['pose']
        h, w = image.shape[:2]
        x = np.arange(0, h)
        y = np.arange(0, w) 
        x, y = np.meshgrid(x, y)
        x = x[:,:, np.newaxis]
        y = y[:,:, np.newaxis]
        image = np.concatenate((image, x), axis=2)
        image = np.concatenate((image, y), axis=2)
        
        return {'image': image,
                'pose': pose}
    
class ToTensor(object):

    def __call__(self, sample):
        image, pose = sample['image'], sample['pose']
        mean=np.array([0.485, 0.456, 0.406])
        std=np.array([0.229, 0.224, 0.225])
        image[:,:,:3] = (image[:,:,:3]-mean)/std
        image = torch.from_numpy(image.transpose((2, 0, 1))).float()
        pose = torch.from_numpy(pose).float()
        
        return {'image': image,
                'pose': pose}

class PoseDataset(Dataset):

    def __init__(self, csv_file, transform):
        
        with open(csv_file) as f:
            self.f_csv = list(csv.reader(f, delimiter='\t'))
        self.transform = transform

    def __len__(self):
        return len(self.f_csv)
        
    def __getitem__(self, idx):
        ROOT_DIR = "/home/yuliang/code/deeppose_tf/datasets/mpii"
        line = self.f_csv[idx][0].split(",")
        img_path = os.path.join(ROOT_DIR,'images',line[0])
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

        # augmentation choices
        seq = iaa.SomeOf(3, [
            iaa.Sometimes(0.2, iaa.Scale((0.5, 1.0))),
            iaa.Sometimes(0.7, iaa.CropAndPad(percent=(-0.25, 0.25), pad_mode=["edge"], keep_size=False)),
            iaa.Fliplr(0.4), 
            iaa.Sometimes(0.3, iaa.AdditiveGaussianNoise(scale=(0, 0.05*50))),
            iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 3.0)))
        ])
        seq_det = seq.to_deterministic()

        image_aug = seq_det.augment_images([image])[0]
        keypoints_aug = seq_det.augment_keypoints([self.pose2keypoints(image,pose)])[0]

        return {'image': image_aug/256.0, 'pose': self.keypoints2pose(keypoints_aug)}