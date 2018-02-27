
# coding: utf-8

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms, utils, models
from torch.autograd import Variable

from skimage import io, transform
import numpy as np
import csv
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings('ignore')
# warnings.filterwarnings(action='once')


# from multiprocessing import set_start_method
# try:
#     set_start_method('spawn')
# except RuntimeError:
#     pass

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
# os.environ["CUDA_VISIBLE_DEVICES"]=""
# os.environ["CUDA_LAUNCH_BLOCKING"]="1"
# torch.cuda.set_device(1) 
torch.backends.cudnn.enabled = True
print(torch.cuda.device_count())
gpus = [1,2]


# In[2]:


def expand_bbox(left, right, top, bottom, img_width, img_height):
    width = right-left
    height = bottom-top
    ratio = 0.15
    new_left = np.clip(left-ratio*width,0,img_width)
    new_right = np.clip(right+ratio*width,0,img_width)
    new_top = np.clip(top-ratio*height,0,img_height)
    new_bottom = np.clip(bottom+ratio*height,0,img_height)
    return [int(new_left), int(new_top), int(new_right), int(new_bottom)]

def display_pose(img, pose):
    pose  = pose.data.cpu().numpy().reshape([-1,2])
    img = img.cpu().numpy().transpose(1,2,0)
    img_width, img_height,_ = img.shape
    ax = plt.gca()
    plt.imshow(img)
    for idx in range(16):
        plt.plot(pose[idx,0], pose[idx,1], marker='o', color='yellow')
    xmin = np.min(pose[:,0])
    ymin = np.min(pose[:,1])
    xmax = np.max(pose[:,0])
    ymax = np.max(pose[:,1])
    bndbox = np.array(expand_bbox(xmin, xmax, ymin, ymax, img_width, img_height))
    coords = (bndbox[0], bndbox[1]), bndbox[2]-bndbox[0]+1, bndbox[3]-bndbox[1]+1
    ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='yellow', linewidth=2))


# In[3]:


get_ipython().run_cell_magic('sh', '', 'jupyter nbextension enable --py --sys-prefix widgetsnbextension')


# In[4]:


import cv2
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
        mean=np.array([0.485, 0.456, 0.406]) * 256
        pad = ((top_pad, bottom_pad), (left_pad, right_pad))
        image = np.stack([np.pad(image[:,:,c], pad, mode='constant', constant_values=mean[c]) 
                        for c in range(3)], axis=2)
        pose = (pose_.reshape([-1,2])/np.array([w,h])*np.array([new_w,new_h]))
        pose += [right_pad, top_pad]
        pose = pose.flatten()
        return {'image': image, 'pose': pose}

#     def __call__(self, sample):
#         image_, pose_ = sample['image'], sample['pose']

#         h, w = image_.shape[:2]
#         if isinstance(self.output_size, int):
#             if h > w:
#                 new_h, new_w = self.output_size * h / w, self.output_size
#             else:
#                 new_h, new_w = self.output_size, self.output_size * w / h
#         else:
#             new_h, new_w = self.output_size

#         new_h, new_w = int(new_h), int(new_w)

#         image = transform.resize(image_, (new_h, new_w))
#         pose = (pose_.reshape([-1,2])/np.array([w,h])*np.array([new_w,new_h])).flatten()
#         return {'image': image, 'pose': pose}

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
 
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        mean=np.array([0.485, 0.456, 0.406])
        std=np.array([0.229, 0.224, 0.225])
        image = (image[:,:,:3]-mean)/std
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


# In[5]:


ROOT_DIR = "/home/yuliang/code/deeppose_tf/datasets/mpii"

train_dataset = PoseDataset(csv_file=os.path.join(ROOT_DIR,'train_joints.csv'),
                                  transform=transforms.Compose([
                                               Rescale((227,227)),
                                               #Expansion(),
                                               ToTensor()
                                           ]))
train_dataloader = DataLoader(train_dataset, batch_size=128,
                        shuffle=False, num_workers = 10)

test_dataset = PoseDataset(csv_file=os.path.join(ROOT_DIR,'test_joints.csv'),
                                  transform=transforms.Compose([
                                               Rescale((227,227)),
                                               #Expansion(),
                                               ToTensor()
                                           ]))
test_dataloader = DataLoader(test_dataset, batch_size=128,
                        shuffle=False, num_workers = 10)


# In[6]:


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = True
        model.fc=nn.Linear(512,32)
        self.resnet = model.cuda()
        
    def forward(self, x):
       
        pose_out = self.resnet(x)
        return pose_out

# net = torch.load('checkpoint20.t7').cuda(device_id=gpus[1])
net = Net()
net = torch.load('/disk3/yinghong/data/mobile-model/scale2-checkpoint20.t7')
criterion = nn.MSELoss().cuda()
# optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001, momentum=0.9)
optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)


# In[7]:


from tqdm import tqdm_notebook

def mse_loss(input, target):
    return torch.sum(torch.pow(input - target,2)) / input.nelement()

train_loss_all = []
valid_loss_all = []

for epoch in tqdm_notebook(range(1000)):  # loop over the dataset multiple times
    
    train_loss_epoch = []
    for i, data in enumerate(train_dataloader):
#         if i % 20 == 0:
#             print ("i=%d in train_dataloader" % i)
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
        checkpoint_file = PATH_PREFIX + 'scale2-checkpoint{}.t7'.format(20+epoch)
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

