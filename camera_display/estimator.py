import itertools
import logging
import math
from collections import namedtuple

import cv2
import numpy as np
import tensorflow as tf
import torch

from scipy.ndimage import maximum_filter, gaussian_filter
from skimage import io, transform

from Net import Net
from torch.autograd import Variable


logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class ResEstimator:
    def __init__(self, graph_path, target_size=(320, 240)):
        self.target_size = target_size
        self.graph_path = graph_path
        self.net = torch.load(graph_path)
        self.net.eval()

    def rescale(self, image, output_size):
        image_ = image
        h, w = image_.shape[:2]
        if isinstance(output_size, int):
            if h > w:
                new_h, new_w = output_size * h / w, output_size
            else:
                new_h, new_w = output_size, output_size * w / h
        else:
            new_h, new_w = output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image_, (new_h, new_w))
        pose_fun = lambda x: (x.reshape([-1,2]) * 1.0 /np.array([new_w, new_h])*np.array([w,h]))
        return {'image': image, 'pose_fun': pose_fun}

    def to_tensor(self, image):
        mean=np.array([0.485, 0.456, 0.406])
        std=np.array([0.229, 0.224, 0.225])
        image = torch.from_numpy(((image-mean)/std).transpose((2, 0, 1))).float()
        return image

    def inference(self, in_npimg, scales=None):
        logger.debug('inference+')
        canvas = np.zeros_like(in_npimg)
        height = canvas.shape[0]
        width = canvas.shape[1]


        print("width=%d, height=%d" % (width, height))
        print("type(in_npimg)", type(in_npimg))
        rescale_out = self.rescale(in_npimg, (227,227))
        image = rescale_out['image']
        image = self.to_tensor(image)
        image = image.unsqueeze(0)
        pose_fun = rescale_out['pose_fun']

        keypoints = self.net(Variable(image))
        print(keypoints)

        keypoints = keypoints.data.cpu().numpy()
        keypoints = pose_fun(keypoints).astype(int)

        return keypoints


    @staticmethod
    def draw_humans(npimg, pose, imgcopy=False):
        if imgcopy:
            npimg = np.copy(npimg)
        image_h, image_w = npimg.shape[:2]
        centers = {}

        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255]]

        pairs = [[8,9],[11,12],[11,10],[2,1],[1,0],[13,14],[14,15],[3,4],[4,5],[8,7],[7,6],[6,2],[6,3],[8,12],[8,13]]
        colors_skeleton = ['r', 'y', 'y', 'g', 'g', 'y', 'y', 'g', 'g', 'm', 'm', 'g', 'g', 'y','y']
        colors_skeleton = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255]]

        for idx in range(len(colors)):
            cv2.circle(npimg, (pose[idx,0], pose[idx,1]), 3, colors[idx], thickness=3, lineType=8, shift=0)
        for idx in range(len(colors_skeleton)):
            npimg = cv2.line(npimg, (pose[pairs[idx][0],0], pose[pairs[idx][0],1]), (pose[pairs[idx][1],0], pose[pairs[idx][1],1]), colors_skeleton[idx], 3)
    
        return npimg

