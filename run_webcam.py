'''
File: run_webcam.py
Project: MobilePose
File Created: Thursday, 8th March 2018 2:19:39 pm
Author: Yuliang Xiu (yuliangxiu@sjtu.edu.cn)
-----
Last Modified: Thursday, 8th March 2018 3:01:35 pm
Modified By: Yuliang Xiu (yuliangxiu@sjtu.edu.cn>)
-----
Copyright 2018 - 2018 Shanghai Jiao Tong University, Machine Vision and Intelligence Group
'''

import argparse
import logging
import time

import cv2
import numpy as np

import torch
import torch.nn as nn
from torchvision import models

from estimator import ResEstimator
import matplotlib.pyplot as plt
from networks import *
from dataloader import crop_camera

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='MobilePose Realtime Webcam.')
    parser.add_argument('--model', type=str, default='resnet', help='mobilenet|resnet')
    parser.add_argument('--camera', type=int, default=0)

    args = parser.parse_args()

    # load the model 
    w, h = model_wh(get_graph_path(args.model))
    e = ResEstimator(get_graph_path(args.model), target_size=(w,h))
    # initial the camera
    cam = cv2.VideoCapture(args.camera)

    ret_val, image = cam.read()
    image = crop_camera(image)

    while True:
        # read image from the camera and preprocess
        ret_val , image = cam.read()
        image = crop_camera(image)
        # forward the image
        humans = e.inference(image, args.model)
        image = ResEstimator.draw_humans(image, humans, imgcopy=False)
        cv2.imshow('MobilePose Demo', image)
        if cv2.waitKey(1) == 27: # ESC
            break

    cv2.destroyAllWindows()
