import argparse
import logging
import time

import cv2
import numpy as np

import torch
import torch.nn as nn
from torchvision import models

from estimator import  ResEstimator
from networks import get_graph_path, model_wh

import matplotlib.pyplot as plt

from Net import Net

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


import torch
import torch.nn as nn
from torchvision import models


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = True
        model.fc=nn.Linear(512,32)
        self.resnet = model

    def forward(self, x):
        pose_out = self.resnet(x)
        return pose_out

def crop20(image):
    height = image.shape[0]
    width = image.shape[1]
    mid_width = width / 2
    width_20 = width * 0.2

    crop_img = image[0:int(height), int(mid_width - width_20):int(mid_width + width_20)]
    return crop_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--model', type=str, default='res_227x227', help='res_227x227 / cmu_640x480 / cmu_640x360 / mobilenet_thin_432x368')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.model)
    e = ResEstimator(get_graph_path(args.model), target_size=(w,h))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)

    ret_val, image = cam.read()
    image = crop20(image)
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    flag = True
    while True:
        ret_val , image = cam.read()
        image = crop20(image)
        logger.debug('image preprocess+')
        logger.debug('image process+')
        humans = e.inference(image)
        logger.debug('postprocess+')
        image = ResEstimator.draw_humans(image, humans, imgcopy=False)

        logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')
        flag = False
    cv2.destroyAllWindows()
