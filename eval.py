# coding: utf-8
'''
File: eval.py
Project: DeepPose
File Created: Thursday, 8th March 2018 1:54:07 pm
Author: Yuliang Xiu (yuliangxiu@sjtu.edu.cn)
-----
Last Modified: Thursday, 8th March 2018 3:01:51 pm
Modified By: Yuliang Xiu (yuliangxiu@sjtu.edu.cn>)
-----
Copyright 2018 - 2018 Shanghai Jiao Tong University, Machine Vision and Intelligence Group
'''
import warnings
warnings.filterwarnings('ignore')

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils, models
from tqdm import tqdm
from skimage import io, transform
from math import ceil
import numpy as np
import torch
import csv
import os
import argparse
import time

from dataloader import *
from coco_utils import *
from networks import *
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

gpus = [0,1]
os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.backends.cudnn.enabled = True
print(torch.cuda.device_count())

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='deeppose demo')
    parser.add_argument('--model', type=str, default="resnet")
    args = parser.parse_args()
    modeltype = args.model

    # user defined parameters
    filename = "final-aug.t7"
    num_threads = 10

    PATH_PREFIX = "/home/yuliang/code/DeepPose-pytorch/results/{}".format(modeltype)
    full_name="/home/yuliang/code/DeepPose-pytorch/models/{}/{}".format(modeltype, filename)
    # full_name = "/home/yuliang/code/DeepPose-pytorch/models/demo/mobilenetv2_224x224-robust.t7"
    full_name = "/home/yuliang/code/DeepPose-pytorch/models/demo/resnet18_227x227.t7"
    ROOT_DIR = "/home/yuliang/code/deeppose_tf/datasets/mpii"
    
    if modeltype == 'resnet':
        input_size = 227
    elif modeltype == 'mobilenet':
        input_size = 224

    print("Loading testing dataset, wait...")

    # load dataset
    test_dataset = PoseDataset(csv_file=os.path.join(ROOT_DIR,'test_joints.csv'),
                                transform=transforms.Compose([
                                            Rescale((input_size, input_size)), # for resnet18
                                            # Wrap((input_size,input_size)), # for mobilenet
                                            Expansion(),
                                            ToTensor()
                                        ]))
    test_dataset_size = len(test_dataset)

    test_dataloader = DataLoader(test_dataset, batch_size=test_dataset_size,
                            shuffle=False, num_workers = num_threads)

    # get all test data
    all_test_data = {}
    for i_batch, sample_batched in enumerate(tqdm(test_dataloader)):
        all_test_data = sample_batched
        
    def eval_coco(net_path, result_gt_json_path, result_pred_json_path):
        """
        Example:
        eval_coco('/home/yuliang/code/PoseFlow/checkpoint140.t7', 
        'result-gt-json.txt', 'result-pred-json.txt')
        """
        # gpu mode
        net = Net().cuda(device_id=gpus[0])
        net = torch.load(net_path).cuda(device_id=gpus[0])

        # cpu mode
        # net = Net()
        # net = torch.load(net_path, map_location=lambda storage, loc: storage)

        ## generate groundtruth json
        total_size = len(all_test_data['image'])
        all_coco_images_arr = [] 
        all_coco_annotations_arr = []
        transform_to_coco_gt(all_test_data['pose'], all_coco_images_arr, all_coco_annotations_arr)
        coco = CocoData(all_coco_images_arr, all_coco_annotations_arr)
        coco_str =  coco.dumps()
        result_gt_json = float2int(coco_str)

        # save ground truth json to file
        f = open(result_gt_json_path, "w")
        f.write(result_gt_json)
        f.close()

        # generate predictioin json
        total_size = len(all_test_data['image'])
        all_coco_pred_annotations_arr = [] 
        for i in tqdm(range(1, int(ceil(total_size / 100.0 + 1)))):
            sample_data = {}

            # gpu mode
            sample_data['image'] = all_test_data['image'][100 * (i - 1) : min(100 * i, total_size)].cuda(device=gpus[0])
            # cpu mode
            # sample_data['image'] = all_test_data['image'][100 * (i - 1) : min(100 * i, total_size)]

            # print('test dataset contains: %d'%(len(sample_data['image'])))
            # t0 = time.time()
            output = net(Variable(sample_data['image'],volatile=True))
            # print('FPS is %f'%(1.0/((time.time()-t0)/len(sample_data['image']))))

            transform_to_coco_pred(output, all_coco_pred_annotations_arr, 100 * (i - 1))

        all_coco_pred_annotations_arr = [item._asdict() for item in all_coco_pred_annotations_arr]
        result_pred_json = json.dumps(all_coco_pred_annotations_arr, cls=MyEncoder)
        result_pred_json = float2int(result_pred_json)

        # save result predict json to file
        f = open(result_pred_json_path, "w")
        f.write(result_pred_json)
        f.close()



    eval_coco(full_name, os.path.join(PATH_PREFIX, 'result-gt-json.txt'), os.path.join(PATH_PREFIX, 'result-pred-json.txt'))

    # evaluation
    annType = ['segm','bbox','keypoints']
    annType = annType[2]
    prefix = 'person_keypoints' if annType=='keypoints' else 'instances'

    print('Running demo for *%s* results.'%(annType))

    annFile = os.path.join(PATH_PREFIX, "result-gt-json.txt")
    cocoGt=COCO(annFile)
    resFile = os.path.join(PATH_PREFIX,"result-pred-json.txt")
    cocoDt=cocoGt.loadRes(resFile)
    imgIds=sorted(cocoGt.getImgIds())

    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
