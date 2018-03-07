# coding: utf-8

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
from dataloader import *
from coco_utils import *
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

os.environ["CUDA_VISIBLE_DEVICES"]="1"
# torch.cuda.set_device(0) 
torch.backends.cudnn.enabled = True
print(torch.cuda.device_count())
gpus = [0,1]

ROOT_DIR = "/home/yuliang/code/deeppose_tf/datasets/mpii"

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        model = models.resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3,bias=False)
        model.fc=nn.Linear(512,32)

        for param in model.parameters():
            param.requires_grad = True
        self.resnet = model.cuda()
        
    def forward(self, x):
       
        pose_out = self.resnet(x)
        return pose_out

print("Loading testing dataset, wait...")

# load dataset
test_dataset = PoseDataset(csv_file=os.path.join(ROOT_DIR,'test_joints.csv'),
                              transform=transforms.Compose([
                                           Rescale((224,224)),
                                           Expansion(),
                                           ToTensor()
                                       ]))
test_dataset_size = len(test_dataset)
# test_dataset_size = 5
test_dataloader = DataLoader(test_dataset, batch_size=test_dataset_size,
                        shuffle=False, num_workers = 20)
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
    # load net
    net = Net().cuda(device_id=gpus[0])
    net = torch.load(net_path).cuda(device_id=gpus[0])
    ##### generate result ground truth json #####
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
    ##### generate result ground truth json #####
    total_size = len(all_test_data['image'])
    all_coco_pred_annotations_arr = [] 
    for i in tqdm(range(1, int(ceil(total_size / 100.0 + 1)))):
        sample_data = {}
        # print(100 * (i - 1), min(100 * i, total_size))
        sample_data['image'] = all_test_data['image'][100 * (i - 1) : min(100 * i, total_size)].cuda(device=gpus[0])
        output = net(Variable(sample_data['image'],volatile=True))
        transform_to_coco_pred(output, all_coco_pred_annotations_arr, 100 * (i - 1))

    all_coco_pred_annotations_arr = [item._asdict() for item in all_coco_pred_annotations_arr]
    result_pred_json = json.dumps(all_coco_pred_annotations_arr, cls=MyEncoder)
    result_pred_json = float2int(result_pred_json)
    # save result predict json to file
    f = open(result_pred_json_path, "w")
    f.write(result_pred_json)
    f.close()


name = "yh"
PATH_PREFIX = "/home/yuliang/code/DeepPose-pytorch/results/{}".format(name)
mdir="/home/yuliang/code/DeepPose-pytorch/models/{}".format(name)

# epoch = 20
# for i in range(0,epoch,10):
#     filename = "checkpoint{}.t7".format(i)
#     full_name = os.path.join(mdir, filename)
#     eval_coco(full_name, os.path.join(PATH_PREFIX, 'result-gt-{}-json.txt'.format(i)),\
#     os.path.join(PATH_PREFIX, 'result-pred-{}-json.txt'.format(i)))

filename = "final.t7"
full_name = os.path.join(mdir, filename)
eval_coco(full_name, os.path.join(PATH_PREFIX, 'result-gt-json.txt'), os.path.join(PATH_PREFIX, 'result-pred-json.txt'))

# evaluation
annType = ['segm','bbox','keypoints']
annType = annType[2]
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'

print('Running demo for *%s* results.'%(annType))

# for i in range(0,epoch,10):

#     annFile = os.path.join(PATH_PREFIX, "result-gt-{}-json.txt".format(i))
#     cocoGt=COCO(annFile)

#     resFile = os.path.join(PATH_PREFIX, "result-pred-{}-json.txt".format(i))

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

