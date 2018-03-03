# coding: utf-8

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils, models
from tqdm import tqdm
from skimage import io, transform
import numpy as np
import torch
import csv
import os
from dataloader import *
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"]="1"
# torch.cuda.set_device(0) 
torch.backends.cudnn.enabled = True
print(torch.cuda.device_count())
gpus = [0,1]

ROOT_DIR = "/home/yuliang/code/deeppose_tf/datasets/mpii"

###############################COCO CLASS###############################
# define coco class
import json
import numpy
from collections import namedtuple, Mapping

# Create namedtuple without defaults
def namedtuple_with_defaults(typename, field_names, default_values=()):
    T = namedtuple(typename, field_names)
    T.__new__.__defaults__ = (None,) * len(T._fields)
    if isinstance(default_values, Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    return T

# Used for solving TypeError: Object of type 'float32' is not JSON serializable
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

# Classes for coco groud truth, CocoImage and CocoAnnotation
CocoImage = namedtuple_with_defaults('image', ['file_name', 'height', 'width', 'id'])
CocoAnnotation = namedtuple_with_defaults('annotation', ['num_keypoints', 'area', 
                                             'iscrowd', 'keypoints', 
                                             'image_id', 'bbox', 'category_id',
                                            'id'])
class CocoData:
    def __init__(self, coco_images_arr, coco_annotations_arr):
        self.Coco = {}
        coco_images_arr = [item._asdict() for item in coco_images_arr]
        coco_annotations_arr = [item._asdict() for item in coco_annotations_arr]
        self.Coco['images'] = coco_images_arr
        self.Coco['annotations'] = coco_annotations_arr
        self.Coco['categories'] = [{"id": 1, "name": "test"}]
        
    def dumps(self):
        return json.dumps(self.Coco, cls=MyEncoder) 

# Change keypoints [x, y, prob] prob = int(prob)
def float2int(str_data):
    json_data = json.loads(str_data)
    annotations = []
    if 'annotations' in json_data:
        annotations = json_data['annotations']
    else:
        annotations = json_data
    json_size = len(annotations)
    for i in range(json_size):
        annotation = annotations[i]
        keypoints = annotation['keypoints']
        keypoints_num = int(len(keypoints) / 3)
        for j in range(keypoints_num):
            keypoints[j * 3 + 2] = int(round(keypoints[j * 3 + 2]))
    return json.dumps(json_data)

# Append coco ground truth to coco_images_arr and coco_annotations_arr
def transform_to_coco_gt(datas, coco_images_arr, coco_annotations_arr):
    """
    data: num_samples * 32, type Tensor
    16 keypoints
    
    output:
    inside coco_images_arr, coco_annotations_arr
    """
    for idx, sample in enumerate(datas):
        coco_image = CocoImage()
        coco_annotation = CocoAnnotation()
        sample = np.array(sample.numpy()).reshape(-1, 2)
        num_keypoints = len(sample)    
        keypoints = np.append(sample, np.array(np.ones(num_keypoints).reshape(-1, 1) * 2), 
                      axis=1)
        xmin = np.min(sample[:,0])
        ymin = np.min(sample[:,1])
        xmax = np.max(sample[:,0])
        ymax = np.max(sample[:,1])
        width = ymax - ymin
        height = xmax - xmin
        coco_image = coco_image._replace(id = idx, width=width, height=height, file_name="")
        coco_annotation = coco_annotation._replace(num_keypoints=num_keypoints)
        coco_annotation = coco_annotation._replace(area=width*height)
        coco_annotation = coco_annotation._replace(keypoints=keypoints.reshape(-1))
        coco_annotation = coco_annotation._replace(image_id=idx)
        coco_annotation = coco_annotation._replace(bbox=[xmin, ymin, width, height])
        coco_annotation = coco_annotation._replace(category_id=1) # default "1" for keypoint
        coco_annotation = coco_annotation._replace(id=idx)
        coco_annotation = coco_annotation._replace(iscrowd=0)
        coco_images_arr.append(coco_image)
        coco_annotations_arr.append(coco_annotation)
    return ()

# Coco predict result class
CocoPredictAnnotation = namedtuple_with_defaults('predict_anno', ['image_id', 'category_id', 'keypoints', 'score'])

# Append coco predict result to coco_images_arr and coco_pred_annotations_arr
def transform_to_coco_pred(datas, coco_pred_annotations_arr, beg_idx):
    """
    data: num_samples * 32, type Variable
    16 keypoints
    
    output:
    inside coco_pred_annotations_arr
    """
    for idx, sample in enumerate(datas):
        coco_pred_annotation = CocoPredictAnnotation()
        
        sample = np.array(sample.data.cpu().numpy()).reshape(-1, 2)
        num_keypoints = len(sample)        
        keypoints = np.append(sample, np.array(np.ones(num_keypoints).reshape(-1, 1) * 2), 
                      axis=1)
        xmin = np.min(sample[:,0])
        ymin = np.min(sample[:,1])
        xmax = np.max(sample[:,0])
        ymax = np.max(sample[:,1])
        width = ymax - ymin
        height = xmax - xmin
        # set value
        cur_idx = beg_idx + idx
        coco_pred_annotation = coco_pred_annotation._replace(image_id=cur_idx)
        coco_pred_annotation = coco_pred_annotation._replace(category_id=1)
        coco_pred_annotation = coco_pred_annotation._replace(keypoints=keypoints.reshape(-1))
        coco_pred_annotation = coco_pred_annotation._replace(score=2)
        # add to arr
        coco_pred_annotations_arr.append(coco_pred_annotation)
    return ()

###############################NET###############################
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

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


from math import ceil

print("Loading testing dataset, wait...")
# load dataset
test_dataset = PoseDataset(csv_file=os.path.join(ROOT_DIR,'test_joints.csv'),
                              transform=transforms.Compose([
                                           Rescale((227,227)),
                                           Expansion(),
                                           ToTensor()
                                       ]))
test_dataset_size = len(test_dataset)
test_dataloader = DataLoader(test_dataset, batch_size=test_dataset_size,
                        shuffle=False, num_workers = 10)
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
    criterion = nn.MSELoss().cuda(device_id=gpus[0])
    optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)
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
    for i in range(1, int(ceil(total_size / 100.0) + 1)):
        sample_data = {}
        print(100 * (i - 1), min(100 * i, total_size))
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


# eval_coco(net_path, result_gt_json_path, result_pred_json_path)
PATH_PREFIX = "./cocoapi/PythonAPI/txts"
if not os.path.exists(PATH_PREFIX):
    os.makedirs(PATH_PREFIX)
# model dir
mdir="/home/yuliang/code/DeepPose-pytorch/models/yh"
# model name
name = "yh"
# final epoch
epoch = 20

for i in range(0,epoch,10):
    filename = "checkpoint{}.t7".format(i)
    full_name = os.path.join(mdir, filename)
    eval_coco(full_name, os.path.join(PATH_PREFIX, 'result-gt-{}-{}-json.txt'.format(name,i)),\
    os.path.join(PATH_PREFIX, 'result-pred-{}-{}-json.txt'.format(name,i)))

# evaluation
os.chdir("/home/yuliang/code/DeepPose-pytorch/cocoapi/PythonAPI")
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import json
import os

annType = ['segm','bbox','keypoints']
annType = annType[2]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
print 'Running demo for *%s* results.'%(annType)

PATH_PREFIX = "./txts"

for i in range(0,epoch,10):

    annFile = os.path.join(PATH_PREFIX, "result-gt-{}-{}-json.txt".format(name,i))
    cocoGt=COCO(annFile)

    resFile = os.path.join(PATH_PREFIX, "result-pred-{}-{}-json.txt".format(name,i))

    cocoDt=cocoGt.loadRes(resFile)
    imgIds=sorted(cocoGt.getImgIds())

    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

