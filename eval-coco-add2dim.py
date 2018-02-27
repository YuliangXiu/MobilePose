
# coding: utf-8

# In[1]:


from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils, models
from tqdm import tqdm_notebook
from skimage import io, transform
import numpy as np
import torch
import csv
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"
torch.cuda.set_device(0) 
torch.backends.cudnn.enabled = True
print(torch.cuda.device_count())
gpus = [0,1]

ROOT_DIR = "/home/yuliang/code/deeppose_tf/datasets/mpii"

def expand_bbox(left, right, top, bottom, img_width, img_height):
    width = right-left
    height = bottom-top
    ratio = 0.15
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
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image_, (new_h, new_w))
        pose = (pose_.reshape([-1,2])/np.array([w,h])*np.array([new_w,new_h])).flatten()
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


# In[6]:


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
for i_batch, sample_batched in enumerate(tqdm_notebook(test_dataloader)):
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
    for i in range(1, ceil(total_size / 100.0) + 1):
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


# In[7]:


#eval_coco(net_path, result_gt_json_path, result_pred_json_path)
PATH_PREFIX = "./cocoapi/PythonAPI/txts/add2dim"
if not os.path.exists(PATH_PREFIX):
    os.makedirs(PATH_PREFIX)
mdir="/disk3/yinghong/data/mobile-model"
for i in range(0,300, 10):
    filename = "add2dim-checkpoint{}.t7".format(i)
    full_name = os.path.join(mdir, filename)
    eval_coco(full_name,     os.path.join(PATH_PREFIX, 'result-gt-add2dim-{}-json.txt'.format(i)), 
    os.path.join(PATH_PREFIX, 'result-pred-add2dim-{}-json.txt'.format(i)))

