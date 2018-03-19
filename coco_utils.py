'''
File: coco_utils.py
Project: MobilePose
File Created: Saturday, 3rd March 2018 7:04:57 pm
Author: Yuliang Xiu (yuliangxiu@sjtu.edu.cn)
-----
Last Modified: Thursday, 8th March 2018 3:02:15 pm
Modified By: Yuliang Xiu (yuliangxiu@sjtu.edu.cn>)
-----
Copyright 2018 - 2018 Shanghai Jiao Tong University, Machine Vision and Intelligence Group
'''


# define coco class
import json
import numpy as np
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
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
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