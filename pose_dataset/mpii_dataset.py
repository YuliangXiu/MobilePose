#!/usr/bin/env python
# Copyright (c) 2016 Shunta Saito (original code)
# Copyright (c) 2016 Artsiom Sanakoyeu

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from scipy.io import loadmat
from itertools import izip
import json
import numpy as np

from scripts.config import *

MPII_DATA_DIR = MPII_DATASET_ROOT
MPII_OUT_DIR = MPII_DATASET_ROOT


def fix_wrong_joints(joint):
    if '12' in joint and '13' in joint and '2' in joint and '3' in joint:
        if ((joint['12'][0] < joint['13'][0]) and
                (joint['3'][0] < joint['2'][0])):
            joint['2'], joint['3'] = joint['3'], joint['2']
        if ((joint['12'][0] > joint['13'][0]) and
                (joint['3'][0] > joint['2'][0])):
            joint['2'], joint['3'] = joint['3'], joint['2']

    return joint


def save_joints():
    """
    Convert annotations mat file to json and save on disk.
    Only persons with annotations of all 16 joints will be written in the json.
    """
    joint_data_fn = os.path.join(MPII_OUT_DIR, 'data.json')
    mat = loadmat(os.path.join(MPII_DATA_DIR, 'mpii_human_pose_v1_u12_1.mat'))

    fp = open(joint_data_fn, 'w')

    for i, (anno, train_flag) in enumerate(
        izip(mat['RELEASE']['annolist'][0, 0][0],
            mat['RELEASE']['img_train'][0, 0][0])):

        img_fn = anno['image']['name'][0, 0][0]
        train_flag = int(train_flag)

        if 'annopoints' in str(anno['annorect'].dtype):
            annopoints = anno['annorect']['annopoints'][0]
            head_x1s = anno['annorect']['x1'][0]
            head_y1s = anno['annorect']['y1'][0]
            head_x2s = anno['annorect']['x2'][0]
            head_y2s = anno['annorect']['y2'][0]
            for annopoint, head_x1, head_y1, head_x2, head_y2 in \
                    izip(annopoints, head_x1s, head_y1s, head_x2s, head_y2s):
                if len(annopoint) > 0:
                    head_rect = [float(head_x1[0, 0]),
                                 float(head_y1[0, 0]),
                                 float(head_x2[0, 0]),
                                 float(head_y2[0, 0])]

                    # joint coordinates
                    annopoint = annopoint['point'][0, 0]
                    j_id = [str(j_i[0, 0]) for j_i in annopoint['id'][0]]
                    x = [x[0, 0] for x in annopoint['x'][0]]
                    y = [y[0, 0] for y in annopoint['y'][0]]
                    joint_pos = {}
                    for _j_id, (_x, _y) in zip(j_id, zip(x, y)):
                        joint_pos[str(_j_id)] = [float(_x), float(_y)]
                    # joint_pos = fix_wrong_joints(joint_pos)

                    # visiblity list
                    if 'is_visible' in str(annopoint.dtype):
                        vis = [v[0] if v else [0]
                               for v in annopoint['is_visible'][0]]
                        vis = dict([(k, int(v[0])) if len(v) > 0 else v
                                    for k, v in zip(j_id, vis)])
                    else:
                        vis = None

                    if len(joint_pos) == 16:
                        data = {
                            'filename': img_fn,
                            'train': train_flag,
                            'head_rect': head_rect,
                            'is_visible': vis,
                            'joint_pos': joint_pos
                        }

                        print(json.dumps(data), file=fp)


def write_line(datum, fp):
    """
    Write a line in format:
      image_name, x1, y1, x2,y2, ...
      where xi, yi - coordinates of the i-th joint
    """
    joints = sorted([[int(k), v] for k, v in datum['joint_pos'].items()])
    joints = np.array([j for i, j in joints]).flatten()

    out = [datum['filename']]
    out.extend(joints)
    out = [str(o) for o in out]
    out = ','.join(out)

    print(out, file=fp)


def split_train_test():
    fp_test = open(os.path.join(MPII_OUT_DIR, 'test_joints.csv'), 'w')
    fp_train = open(os.path.join(MPII_OUT_DIR, 'train_joints.csv'), 'w')
    all_data = open(os.path.join(MPII_OUT_DIR, 'data.json')).readlines()
    N = len(all_data)
    N_test = int(N * 0.1)
    N_train = N - N_test

    print('N:{}'.format(N))
    print('N_train:{}'.format(N_train))
    print('N_test:{}'.format(N_test))

    np.random.seed(1701)
    perm = np.random.permutation(N)
    test_indices = perm[:N_test]
    train_indices = perm[N_test:]

    print('train_indices:{}'.format(len(train_indices)))
    print('test_indices:{}'.format(len(test_indices)))

    for i in train_indices:
        datum = json.loads(all_data[i].strip())
        write_line(datum, fp_train)

    for i in test_indices:
        datum = json.loads(all_data[i].strip())
        write_line(datum, fp_test)


if __name__ == '__main__':
    save_joints()
    split_train_test()
