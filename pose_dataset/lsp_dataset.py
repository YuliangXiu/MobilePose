# Copyright (c) 2016 Artsiom Sanakoyeu

from __future__ import division
from __future__ import print_function

from os.path import basename
from scipy.io import loadmat
import argparse
import glob
import re
import os.path

from scripts.config import *


def create_data(images_dir, joints_mat_path, transpose_order=(2, 0, 1)):
    """
    Create a list of lines in format:
      image_path, x1, y1, x2,y2, ...
      where xi, yi - coordinates of the i-th joint
    """
    joints = loadmat(joints_mat_path)
    print(joints['joints'].shape)
    joints = joints['joints'].transpose(*transpose_order)
    print(joints.shape)
    joints = joints[:, :, :2]
    print(joints.shape)
    if joints.shape[1:] != (14, 2):
        raise ValueError('Incorrect shape of the joints matrix of joints.mat. '
                         'Expected: (?, 14, 2); received: (?, {}, {})'.format(
                            joints.shape[1], joints.shape[2]))

    lines = list()
    for img_path in sorted(glob.glob(os.path.join(images_dir, '*.jpg'))):
        index = int(re.search(r'im([0-9]+)', basename(img_path)).groups()[0]) - 1
        joints_str_list = [str(j) if j > 0 else '-1' for j in joints[index].flatten().tolist()]

        out_list = [img_path]
        out_list.extend(joints_str_list)
        out_str = ','.join(out_list)

        lines.append(out_str)
    return lines


if __name__ == '__main__':
    """
    Write train.csv and test.csv.
    Each line in csv file will be in the following format:
      image_name, x1, y1, x2,y2, ...
      where xi, yi - coordinates of the i-th joint
    Train file consists of 11000 lines (all images from extended LSP + first 1000 images from small LSP).
    Test file consists of 1000 lines (last 1000 images from small LSP).
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--extended_lsp_images_dir', type=str, default=os.path.join(LSP_EXT_DATASET_ROOT, 'images'))
    parser.add_argument('--extended_lsp_joints_path', type=str, default=os.path.join(LSP_EXT_DATASET_ROOT, 'joints.mat'))
    parser.add_argument('--small_lsp_images_dir', type=str, default=os.path.join(LSP_DATASET_ROOT, 'images'))
    parser.add_argument('--small_lsp_joints_path', type=str, default=os.path.join(LSP_DATASET_ROOT, 'joints.mat'))
    parser.add_argument('--output_dir', type=str, default=LSP_EXT_DATASET_ROOT)
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    file_train = open('%s/train_joints.csv' % args.output_dir, 'w')
    file_test = open('%s/test_joints.csv' % args.output_dir, 'w')
    file_train_lsp_small = open('%s/train_lsp_small_joints.csv' % args.output_dir, 'w')

    print('Read LSP_EXT')
    lsp_ext_lines = create_data(args.extended_lsp_images_dir, args.extended_lsp_joints_path,
                                transpose_order=(2, 0, 1))
    print('Read LSP')
    lsp_small_lines = create_data(args.small_lsp_images_dir, args.small_lsp_joints_path,
                                  transpose_order=(2, 1, 0))  # different dim order
    print('Extended LSP images:', len(lsp_ext_lines))
    print('Small LSP images:', len(lsp_small_lines))
    if len(lsp_ext_lines) != 10000:
        raise Exception('Extended LSP dataset must contain 10000 images!')
    if len(lsp_small_lines) != 2000:
        raise Exception('Small LSP dataset must contain 2000 images!')
    num_small_lsp_train = 1000

    for line in lsp_ext_lines:
        print(line, file=file_train)
    for line in lsp_small_lines[:num_small_lsp_train]:
        print(line, file=file_train)
        print(line, file=file_train_lsp_small)
    for line in lsp_small_lines[num_small_lsp_train:]:
        print(line, file=file_test)

    file_train.close()
    file_test.close()
    file_train_lsp_small.close()
