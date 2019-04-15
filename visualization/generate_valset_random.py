#
# @author:charlotte.Song
# @file: generate_val_set.py
# @Date: 2019/3/3 22:27
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import argparse
import torch
import os
import os.path as osp
import json
from collections import OrderedDict
import shutil

""" generate validation set for full images """
val_set = 500
data_root = '/home/data3/IC19/'
LSVT_full_image_data_root = data_root + 'LSVT/train_full_images/'
LSVT_full_gt_data_root = data_root + 'LSVT/annotations/'
vis_path = './results/'
LSVT_trainset_path = data_root + 'LSVT/sp_train_full_images/'
LSVT_valset_path = data_root + 'LSVT/sp_val_full_images/'

ArT_image_data_root = data_root + 'ArT/train_images_art/'
ArT_gt_data_root = data_root + 'ArT/annotations/'
ArT_trainset_path = data_root + 'ArT/sp_train_images/'
ArT_valset_path = data_root + 'ArT/sp_val_images/'


def random_split_validation(image_path, ann_file, train_ann_file, val_ann_file,
                            trainset_path, valset_path, random_seed=123456):
    assert osp.isdir(image_path), 'wrong path'
    assert osp.isfile(ann_file), 'wrong file'
    img_files = os.listdir(image_path)
    train_annotations = {}
    val_annotations = {}
    with open(ann_file, 'r', encoding='utf-8') as f:
        gt_annotations = json.loads(f.read(), object_pairs_hook=OrderedDict)

    val_num, train_num = int(val_set), len(img_files) - int(val_set)
    np.random.seed(random_seed)
    val_files = np.random.choice(img_files, val_num, replace=False)

    for file in img_files:
        if file in val_files:
            shutil.copy(osp.join(image_path, file), osp.join(valset_path, file))
            name = osp.splitext(file)[0]
            val_annotations[name] = gt_annotations[name]
        else:
            shutil.copy(osp.join(image_path, file), osp.join(trainset_path, file))
            name = osp.splitext(file)[0]
            train_annotations[name] = gt_annotations[name]

    with open(train_ann_file, 'w+', encoding='utf-8') as f:
        json.dump(train_annotations, f)
    with open(val_ann_file, 'w+', encoding='utf-8') as f:
        json.dump(val_annotations, f)


if __name__ == '__main__':
    gt_path = LSVT_full_gt_data_root + 'train_full_labels.json'
    if not osp.exists(LSVT_trainset_path):
        os.makedirs(LSVT_trainset_path)
    if not osp.exists(LSVT_valset_path):
        os.makedirs(LSVT_valset_path)

    LSVT_train_file = osp.join(LSVT_full_gt_data_root, 'sp_train_full_labels.json')
    LSVT_val_file = osp.join(LSVT_full_gt_data_root, 'sp_val_full_labels.json')
    random_split_validation(LSVT_full_image_data_root, gt_path,
                            LSVT_train_file, LSVT_val_file,
                            LSVT_trainset_path, LSVT_valset_path)














