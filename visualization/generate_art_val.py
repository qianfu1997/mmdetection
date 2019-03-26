#
# @author:charlotte.Song
# @file: generate_art_val.py
# @Date: 2019/3/5 20:16
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

""" generate validation set from art_val.odgt """
data_root = '../data/'
ArT_image_data_root = data_root + 'ArT/train_images_art/'
ArT_gt_data_root = data_root + 'ArT/annotations/'
ArT_trainset_path = data_root + 'ArT/sp_train_art_images/'
ArT_valset_path = data_root + 'ArT/sp_val_art_images/'


def split_validation(image_path, ann_file, val_file,
                     train_ann_file, val_ann_file, trainset_pth, valset_pth):
    assert osp.isdir(image_path)
    assert osp.isfile(ann_file)
    assert osp.isfile(val_file)
    val_names = []
    with open(ann_file, 'r', encoding='utf-8') as f:
        gt_annotations = json.loads(f.read(), object_pairs_hook=OrderedDict)
    with open(val_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            annotation = eval(line)
            name = osp.splitext(annotation["fpath"])[0]
            val_names.append(name)

    train_annotations = {}
    val_annotations = {}
    for name in gt_annotations.keys():
        file = name + '.jpg'
        if name in val_names:
            val_annotations[name] = gt_annotations[name]
            shutil.copy(osp.join(image_path, file), osp.join(ArT_valset_path, file))
        else:
            train_annotations[name] = gt_annotations[name]
            shutil.copy(osp.join(image_path, file), osp.join(ArT_trainset_path, file))

    with open(train_ann_file, 'w+', encoding='utf-8') as f:
        json.dump(train_annotations, f)
    print('write down {}'.format(train_ann_file))
    with open(val_ann_file, 'w+', encoding='utf-8') as f:
        json.dump(val_annotations, f)
    print('write down {}'.format(val_ann_file))


if __name__ == '__main__':
    gt_path = ArT_gt_data_root + 'train_labels.json'
    if not osp.isdir(ArT_trainset_path):
        os.mkdir(ArT_trainset_path)
    if not osp.isdir(ArT_valset_path):
        os.mkdir(ArT_valset_path)

    odgt_val_file = osp.join(ArT_gt_data_root, 'IC19ArT_val.odgt')
    ArT_train_file = osp.join(ArT_gt_data_root, 'sp_train_art_labels.json')
    ArT_val_file = osp.join(ArT_gt_data_root, 'sp_val_art_labels.json')
    split_validation(ArT_image_data_root, gt_path, odgt_val_file,
                     ArT_train_file, ArT_val_file, ArT_trainset_path, ArT_valset_path)

