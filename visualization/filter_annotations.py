#
# @author:charlotte.Song
# @file: filter_annotations.py
# @Date: 2019/3/4 16:52
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
import cv2


""" add height and width imformation for image annotations """

data_root = '/home/data3/IC19/'
LSVT_gt_root = data_root + 'LSVT/annotations/'
LSVT_sp_train_data_root = data_root + 'LSVT/sp_train_full_images/'
LSVT_sp_val_data_root = data_root + 'LSVT/sp_val_full_images'

ArT_gt_root = data_root + 'ArT/annotations/'
ArT_sp_train_data_root = data_root + 'ArT/sp_train_art_images/'
ArT_sp_val_data_root = data_root + 'ArT/sp_val_art_images/'


def rewrite_annotations(image_path, ann_file, new_ann_file):
    """ keep the order! """
    assert osp.isdir(image_path)
    assert osp.isfile(ann_file)
    with open(ann_file, 'r', encoding='utf-8') as f:
        gt_annotations = json.loads(f.read(), object_pairs_hook=OrderedDict)
    detailed_annotation = {}
    for name, value in gt_annotations.items():
        file = osp.join(image_path, name + '.jpg')
        assert osp.isfile(file)
        img = cv2.imread(file)
        height, width = img.shape[0:2]
        detailed_annotation[name] = {
            'height': height,
            'width': width,
        }
    with open(new_ann_file, 'w+', encoding='utf-8') as f:
        json.dump(OrderedDict(detailed_annotation), f)
    print('write down {}'.format(new_ann_file))
    # with open(ann_file, 'w+', encoding='utf-8') as f:
    #     json.dump(OrderedDict(gt_annotations), f)
    # print('write down {}'.format(ann_file))


def add_eval_json(ann_file, new_ann_file):
    """ """
    assert osp.isfile(ann_file)
    with open(ann_file, 'r', encoding='utf-8') as f:
        gt_annotations = json.loads(f.read(), object_pairs_hook=OrderedDict)
    new_val_annotation = {}
    for name, value in gt_annotations.items():
        val_name = name.replace('gt_', 'res_')
        new_val_annotation[val_name] = gt_annotations[name]
    with open(new_ann_file, 'w+', encoding='utf-8') as f:
        json.dump(OrderedDict(new_val_annotation), f)
    print('write down {}'.format(new_ann_file))


if __name__ == '__main__':
    train_ann = osp.join(LSVT_gt_root, 'sp_train_full_labels.json')
    val_ann = osp.join(LSVT_gt_root, 'sp_val_full_labels.json')
    new_train_ann = osp.join(LSVT_gt_root, 'sp_train_detail_labels.json')
    new_val_ann = osp.join(LSVT_gt_root, 'sp_val_detail_labels.json')
    rewrite_annotations(LSVT_sp_train_data_root, train_ann, new_train_ann)
    rewrite_annotations(LSVT_sp_val_data_root, val_ann, new_val_ann)
    val_ann = osp.join(LSVT_gt_root, 'sp_val_full_labels.json')
    new_val_ann = osp.join(LSVT_gt_root, 'sp_val_eval_labels.json')
    add_eval_json(val_ann, new_val_ann)
    # train_ann = osp.join(ArT_gt_root, 'sp_train_art_labels.json')
    # val_ann = osp.join(ArT_gt_root, 'sp_val_art_labels.json')
    # new_train_ann = osp.join(ArT_gt_root, 'sp_train_detail_labels.json')
    # new_val_ann = osp.join(ArT_gt_root, 'sp_val_detail_labels.json')
    # eval_val_ann = osp.join(ArT_gt_root, 'sp_val_eval_labels.json')
    # rewrite_annotations(ArT_sp_train_data_root, train_ann, new_train_ann)
    # rewrite_annotations(ArT_sp_val_data_root, val_ann, new_val_ann)
    # add_eval_json(val_ann, eval_val_ann)

