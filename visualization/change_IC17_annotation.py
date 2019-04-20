#
# @author:charlotte.Song
# @file: change_IC17_annotation.py
# @Date: 2019/4/15 13:21
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import argparse
import os
import os.path as osp
from collections import OrderedDict
import shutil
import cv2
from PIL import Image, ImageSequence
import json

# only deal with train set.
data_root = '/home/data3/IC19/ICDAR2017_MLT/'
annotation_root = data_root + 'annotations/'
train_data_root = data_root + 'train/'
train_gt_root = data_root + 'train_gt/'
new_train_data_root = data_root + 'sp_train/'
standard_ext = ['.jpg', '.png', '.gif']

def parser_args():
    parser = argparse.ArgumentParser(description='deal with IC17/IC19')
    parser.add_argument('trainset_root', type=str)
    parser.add_argument('trainset_gt_root', type=str)
    parser.add_argument('annotation_root', type=str)
    parser.add_argument('new_trainset_root', type=str)
    parser.add_argument('--new_ann_file', type=str)
    parser.add_argument('--new_detail_file', type=str)
    args = parser.parse_args()
    return args


def write_ann_to_json(image_path, ann_path, new_ann_json,
                      new_detail_json, new_image_path):
    assert osp.isdir(image_path)
    assert osp.isdir(ann_path)
    assert osp.isdir(new_image_path)

    files = os.listdir(image_path)
    total_gt_annotations = {}
    total_gt_detail_annotations = {}
    for file in files:
        if osp.splitext(file)[-1] not in standard_ext:
            continue
        name = osp.splitext(file)[0]
        if 'IC19' in image_path:
            ann_file = osp.join(ann_path, name + '.txt')
        else:
            ann_file = osp.join(ann_path, 'gt_' + name + '.txt')
        with open(ann_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        boxes = []
        for line in lines:
            line = line.strip().replace('\xef\xbb\xbf', '')
            line = line.split(',')
            language = line[-2]
            transcription = line[-1]
            illegibility = bool('#' in transcription)
            points = [[int(line[i * 2]), int(line[i * 2 + 1])] for i in range(4)]
            box = {
                "transcription": transcription,
                "points": points,
                "language": language,
                "illegibility": illegibility
            }
            boxes.append(box)
        total_gt_annotations[name] = boxes
        if osp.splitext(file)[-1] == '.gif':
            img = Image.open(osp.join(image_path, file)).seek(0)
            width, height = img.size
            img.save(osp.join(new_image_path, file.replace('.gif', '.png')))
        else:
            img = cv2.imread(osp.join(image_path, file))
            height, width = img.shape[:2]
            shutil.copy(osp.join(image_path, file), osp.join(new_image_path, file))
        total_gt_detail_annotations[name] = {
            'height': height,
            'width': width}
    with open(new_ann_json, 'w+', encoding='utf-8') as f:
        json.dump(OrderedDict(total_gt_annotations), f)
    print('write down {}'.format(new_ann_json))
    with open(new_detail_json, 'w+', encoding='utf-8') as f:
        json.dump(OrderedDict(total_gt_detail_annotations), f)
    print('write down {}'.format(new_detail_json))


if __name__ == '__main__':
    args = parser_args()
    annotation_root = args.annotation_root
    new_train_data_root = args.new_trainset_root
    if not osp.isdir(annotation_root):
        os.makedirs(annotation_root)
    if not osp.isdir(new_train_data_root):
        os.makedirs(new_train_data_root)
    # new_train_json = osp.join(annotation_root, 'sp_train_IC17_labels.json')
    # new_detail_json = osp.join(annotation_root, 'sp_train_detail_labels.json')
    new_train_json = osp.join(annotation_root, args.new_ann_file)
    new_detail_json = osp.join(annotation_root, args.new_detail_file)
    # write_ann_to_json(image_path=train_data_root, ann_path=train_gt_root,
    #                   new_ann_json=new_train_json, new_detail_json=new_detail_json,
    #                   new_image_path=new_train_data_root)
    write_ann_to_json(image_path=args.trainset_root, ann_path=args.trainset_gt_root,
                      new_ann_json=new_train_json, new_detail_json=new_detail_json,
                      new_image_path=new_train_data_root)











