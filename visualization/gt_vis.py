#
# @author:charlotte.Song
# @file: gt_vis.py
# @Date: 2019/3/3 18:39
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import argparse
import torch
import cv2
import json
import os
import os.path as osp
from collections import OrderedDict

"""
    read the gt json, select 100 images from dictionary.
    generate gt bounding boxes and gt masks for them.
    
"""
data_root = '../data/'
LSVT_full_image_data_root = data_root + 'LSVT/train_full_images/'
LSVT_full_gt_data_root = data_root + 'LSVT/annotations/'
ArT_image_data_root = data_root + 'ArT/train_images_art/'
ArT_gt_data_root = data_root + 'ArT/annotations/'
vis_path = './results/art/'


def vis_gt(image_path, gt_path):
    assert osp.isdir(image_path), 'Error: wrong image dictionary.'
    assert osp.isfile(gt_path), 'Error: wrong gt file.'
    # get image names
    # img_names = []
    # for file in os.listdir(image_path):
    #     img_names.append(file)
    img_names = os.listdir(image_path)
        
    with open(gt_path, 'r', encoding='utf-8_sig') as f:
        gt_annotations = json.loads(f.read(), object_pairs_hook=OrderedDict)
    # select first 100 images
    img_names = img_names[:100]
    for name in img_names:
        img = cv2.imread(osp.join(image_path, name))
        name = osp.splitext(name)[0]
        mask = np.zeros(img.shape, dtype=np.uint8)
        annotations = gt_annotations[name]
        for annotation in annotations:
            if annotation["transcription"] == "###":
                continue
            # extract rectangle for points and then draw it.
            points = np.array(annotation["points"])
            # rect = cv2.minAreaRect(points)
            # box = np.int(cv2.BoxPoints(rect))
            # x, y, w, h = cv2.boundingRect(points)
            box = np.zeros(4)
            box[:2] = np.min(points, axis=0)
            box[2:] = np.max(points, axis=0)
            cv2.drawContours(img, [points], -1, (0, 0, 255), 2)
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.rectangle(mask, (x, y), (x + w, y + h), (0, 0, 255), -1)
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.fillPoly(mask, [points], (255, 255, 255))

        cv2.imwrite(vis_path + name + '.jpg', img)
        cv2.imwrite(vis_path + name + '_mask.jpg', mask)
        print(name)


if __name__ == '__main__':
    if not osp.isdir(vis_path):
        os.mkdir(vis_path)
    gt_path = LSVT_full_gt_data_root + 'train_full_labels.json'
    art_gt_path = ArT_gt_data_root + 'train_labels.json'
    # vis_gt(LSVT_full_image_data_root, gt_path)
    vis_gt(ArT_image_data_root, art_gt_path)








    

