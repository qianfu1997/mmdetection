#
# @author:charlotte.Song
# @file: filter_art_annotations.py
# @Date: 2019/3/13 17:09
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import argparse
import torch
from PIL import Image
import os
import os.path as osp
import json
from collections import OrderedDict
import cv2
""" to rotate the wrong image in ArT """

debug_root = './debug/'
data_root = '../data/'
ArT_annotation_path = data_root + 'ArT/annotations/'
ArT_data_path = data_root + 'ArT/train_images_art/'
ArT_sp_train_data_root = data_root + 'ArT/sp_train_art_images/'
ArT_sp_val_data_root = data_root + 'ArT/sp_val_art_images/'
ArT_trainset_path = data_root + 'ArT/sp_train_art_images/'
ArT_valset_path = data_root + 'ArT/sp_val_art_images/'


def rewrite_art_annotations(image_path, ann_file, new_detail_ann,
                            img_save_path):
    """
    read images by PIL and then store them into art_train.
    :param image_path: train_images_art/
    :param ann_file: annotations/sp_train_art_labels.json/
    :param new_detail_ann:
    :return:
    """
    assert osp.isdir(image_path)
    assert osp.isfile(ann_file)
    assert osp.isdir(img_save_path)
    with open(ann_file, 'r', encoding='utf-8') as f:
        gt_annotations = json.loads(f.read(), object_pairs_hook=OrderedDict)
    detailed_annotation = {}
    # use PIL to read the image and then store it.
    # check if any bboxes overlap the boundaries.
    for name, value in gt_annotations.items():
        file = osp.join(image_path, name + '.jpg')
        assert osp.isfile(file)
        img = Image.open(file)
        width, height = img.size[0], img.size[1]
        # img = cv2.imread(file)
        img = np.asarray(img)
        # get all boxes.
        for bbox_ann in value:
            bbox_points = np.array(bbox_ann["points"]).reshape(-1, 2)
            if not (np.min(bbox_points[:, 0]) >= 0 and np.max(bbox_points[:, 0]) <= width):
                print('Error file {}'.format(file))
                cv2.drawContours(img, [bbox_points], -1, (0, 0, 255), 2)
                cv2.imwrite(osp.join(debug_root, name + '.jpg'), img)
                # img.save(osp.join(debug_root, name + '.jpg'))
            elif not (np.min(bbox_points[:, 1]) >= 0 and np.max(bbox_points[:, 1]) <= height):
                print('Error file {}'.format(file))
                cv2.drawContours(img, [bbox_points], -1, (0, 0, 255), 2)
                # img.save(osp.join(debug_root, name + '.jpg'))
                cv2.imwrite(osp.join(debug_root, name + '.jpg'), img)
    #     detailed_annotation[name] = {
    #         'height': height,
    #         'width': width,
    #     }
    #     # img.save(osp.join(img_save_path, name + '.jpg'))
    # with open(new_detail_ann, 'w+', encoding='utf-8') as f:
    #     json.dump(OrderedDict(detailed_annotation), f)
    print('write down {}'.format(new_detail_ann))


def add_eval_json(ann_file, new_ann_file):
    """ """
    assert osp.isfile(ann_file)
    with open(ann_file, 'r', encoding='utf-8') as f:
        gt_annotations = json.loads(f.read(), object_pairs_hook=OrderedDict)
    new_val_annotation = {}
    for name, value in gt_annotations.items():
        val_name = name.replace('gt_', 'res_')
        new_val_annotation[val_name] = gt_annotations[name]
    # with open(new_ann_file, 'w+', encoding='utf-8') as f:
    #     json.dump(OrderedDict(new_val_annotation), f)
    print('write down {}'.format(new_ann_file))


if __name__ == '__main__':
    train_ann = osp.join(ArT_annotation_path, 'sp_train_art_labels.json')
    val_ann = osp.join(ArT_annotation_path, 'sp_val_art_labels.json')
    train_ann_detail = osp.join(ArT_annotation_path, 'sp_train_detail_labels.json')
    val_ann_detail = osp.join(ArT_annotation_path, 'sp_val_detail_labels.json')
    eval_val_ann = osp.join(ArT_annotation_path, 'sp_val_eval_labels.json')
    rewrite_art_annotations(ArT_data_path, train_ann, train_ann_detail,
                            ArT_trainset_path)
    rewrite_art_annotations(ArT_data_path, val_ann, val_ann_detail,
                            ArT_valset_path)
    add_eval_json(val_ann, eval_val_ann)








