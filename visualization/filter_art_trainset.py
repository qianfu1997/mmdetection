#
# @author:charlotte.Song
# @file: filter_var_trainset.py
# @Date: 2019/4/9 10:50
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import argparse
import os
import os.path as osp
import cv2
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import json
from collections import OrderedDict



""" use plt and cv2 to read image, if the shape not equal, save the plt image """
def rewrite(path, imgname, gt_annotation, debug_path):
    assert osp.exists(path) and osp.isfile(osp.join(path, imgname))
    cv_img = cv2.imread(osp.join(path, imgname))
    plt_img = plt.imread(osp.join(path, imgname))
    Image_img = Image.open(osp.join(path, imgname))

    h, w = cv_img.shape[:2]
    plt_h, plt_w = plt_img.shape[:2]
    pil_w, pil_h = Image_img.size

    if not(plt_h == pil_h == h) and (plt_w == w == pil_w):
        """ show the wrong image and the correct image with gt"""
        print(imgname)
        show_correct(Image_img, imgname, gt_annotation, debug_path)
        Image_img.save(osp.join(path, imgname))


def show_correct(img, imgname, gt_annotation, debug_path):
    """ img will be changed to ndarray here """
    img = np.array(img)

    for ann in gt_annotation:
        gt_box = np.array(ann['points']).reshape(-1, 2).astype(np.int64)
        if ann['illegibility']:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
        cv2.drawContours(img, [gt_box], -1, color, 2)
    img = Image.fromarray(img)
    img.save(osp.join(debug_path, imgname))


def parse_args():
    parser = argparse.ArgumentParser(description='filter training image')
    parser.add_argument('--train_set', type=str, default=None,
                        help='path of trainset')
    parser.add_argument('--gt_path', type=str, default=None,
                        help='gt annotation path')
    parser.add_argument('--debug_path', type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert osp.isdir(args.train_set) and osp.isfile(args.gt_path)
    if not osp.exists(args.debug_path):
        os.makedirs(args.debug_path)
    with open(args.gt_path, 'r', encoding='utf-8') as f:
        gt_annotations = json.loads(f.read(), object_pairs_hook=OrderedDict)

    for name, gt_annotation in gt_annotations.items():
        rewrite(args.train_set, name + '.jpg', gt_annotation, args.debug_path)

    print('done')







