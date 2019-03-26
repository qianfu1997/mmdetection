#
# @author:charlotte.Song
# @file: art_loader.py
# @Date: 2019/3/15 20:58
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import torchvision.transforms as transforms
import random
import pyclipper
import Polygon as plg
import cv2
import util
import os
import os.path as osp
from collections import OrderedDict
import json


art_root = './data_art/ArT/'
art_train_data_root = art_root + 'sp_train_art_images/'
art_val_data_root = art_root + 'sp_val_art_images/'
art_train_ann_path = art_root + 'annotations/sp_train_art_labels.json'
art_val_ann_path = art_root + 'annotations/sp_val_art_labels.json'

random.seed(123456)
""" a dataloader for art """


def get_img(img_path):
    try:
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]
    except Exception as e:
        print(img_path)
        raise
    return img

def refine_polygon_anns(img, anns):
    """ """
    h, w = img.shape[:2]
    for idx in range(len(anns)):
        points = np.array(anns[idx]["points"]).reshape(-1, 2).astype(np.float32)
        points[:, 0::2] = points[:, 0::2] / w * 1.0
        points[:, 1::2] = points[:, 1::2] / h * 1.0
        points = points.tolist()
        anns[idx]["points"] = points
    return anns


def scale_aligned(img, scale):
    h, w = img.shape[0:2]
    h = (int)(h * scale + 0.5)
    w = (int)(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


def random_scale(img, min_size, short_size=512):
    h, w = img.shape[0:2]

    random_scale = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    scale = (np.random.choice(random_scale) * short_size) / min(h, w)

    # img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    img = scale_aligned(img, scale)
    return img


def dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri


def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        pco = pyclipper.PyclipperOffset()
        pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        offset = min((int)(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)

        shrinked_bbox = np.array(pco.Execute(-offset))

        if len(shrinked_bbox.shape) != 3 or shrinked_bbox.shape[1] <= 2:
            shrinked_bboxes.append(bbox)
        else:
            shrinked_bboxes.append(shrinked_bbox[0])

    return np.array(shrinked_bboxes)

def shrink_plg(plg_points, rate, max_shr=20):
    rate = rate ** 2
    area = plg.Polygon(plg_points).area()
    peri = perimeter(plg_points)

    pco = pyclipper.PyclipperOffset()
    pco.AddPath(plg_points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    offset = min((int)(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)

    shrinked_plg_points = np.array(pco.Execute(-offset))

    if len(shrinked_plg_points.shape) != 3 or shrinked_plg_points.shape[1] <= 2:
        return np.array(shrinked_plg_points)
    else:
        return np.array(shrinked_plg_points[0])


def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs

def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
        imgs[i] = img_rotation
    return imgs

def random_crop_padding(imgs, target_size):
    """ using padding and the final crop size is (800, 800) """
    h, w = imgs[0].shape[0:2]
    t_w, t_h = target_size
    p_w, p_h = target_size
    if w == t_w and h == t_h:
        return imgs

    t_h = t_h if t_h < h else h
    t_w = t_w if t_w < w else w

    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        # make sure to crop the text region
        tl = np.min(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        br[br < 0] = 0
        br[0] = min(br[0], h - t_h)
        br[1] = min(br[1], w - t_w)

        i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
        j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
    else:
        i = random.randint(0, h - t_h) if h - t_h > 0 else 0
        j = random.randint(0, w - t_w) if w - t_w > 0 else 0

    n_imgs = []
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            s3_length = int(imgs[idx].shape[-1])
            img = imgs[idx][i:i + t_h, j:j + t_w, :]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT, value=tuple(0 for i in range(s3_length)))
        else:
            img = imgs[idx][i:i + t_h, j:j + t_w]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT, value=(0,))
        n_imgs.append(img_p)
    return n_imgs


class ArTLoader(data.Dataset):
    def __init__(self, split='train', is_transform=False, img_size=None,
                 kernel_scale=0.7, short_size=512):
        """ use the img_name as index """
        self.split = split
        self.is_transform = is_transform

        self.img_size = img_size if (img_size is None or isinstance(img_size, tuple)) else (img_size, img_size)
        self.kernel_scale = kernel_scale
        self.short_size = short_size

        if split == 'train':
            img_prefix = art_train_data_root
            ann_file = art_train_ann_path
        else:
            img_prefix = art_val_data_root
            ann_file = art_val_ann_path

        with open(ann_file, 'r') as f:
            gt_annotations = json.loads(f.read(), object_pairs_hook=OrderedDict)
        self.img_names = [name for name, _ in gt_annotations.items()]
        self.img_paths = [osp.join(img_prefix, name + '.jpg') for name, _ in gt_annotations.items()]
        self.img_prefix = img_prefix
        self.gt_annotations = gt_annotations

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img = get_img(osp.join(self.img_prefix, img_name + '.jpg'))
        img_annotation = self.gt_annotations[img_name]
        img_annotation = refine_polygon_anns(img, img_annotation)

        if self.is_transform:
            img = random_scale(img, self.img_size[0], self.short_size)

        gt_instance = np.zeros(img.shape[0:2], dtype='uint8')
        training_mask = np.ones(img.shape[0:2], dtype='uint8')
        #
        for plg_idx in range(len(img_annotation)):
            points = np.array(img_annotation[plg_idx]["points"]).reshape(-1, 2).astype(np.float32)
            points[:, 0::2] = points[:, 0::2] * img.shape[1]
            points[:, 1::2] = points[:, 1::2] * img.shape[0]
            points = points.astype(np.int64)
            cv2.drawContours(gt_instance, [points], -1, plg_idx + 1, -1)
            if img_annotation[plg_idx]["illegibility"]:
                cv2.drawContours(training_mask, [points], -1, 0, -1)
            img_annotation[plg_idx]["points"] = points.tolist()

        gt_kernals = []
        for rate in [self.kernel_scale]:
            gt_kernal = np.zeros(img.shape[:2], dtype='uint8')
            for plg_idx in range(len(img_annotation)):
                points = np.array(img_annotation[plg_idx]["points"]).reshape(-1, 2).astype(np.int64)
                kernal_points = shrink_plg(points, rate)
                if len(kernal_points.shape) == 1 and kernal_points.shape[0] >= 2:
                    for i in range(kernal_points.shape[0]):
                        if len(np.array(kernal_points[i]).shape) == 2 and np.array(kernal_points[i]).shape[0] > 3:
                            kernal_points = np.array(kernal_points[i]).reshape(-1, 2)
                            break
                kernal_points = np.array(kernal_points).astype(np.int64)
                if len(kernal_points) != 0:
                    cv2.drawContours(gt_kernal, [kernal_points], -1, 1, -1)
            gt_kernals.append(gt_kernal)

        if self.is_transform:
            imgs = [img, gt_instance, training_mask]
            imgs.extend(gt_kernals)

            imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs)
            imgs = random_crop_padding(imgs, self.img_size)

            img, gt_instance, training_mask, gt_kernals = imgs[0], imgs[1], imgs[2], imgs[3:]

        gt_text = gt_instance.copy()
        gt_text[gt_text > 0] = 1
        gt_kernals = np.array(gt_kernals)

        # '''
        if self.is_transform:
            img = Image.fromarray(img)
            img = img.convert('RGB')
            # img = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(img)
        else:
            img = Image.fromarray(img)
            img = img.convert('RGB')

        img = transforms.ToTensor()(img)
        # img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        gt_instance = torch.from_numpy(gt_instance).long()
        gt_text = torch.from_numpy(gt_text).float()
        gt_kernals = torch.from_numpy(gt_kernals).float()
        training_mask = torch.from_numpy(training_mask).float()
        # '''
        return img, gt_text, gt_kernals, training_mask, gt_instance













