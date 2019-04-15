#
# @author:charlotte.Song
# @file: LsvtCropDataset.py
# @Date: 2019/4/11 21:16
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path as osp
import os
import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         Numpy2Tensor)
from .utils import to_tensor, random_scale
from .extra_aug import ExtraAugmentationIC
from .custom_crop import CustomCropDataset
import json
import cv2
from collections import OrderedDict
""" a crop version of lsvt """


def lsvt_classes():
    return ['text']

label_ids = {name: i + 1 for i, name in enumerate(lsvt_classes())}
debug_path = '/home/data3/sxg/IC19/mmdetection-master/visualization/debug/'


class LsvtCropDataset(CustomCropDataset):
    """Custom dataset for detection.

        Annotation format:
        [
            {
                'filename': 'a.jpg',
                'width': 1280,
                'height': 720,
                'ann': {
                    'bboxes': <np.ndarray> (n, 4),
                    'labels': <np.ndarray> (n, ),
                    'bboxes_ignore': <np.ndarray> (k, 4),
                    'labels_ignore': <np.ndarray> (k, 4) (optional field)
                    'img_annotation': <list<dict>>
                    'polygon_points:' <list<np.ndarray>>
                }
            },
            ...
        ]
        The `ann` field is optional for testing.
        """

    CLASSES = ('text')

    def check_box(self, width, height, bboxes):
        assert bboxes and len(bboxes.shape) == 2
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, width - 1)
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, height - 1)
        return bboxes

    def getBboxAndLabels(self, height, width, annotations):
        bboxes, labels = [], []
        bboxes_ignore, labels_ignore = [], []
        for i in range(len(annotations)):
            points = np.asarray(annotations[i]["points"]).reshape(-1, 2).astype(np.int64)
            box = np.hstack((np.min(points), np.max(points)))
            label = label_ids['text']
            if annotations[i]["illegibility"]:
                bboxes_ignore.append(box)
                labels_ignore.append(label)
            else:
                bboxes.append(box)
                labels.append(label)

        if bboxes:
            bboxes = np.array(bboxes, dtype=np.float32)
            bboxes = self.check_box(width, height, bboxes)
            labels = np.array(labels, dtype=np.int64)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.array([], dtype=np.int64)
        if bboxes_ignore:
            bboxes_ignore = np.array(bboxes_ignore, dtype=np.float32)
            bboxes_ignore = self.check_box(width, height, bboxes_ignore)
            labels_ignore = np.array(labels_ignore, dtype=np.int64)
        else:
            bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
            labels_ignore = np.array([], dtype=np.int64)

        return bboxes, labels, bboxes_ignore, labels_ignore

    def load_annotations(self, ann_file):
        assert osp.isdir(self.img_prefix), 'Error : wrong path.'
        assert osp.isfile(ann_file), 'Error: Wrong ann file'

        detailed_ann_file = ann_file.replace('_full_', '_detail_')
        detailed_annotation = None
        img_infos = []
        files = os.listdir(self.img_prefix)

        with open(ann_file, 'r', encoding='utf-8') as f:
            gt_annotations = json.loads(f.read(), object_pairs_hook=OrderedDict)
        if osp.isfile(detailed_ann_file):
            with open(detailed_ann_file, 'r', encoding='utf-8') as f:
                detailed_annotation = json.loads(f.read(), object_pairs_hook=OrderedDict)

        for indx in range(len(files)):
            name = osp.splitext(files[indx])[0]
            if detailed_annotation is None:
                img = cv2.imread(osp.join(self.img_prefix, files[indx]))
                height, width = img.shape[:2]
            else:
                height, width = detailed_annotation[name]['height'], detailed_annotation[name]['width']
            info = {
                'filename': files[indx],
                'heigt': height,
                'width': width,
                'img_annotation': gt_annotations[name]}
            img_infos.append(info)
            if indx % 2000 == 0:
                print('{:d} % {:d}'.format(indx, len(files)))

        return img_infos

    def generate_mask_ann(self, height, width, annotations):
        gt_masks, ignore_masks = [], []
        for i in range(len(annotations)):
            points = np.array(annotations[i]["points"]).reshape(-1, 2).astype(np.int64)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask, [points], -1, label_ids['text'], -1)
            if annotations[i]["illegiblity"]:
                ignore_masks.append(mask)
            else:
                gt_masks.append(mask)
        return gt_masks, ignore_masks

    def debug(self, idx, ann):
        if not osp.isdir(debug_path):
            os.makedirs(debug_path)
        masks = ann['masks']
        bboxes = ann['bboxes']
        assert len(bboxes) == len(masks)
        instance_show = np.zeros((masks[0].shape[0], masks[0].shape[1], 3), dtype=np.uint8)
        for i in range(len(bboxes)):
            instance_show[masks[i] > 0] = (255, 255, 255)
            top_left, bottom_right = bboxes[i, :2], bboxes[i, 2:]
            cv2.rectangle(instance_show, tuple(top_left), tuple(bottom_right), (0, 255, 0), 2)
        cv2.imwrite(osp.join(debug_path, '{:d}.jpg'.format(idx)))

    def get_ann_info(self, idx):
        height = self.img_infos[idx]['height']
        width = self.img_infos[idx]['width']
        bboxes, labels, bboxes_ignore, labels_ignore = self.getBboxAndLabels(
            height=height, width=width, annotations=self.img_infos[idx]['img_annotation'])
        ann = {
            'bboxes': bboxes,
            'labels': labels,
            'bboxes_ignore': bboxes_ignore,
            'labels_ignore': labels_ignore}

        if self.with_mask:
            gt_masks, gt_ignore_masks = self.generate_mask_ann(height, width, self.img_infos[idx]['img_annotation'])
            ann['masks'] = gt_masks
            ann['ignore_masks'] = gt_ignore_masks
            assert len(gt_masks) == bboxes.shape[0]
        return ann

    def prepare_test_img(self, idx):
        """
        prepare an image for testing (multi-scale and flipping)
        add the filename in img_metas
        :param idx:
        :return:
        """
        img_info = self.img_infos[idx]
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        if self.proposals is not None:
            proposal = self.proposals[idx][:self.num_max_proposals]
            if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposal.shape))
        else:
            proposal = None

        def prepare_single(img, scale, flip, proposal=None):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack(
                    [_proposal, score]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _img, _img_meta, _proposal = prepare_single(
                img, scale, False, proposal)
            _img_meta['filename'] = img_info['filename']
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                # if flip_ratio, add an rotation img.
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, proposal)
                _img_meta['filename'] = img_info['filename']
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        data = dict(img=imgs, img_meta=img_metas)
        if self.proposals is not None:
            data['proposals'] = proposals
        return data








