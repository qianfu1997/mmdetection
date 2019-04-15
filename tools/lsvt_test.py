#
# @author:charlotte.Song
# @file: lsvt_test.py
# @Date: 2019/3/7 15:09
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import argparse
import torch
import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel
from mmdet import datasets
from mmdet.core import results2json, coco_eval
from mmdet.datasets import build_dataloader
from mmdet.models import build_detector, detectors
from mmdet.core import tensor2imgs, get_classes
import os
import cv2
import os.path as osp
import pycocotools.mask as maskUtils
from postmodule.post_processor import PostProcessor
import json


""" test augmentation: rescale + flip """


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet text for ArT')
    parser.add_argument('config', help='test config file parser')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--gpus', default=1, type=int)
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show result')
    args = parser.parse_args()
    return args


def single_test(model, data_loader, show=False):
    """ use img_meta to get ori_shape, img_shape,
        and flip.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg,
                                     dataset=dataset.CLASSES)
        batch_size = data['img'].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def single_test_json(model, data_loader, post_processor, save_json_file, show=True, show_path=None):
    """
    :param model: model
    :param data_loader: data_loader
    :param post_processor:  use this to generate bbox from mask
    :param save_json_file:  the path of json file.
    :return:
    """
    # first get the pred, then get the bboxes from masks
    # use post_processor to filter the bboxes and then save in json file.
    # masks are generated by a set of data augmentation functions.
    model.eval()
    results = {}
    # get the img_infos from dataset,
    dataset = data_loader.dataset
    img_prefix = dataset.img_prefix
    img_norm_cfg = dataset.img_norm_cfg
    dataset_class = dataset.CLASSES
    prog_bar = mmcv.ProgressBar(len(dataset))
    imgs_bboxes_results = {}
    for i, data in enumerate(data_loader):
        """ get the name, height, width from data['img_meta'] 
            for multi-scale test, img_meta contains several img_meta
        """
        with torch.no_grad():
            # can change this to True.
            result = model(return_loss=False, rescale=False, **data)
        # deal with the mask and post processing here.
        # as masks are fit to the original imgs, just reload the original imgs
        #
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
        img_tensor = data['img'][0]
        img_metas = data['img_meta'][0].data[0]
        filename = img_metas[0]['filename']
        img_name = osp.splitext(filename)[0]
        imgs = tensor2imgs(img_tensor, **img_norm_cfg)
        assert len(imgs) == len(img_metas)
        # bboxes and masks will be rescaled to the size of imgs[0]
        img_0, img_meta_0 = imgs[0], img_metas[0]
        h, w, _ = img_meta_0['img_shape']
        ori_h, ori_w, _ = img_meta_0['ori_shape']
        scales = (ori_w * 1.0 / w, ori_h * 1.0 / h)
        # use the scores and segm_result to generate bbox and mask.
        vs_bbox_result = np.vstack(bbox_result)
        if segm_result is None:
            pred_bboxes = []
        else:
            segm_scores = np.asarray(vs_bbox_result[:, -1])
            segms = mmcv.concat_list(segm_result)
            # the bboxes returned by processor are fit to the original images.
            pred_bboxes, pred_bbox_scores = post_processor.process(segms, segm_scores,
                                                                  mask_shape=img_meta_0['img_shape'], scale_factor=scales)
        # save the results.
        single_pred_results = []
        for pred_bbox, pred_bbox_score in zip(pred_bboxes, pred_bbox_scores):
            pred_bbox = pred_bbox.reshape((4, 2))
            single_bbox_dict = {
                "points": pred_bbox,
                "confidence": float(pred_bbox_score)
            }
            single_pred_results.append(single_bbox_dict)
        imgs_bboxes_results[img_name] = single_pred_results
        # print the postmodule pics.
    if show:
        """ use the result: imgs_bboxes_result to generate show pics
        """
        if show_path is not None and osp.isdir(show_path):
            os.mkdir(show_path)
        if imgs_bboxes_results is not None:
            for name, values in imgs_bboxes_results.items():
                filename = name + '.jpg'
                img = cv2.imread(osp.join(img_prefix, filename))
                for idx in range(len(values)):
                    bbox = np.asarray(values[idx]["points"]).reshape(-1, 2).astype(np.int64)
                    cv2.drawContours(img, [bbox], -1, (0, 0, 255), 2)
                cv2.imwrite(osp.join(show_path, filename), img)
    # save the results.
    with open(save_json_file, 'w+', encoding='utf-8') as f:
        json.dump(imgs_bboxes_results, f)




















