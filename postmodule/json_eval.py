#
# @author:charlotte.Song
# @file: json_eval.py
# @Date: 2019/3/7 15:25
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import argparse
import os
import os.path as osp
import torch
import Polygon as plg
import json


""" a eval script for IC19. 
    load submit.json and compare the results with eval annotations.
"""


def parse_args():
    """ get settings. """
    parser = argparse.ArgumentParser(description='json_eval')
    parser.add_argument('-g', '--gt', type=str, help='path of gt json file.',
                        default='')
    parser.add_argument('-s', '--submit', type=str, help='path of submit json file',
                        default=None)
    parser.add_argument('-t', '--threshold', type=float,
                        help='IoU threshold for evalutaion.',
                        default=0.5)
    parser.add_argument('-c', '--confidence', type=float,
                        help='the confidence threshold for postmodule')
    args = parser.parse_args()
    return args


class json_eval():
    def __init__(self, gt, submit, threshold=0.5, confidence=0.3, filter_confidence=False, **kwargs):
        self.gt = gt
        self.submit = submit
        self.threshold = threshold
        self.confidence = confidence
        self.filter_confidence = filter_confidence

    def get_union(self, pa, pb):
        return pa.area() + pb.area()

    def get_intersection(self, pa, pb):
        pc = pa & pb
        return pc.area()

    def evaluation(self):
        with open(self.gt, 'r', encoding='utf-8') as fg:
            gt_annotations = json.loads(fg.read())
        with open(self.submit, 'r', encoding='utf-8') as fp:
            pred_annotations = json.loads(fp.read())
        tp, fp, npos = 0, 0, 0
        """ filter out the ignored gt annotation """
        for gt_polygon_tran in gt_annotations:
            if gt_polygon_tran["illegibility"]:
                continue
            npos += 1
        cover = set()
        # No need to filter the confidence, but can try
        for pred_polygon_pro in pred_annotations:
            pred_polygon = np.array(pred_polygon_pro["points"]).reshape(-1, 2).astype(np.int64)
            pred_prob = np.float32(pred_polygon_pro["confidence"])
            if self.filter_confidence and pred_prob < self.confidence:
                continue
            pred_polygon = plg.Polygon(pred_polygon)
            flag = False
            is_ignore = False

            for gt_id, gt_polygon_tran in enumerate(gt_annotations):
                gt_ignore = gt_polygon_tran["illegibility"]
                gt_polygon = np.array(gt_polygon_tran["points"])
                gt_polygon = plg.Polygon(gt_polygon)

                union = self.get_union(gt_polygon, pred_polygon)
                intersection = self.get_intersection(gt_polygon, pred_polygon)
                if intersection * 1.0 / union < self.threshold:
                    if gt_id not in cover:
                        flag = True
                        if gt_ignore:
                            is_ignore = True
                        cover.add(gt_id)
            if flag:
                tp += 0.0 if is_ignore else 1.0
            else:
                fp += 1.0
        precision = tp / (fp + tp)
        recall = tp / npos
        hmean = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        print("p: {:f} r:{}  h: {:f}".format(precision, recall, hmean))





    













