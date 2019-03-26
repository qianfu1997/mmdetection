#
# @author:charlotte.Song
# @file: lsvt_eval.py
# @Date: 2019/3/4 17:12
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
import Polygon as plg


""" load submit.json and compare with eval annotations. 
    using scripts like ctw_script.py
    in submit.json the key name is 'res_{image_id}'  
    and in eval_ann_file(xxx.json) the key name is 'res_{image_id}'
"""


eval_ann_file = '../data/IC19/LSVT/annotations/sp_val_eval_labels.json'
submit_path = '../submit/lsvt/'
art_eval_ann_file = '../data/IC19/ArT/annotations/sp_val_eval_labels.json'
art_submit_path = '../submit/art/'


def get_union(pa, pb):
    pa_area = pa.area()
    pb_area = pb.area()
    return pa_area + pb_area - get_intersection(pa, pb)


def get_intersection(pa, pb):
    pInt = pa & pb
    if len(pInt) == 0:
        return 0
    else:
        return pInt.area()


def evaluation(submit_file, eval_ann, threshold=0.1, confidence=0.3):
    assert osp.isfile(submit_file)
    assert osp.isfile(eval_ann)
    with open(eval_ann, 'r', encoding='utf-8') as f:
        gt_annotations = json.loads(f.read(), object_pairs_hook=OrderedDict)
    with open(submit_file, 'r', encoding='utf-8') as f:
        preds = json.loads(f.read(), object_pairs_hook=OrderedDict)

    tp, fp, npos = 0, 0, 0
    """ for validation the annotation is same as the trainset. """
    for name, pred_ann in preds.items():
        gt_annotation = gt_annotations[name]    # is a list
        # npos should filter the ignored box.
        # npos += len(gt_annotation)
        for gt_polygon_tran in gt_annotation:
            if gt_polygon_tran["illegibility"]:
                continue
            npos += 1

        cover = set()
        # compare each pred_polygon with each gt_polygon
        for pred_polygon_pro in pred_ann:
            pred_polygon = np.array(pred_polygon_pro["points"])   # shape: [n, 2]
            pred_prob = np.float32(pred_polygon_pro["confidence"])
            if pred_prob < confidence:      # confidence threshold.
                continue
            pred_polygon = plg.Polygon(pred_polygon)
            flag = False
            is_ignore = False

            for gt_id, gt_polygon_tran in enumerate(gt_annotation):
                gt_illegibility = gt_polygon_tran["illegibility"]
                gt_polygon = np.array(gt_polygon_tran["points"]).reshape(-1, 2).astype(np.int64)
                gt_polygon = plg.Polyogn(gt_polygon)

                union = get_union(pred_polygon, gt_polygon)
                intersection = get_intersection(pred_polygon, gt_polygon)
                if intersection * 1.0 / union >= threshold and flag is False:
                    if gt_id not in cover:
                        flag = True
                        if gt_illegibility:
                            is_ignore = True
                        cover.add(gt_id)
            if flag:
                tp += 0.0 if is_ignore else 1.0
            else:
                fp += 1.0

    precision = tp / (fp + tp)
    recall = tp / npos      # recall do not calculate the ignored ones.
    hmean = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    print("p: {:f}  r: {:f}  h: {:f}".format(precision, recall, hmean))


def parse_args():
    parser = argparse.ArgumentParser(description='art postmodule')
    parser.add_argument('--submit_file', type=str, default=None)
    parser.add_argument('--gt_file', type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    eval_tool = evaluation(submit_file=args.submit_file, eval_ann=args.gt_file,)




















