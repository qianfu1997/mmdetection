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
import cv2


""" load submit.json and compare with eval annotations. 
    using scripts like ctw_script.py
    in submit.json the key name is 'res_{image_id}'  
    and in eval_ann_file(xxx.json) the key name is 'res_{image_id}'
    
    load the eval settings from json file. 'eval_setting.json'
"""


eval_ann_file = '../data/LSVT/annotations/sp_val_eval_labels.json'
submit_path = '../submit/lsvt/'
art_eval_ann_file = '../data/ArT/annotations/sp_val_eval_labels.json'
art_submit_path = '../submit/art/'
art_val_data_root = '/home/data1/IC19/ArT/sp_val_art_images/'

debug_path = '../visualization/debug/'
stastic_path = '../visualization/stastics/'


def parse_args():
    parser = argparse.ArgumentParser(description='art postmodule')
    parser.add_argument('--submit_file', type=str, default=None)
    parser.add_argument('--gt_file', type=str, default=None)
    parser.add_argument('--nms_thr', type=float, default=0.5)
    parser.add_argument('--con_thr', type=float, default=0.0)
    parser.add_argument('--eval_f', type=str, default='ctw',
                        help='ctw or xez')
    parser.add_argument('--stastics', action='store_true')
    parser.add_argument('--search', action='store_true')        # search best h1 score
    # for search mode
    parser.add_argument('--low_score', default=0.5, type=float)
    parser.add_argument('--high_score', default=0.95, type=float)
    parser.add_argument('--step', type=float, default=0.05)
    args = parser.parse_args()
    return args


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


# get scales of polygon
def get_rescale_factor(name, long_size=2560, short_size=800, keep_ratio=True):
    name = osp.join(art_val_data_root, name.replace('res_', 'gt_') + '.jpg')
    img = cv2.imread(name)
    h, w = img.shape[:2]
    if keep_ratio:
        scale_factor = min(
            long_size * 1.0 / max(h, w),
            short_size * 1.0 / min(h, w),
        )
    else:
        scale_factor = [
            long_size * 1.0 / max(h, w),
            short_size * 1.0 / min(h, w)
        ]
        scale_factor = [scale_factor[0], scale_factor[1]] if h >= w else [scale_factor[1], scale_factor[0]]
        scale_factor = np.array(scale_factor).astype(np.float32)
    return scale_factor


def evaluation(submit_file, eval_ann, threshold=0.5, confidence=0.0,
               debug_mode=False, stastics_mode=False):
    assert osp.isfile(submit_file)
    assert osp.isfile(eval_ann)
    with open(eval_ann, 'r', encoding='utf-8') as f:
        gt_annotations = json.loads(f.read(), object_pairs_hook=OrderedDict)
    with open(submit_file, 'r', encoding='utf-8') as f:
        preds = json.loads(f.read(), object_pairs_hook=OrderedDict)

    tp, fp, npos = 0, 0, 0
    lost_gt_polygon = []
    """ for validation the annotation is same as the trainset. """
    for name, pred_ann in preds.items():
        gt_annotation = gt_annotations[name]    # is a list
        # npos should filter the ignored box.
        for gt_polygon_tran in gt_annotation:
            if gt_polygon_tran["illegibility"]:
                continue
            npos += 1

        cover = set()
        # compare each pred_polygon with each gt_polygon
        for pred_polygon_pro in pred_ann:
            pred_polygon = np.array(pred_polygon_pro["points"])   # shape: [n, 2]
            pred_polygon = plg.Polygon(pred_polygon)
            # set the score threshold
            if pred_polygon_pro["confidence"] < confidence:
                continue
            flag = False
            is_ignore = False

            for gt_id, gt_polygon_tran in enumerate(gt_annotation):
                gt_illegibility = gt_polygon_tran["illegibility"]
                gt_polygon = np.array(gt_polygon_tran["points"]).reshape(-1, 2).astype(np.int64)
                gt_polygon = plg.Polygon(gt_polygon)

                union = get_union(pred_polygon, gt_polygon)
                intersection = get_intersection(pred_polygon, gt_polygon)
                if (flag is False) and (intersection * 1.0 / union) >= threshold:
                    if gt_id not in cover:
                        flag = True
                        cover.add(gt_id)
                        if gt_illegibility:
                            is_ignore = True

            if flag:
                tp += 0.0 if is_ignore else 1.0
            else:
                fp += 1.0

        if stastics_mode:
            """ calculate all lost gt bboxes """
            scale_factor = get_rescale_factor(name)
            for gt_id, gt_polygon_tran in enumerate(gt_annotation):
                gt_polygon = np.array(gt_polygon_tran["points"]).reshape(-1, 2).astype(np.float32)
                # needed to
                gt_polygon = (gt_polygon * scale_factor).astype(np.int64)
                # gt_polygon_area = plg.Polygon(gt_polygon).area()
                ltop = np.min(gt_polygon, axis=0)
                rbottom = np.max(gt_polygon, axis=0)
                gt_polygon_area = (rbottom[0] - ltop[0]) * (rbottom[1] - ltop[1])
                if (gt_id not in cover) and (gt_polygon_tran["illegibility"] is False):
                    lost_gt_polygon.append(gt_polygon_area)

        if debug_mode:
            precision = tp / (fp + tp)
            recall = tp / npos
            hmean = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
            print("p: {:f}  r: {:f}  h: {:f}".format(precision, recall, hmean))
            return name

    precision = tp / (fp + tp)
    recall = tp / npos      # recall do not calculate the ignored ones.
    hmean = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    print("p: {:f}  r: {:f}  h: {:f}".format(precision, recall, hmean))

    if stastics_mode:
        """ print """
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        if not os.path.exists(stastic_path):
            os.makedirs(stastic_path)
        if len(lost_gt_polygon) > 0:
            lost_gt_polygon = np.asarray(lost_gt_polygon)
            plt.hist(lost_gt_polygon[lost_gt_polygon < 10000], bins='auto', histtype='bar')
            plt.title('lost gt bbox area')
            plt.xlabel('area')
            plt.ylabel('frequency')
            plt.savefig(os.path.join(stastic_path, 'stastics_plot.jpg'))
            print('totalnum: {:d}; min: {:f}; max: {:f}'.format(len(lost_gt_polygon), min(lost_gt_polygon), max(lost_gt_polygon)))
            print('write down {}'.format(os.path.join(stastic_path, 'stastics_plot.jp')))

    return hmean


def sigma_calculation(det_p, gt_p):
    """
    sigma = inter_area / gt_area
    """
    return get_intersection(det_p, gt_p) / gt_p.area()


def tau_calculation(det_p, gt_p):
    """ tau = inter_area / det_area """
    return get_intersection(det_p, gt_p) / det_p.area()


# detection_filtering
# filter out the pred bboxes which has a IoU > thr with ignored gt bboxes.
def detection_filtering(pred_ann, gt_ann, thr=0.5):
    filtered_pred_anns = []
    filtered_gt_anns = []
    filtered = []
    for gt_indx in range(len(gt_ann)):
        if not gt_ann[gt_indx]["illegibility"]:
            filtered_gt_anns.append(gt_ann[gt_indx])
            continue
        gt_points = np.array(gt_ann[gt_indx]["points"]).reshape(-1, 2).astype(np.int64)
        gt_polygon = plg.Polygon(gt_points)

        for pred_indx in range(len(pred_ann)):
            pred_points = np.array(pred_ann[pred_indx]["points"]).reshape(-1, 2).astype(np.int64)
            pred_polygon = plg.Polygon(pred_points)

            pred_gt_iou = get_intersection(pred_polygon, gt_polygon) / (pred_polygon.area() + 1e-5)
            if pred_gt_iou >= thr:
                # keep the boxes that have small iou with ignored gt box
                # filtered_pred_anns.append(pred_ann[pred_indx])
                filtered.append(pred_indx)
    for pred_indx in range(len(pred_ann)):
        if pred_indx not in filtered:
            filtered_pred_anns.append(pred_ann[pred_indx])
    return filtered_pred_anns, filtered_gt_anns


def one2one(local_sigma_table, local_tau_table, local_accumulative_recall,
            local_accumulative_precision, global_accumulative_recall, global_accumulative_precision,
            gt_flag, pred_flag, num_gt, tr, tp):
    """
    :param local_sigma_table: local sigma iou positive
    :param local_tau_table: local tau iou positive
    :param local_accumulative_recall: local recall
    :param local_accumulative_precision: local prec
    :param global_accumulative_recall: global recall
    :param global_accumulative_precision: global prec
    :param gt_flag:
    :param det_flag:
    :return:
    """
    for gt_id in range(num_gt):
        # sigma and tau are different iou results.
        # select the ones that iou > thr.
        qualified_sigma_candidiates = np.where(local_sigma_table[gt_id, :] > tr)
        num_qualified_sigma_candidates = qualified_sigma_candidiates[0].shape[0]
        qualified_tau_candidates = np.where(local_tau_table[gt_id, :] > tp)
        num_qualified_tau_candidates = qualified_tau_candidates[0].shape[0]

        if (num_qualified_sigma_candidates == 1) and (num_qualified_tau_candidates == 1):
            global_accumulative_recall = global_accumulative_recall + 1.0
            global_accumulative_precision = global_accumulative_precision + 1.0
            local_accumulative_recall = local_accumulative_recall + 1.0
            local_accumulative_precision = local_accumulative_precision + 1.0

            gt_flag[0, gt_id] = 1       # one2one
            matched_pred_id = np.where(local_sigma_table[gt_id, :] > tr)
            pred_flag[0, matched_pred_id] = 1       # one2one
    return local_accumulative_recall, local_accumulative_precision, \
           global_accumulative_recall, global_accumulative_precision, \
           gt_flag, pred_flag


def one2many(local_sigma_table, local_tau_table, local_accumulative_recall,
            local_accumulative_precision, global_accumulative_recall, global_accumulative_precision,
            gt_flag, pred_flag, num_gt, tr, tp, k, fsc_k):
    for gt_id in range(num_gt):
        if gt_flag[0, gt_id] > 0:   # one2one.
            continue

        non_zero_in_sigma = np.where(local_sigma_table[gt_id, :] > 0)
        num_non_zero_in_sigma = non_zero_in_sigma[0].shape[0]

        if num_non_zero_in_sigma >= k:
            ####search for all detections that overlaps with this groundtruth
            qualified_tau_candidates = np.where((local_tau_table[gt_id, :] >= tp) & (pred_flag[0, :] == 0))
            num_qualified_tau_candidates = qualified_tau_candidates[0].shape[0]

            if num_qualified_tau_candidates == 1:
                if ((local_tau_table[gt_id, qualified_tau_candidates] >= tp) and (
                        local_sigma_table[gt_id, qualified_tau_candidates] >= tr)):
                    # became an one-to-one case
                    global_accumulative_recall = global_accumulative_recall + 1.0
                    global_accumulative_precision = global_accumulative_precision + 1.0
                    local_accumulative_recall = local_accumulative_recall + 1.0
                    local_accumulative_precision = local_accumulative_precision + 1.0

                    gt_flag[0, gt_id] = 1   # a good pair.
                    pred_flag[0, qualified_tau_candidates] = 1
            elif (np.sum(local_sigma_table[gt_id, qualified_tau_candidates]) >= tr):
                gt_flag[0, gt_id] = 1
                pred_flag[0, qualified_tau_candidates] = 1

                global_accumulative_recall = global_accumulative_recall + fsc_k
                global_accumulative_precision = global_accumulative_precision + num_qualified_tau_candidates * fsc_k

                local_accumulative_recall = local_accumulative_recall + fsc_k
                local_accumulative_precision = local_accumulative_precision + num_qualified_tau_candidates * fsc_k

    return local_accumulative_recall, local_accumulative_precision, \
           global_accumulative_recall, global_accumulative_precision, \
           gt_flag, pred_flag


def many2many(local_sigma_table, local_tau_table, local_accumulative_recall,
            local_accumulative_precision, global_accumulative_recall, global_accumulative_precision,
            gt_flag, pred_flag, num_pred, tr, tp, k, fsc_k):

    for pred_id in range(num_pred):
        if pred_flag[0, pred_id] > 0:
            continue

        non_zero_in_tau = np.where(local_tau_table[:, pred_id] > 0)
        num_non_zero_in_tau = non_zero_in_tau[0].shape[0]

        if num_non_zero_in_tau >= k:
            ####search for all detections that overlaps with this groundtruth
            qualified_sigma_candidates = np.where((local_sigma_table[:, pred_id] >= tp) & (gt_flag[0, :] == 0))
            num_qualified_sigma_candidates = qualified_sigma_candidates[0].shape[0]

            if num_qualified_sigma_candidates == 1:
                if ((local_tau_table[qualified_sigma_candidates, pred_id] >= tp) and (
                        local_sigma_table[qualified_sigma_candidates, pred_id] >= tr)):
                    # became an one-to-one case
                    global_accumulative_recall = global_accumulative_recall + 1.0
                    global_accumulative_precision = global_accumulative_precision + 1.0
                    local_accumulative_recall = local_accumulative_recall + 1.0
                    local_accumulative_precision = local_accumulative_precision + 1.0

                    gt_flag[0, qualified_sigma_candidates] = 1
                    pred_flag[0, pred_id] = 1
            elif (np.sum(local_tau_table[qualified_sigma_candidates, pred_id]) >= tp):
                pred_flag[0, pred_id] = 1
                gt_flag[0, qualified_sigma_candidates] = 1

                global_accumulative_recall = global_accumulative_recall + num_qualified_sigma_candidates * fsc_k
                global_accumulative_precision = global_accumulative_precision + fsc_k

                local_accumulative_recall = local_accumulative_recall + num_qualified_sigma_candidates * fsc_k
                local_accumulative_precision = local_accumulative_precision + fsc_k
    return local_accumulative_recall, local_accumulative_precision, \
           global_accumulative_recall, global_accumulative_precision, \
           gt_flag, pred_flag


def det_evaluation(submit_file, eval_ann, iou_thr=0.5, debug_mode=False, debug_name=None):
    """ settings """
    global_sigma, global_tau = [], []
    tr, tp, fsc_k, k = 0.7, 0.6, 0.8, 2
    global_accumulative_recall = 0
    global_accumulative_precision = 0
    total_num_gt = 0
    total_num_pred = 0

    """ read pred ann and gt ann """
    assert osp.isfile(submit_file)
    assert osp.isfile(eval_ann)
    with open(eval_ann, 'r', encoding='utf-8') as f:
        gt_annotations = json.loads(f.read(), object_pairs_hook=OrderedDict)
    with open(submit_file, 'r', encoding='utf-8') as f:
        pred_annotations = json.loads(f.read(), object_pairs_hook=OrderedDict)

    """ eval """
    if debug_mode:
        pred_ann = pred_annotations[debug_name]
        # first get the corresponding gt ann
        gt_ann = gt_annotations[debug_name]
        # filtered_pred_ann, and filtered_gt_ann has filtered out all
        # bboxes that overlap with ignored bboxes.
        filtered_pred_ann, filtered_gt_ann = detection_filtering(pred_ann, gt_ann)
        local_sigma_table = np.zeros((len(filtered_gt_ann), len(filtered_pred_ann)))
        local_tau_table = np.zeros((len(filtered_gt_ann), len(filtered_pred_ann)))

        """ compare each pred box with each gt box """
        for gt_idx in range(len(filtered_gt_ann)):
            if len(filtered_pred_ann) > 0:
                gt_points = np.array(filtered_gt_ann[gt_idx]["points"]).reshape(-1, 2).astype(np.int64)
                gt_polygon = plg.Polygon(gt_points)
                for pred_idx in range(len(filtered_pred_ann)):
                    pred_points = np.array(filtered_pred_ann[pred_idx]["points"]).reshape(-1, 2).astype(np.int64)
                    pred_polygon = plg.Polygon(pred_points)
                    # store each img's information of gt_idx and pred_idx.
                    local_sigma_table[gt_idx, pred_idx] = sigma_calculation(pred_polygon, gt_polygon)
                    local_tau_table[gt_idx, pred_idx] = tau_calculation(pred_polygon, gt_polygon)
        global_sigma.append(local_sigma_table)
        global_tau.append(local_tau_table)
    else:
        for name, pred_ann in pred_annotations.items():
            # first get the corresponding gt ann
            gt_ann = gt_annotations[name]
            # filtered_pred_ann, and filtered_gt_ann has filtered out all
            # bboxes that overlap with ignored bboxes.
            filtered_pred_ann, filtered_gt_ann = detection_filtering(pred_ann, gt_ann)
            local_sigma_table = np.zeros((len(filtered_gt_ann), len(filtered_pred_ann)))
            local_tau_table = np.zeros((len(filtered_gt_ann), len(filtered_pred_ann)))

            """ compare each pred box with each gt box """
            for gt_idx in range(len(filtered_gt_ann)):
                if len(filtered_pred_ann) > 0:
                    gt_points = np.array(filtered_gt_ann[gt_idx]["points"]).reshape(-1, 2).astype(np.int64)
                    gt_polygon = plg.Polygon(gt_points)
                    for pred_idx in range(len(filtered_pred_ann)):
                        pred_points = np.array(filtered_pred_ann[pred_idx]["points"]).reshape(-1, 2).astype(np.int64)
                        pred_polygon = plg.Polygon(pred_points)
                        # store each img's information of gt_idx and pred_idx.
                        local_sigma_table[gt_idx, pred_idx] = sigma_calculation(pred_polygon, gt_polygon)
                        local_tau_table[gt_idx, pred_idx] = tau_calculation(pred_polygon, gt_polygon)
            global_sigma.append(local_sigma_table)
            global_tau.append(local_tau_table)

    """ calculate recall and prec """
    for idx in range(len(global_sigma)):
        local_sigma_table = global_sigma[idx]
        local_tau_table = global_tau[idx]

        num_gt = local_sigma_table.shape[0]
        num_pred = local_sigma_table.shape[1]

        total_num_gt = total_num_gt + num_gt
        total_num_pred = total_num_pred + num_pred

        local_accumulative_recall = 0
        local_accumulative_precision = 0
        gt_flag = np.zeros((1, num_gt))
        pred_flag = np.zeros((1, num_pred))

        #######first check for one-to-one case##########
        """ see what one2one do """
        local_accumulative_recall, local_accumulative_precision, \
        global_accumulative_recall, global_accumulative_precision, \
        gt_flag, pred_flag = one2one(local_sigma_table, local_tau_table, local_accumulative_recall,
                                     local_accumulative_precision, global_accumulative_recall, global_accumulative_precision,
                                     gt_flag, pred_flag, num_gt, tr, tp)
        ######then check for one-to-many case##########
        local_accumulative_recall, local_accumulative_precision, \
        global_accumulative_recall, global_accumulative_precision, \
        gt_flag, pred_flag = one2many(local_sigma_table, local_tau_table, local_accumulative_recall,
                                      local_accumulative_precision, global_accumulative_recall, global_accumulative_precision,
                                      gt_flag, pred_flag, num_gt, tr, tp, k, fsc_k)
        #######then check for many-to-many case##########
        local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, \
        gt_flag, det_flag = many2many(local_sigma_table, local_tau_table,
                                         local_accumulative_recall, local_accumulative_precision,
                                         global_accumulative_recall, global_accumulative_precision,
                                         gt_flag, pred_flag, num_pred, tr, tp, k, fsc_k)

        try:
            local_precision = local_accumulative_precision / num_pred
        except ZeroDivisionError:
            local_precision = 0

        try:
            local_recall = local_accumulative_recall / num_gt
        except ZeroDivisionError:
            local_recall = 0

        # temp = '%s______/Precision:_%s_______/Recall:_%s\n' % (idx, str(local_precision), str(local_recall))
        # print("local:" + temp)
    try:
        recall = global_accumulative_recall / total_num_gt
    except ZeroDivisionError:
        recall = 0

    try:
        precision = global_accumulative_precision / total_num_pred
    except ZeroDivisionError:
        precision = 0

    try:
        f_score = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        f_score = 0

    hmean = 2 * precision * recall / (precision + recall)
    temp = ('Precision:_%s_______/Recall:_%s/Hmean:_%s\n' % (str(precision), str(recall), str(hmean)))
    print(temp)


if __name__ == '__main__':
    args = parse_args()
    if args.eval_f == 'ctw':
        if not args.search:
            hmean = evaluation(submit_file=args.submit_file, eval_ann=args.gt_file,
                               threshold=args.nms_thr, confidence=args.con_thr,
                               stastics_mode=args.stastics)
        else:
            best = 0
            best_conf = args.low_score
            for conf in np.arange(args.low_score, args.high_score, args.step):
                hmean = evaluation(submit_file=args.submit_file, eval_ann=args.gt_file,
                                   threshold=args.nms_thr, confidence=conf,
                                   stastics_mode=args.stastics)
                if hmean > best:
                    best = hmean
                    best_conf = conf
            print('conf {:.4f}: best-hmean:{:.6f}'.format(best_conf, best))

    elif args.eval_f == 'debug':
        name = evaluation(submit_file=args.submit_file, eval_ann=args.gt_file,
                          threshold=args.nms_thr, confidence=args.con_thr, debug_mode=True)
        print(name)
        det_evaluation(submit_file=args.submit_file, eval_ann=args.gt_file, iou_thr=args.nms_thr,
                       debug_mode=True, debug_name=name)

    else:
        eval_tool = det_evaluation(submit_file=args.submit_file, eval_ann=args.gt_file,
                                   iou_thr=args.nms_thr)





















