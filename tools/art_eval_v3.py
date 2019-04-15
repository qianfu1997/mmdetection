import argparse
import copy
import json
import os
from collections import defaultdict
from collections import OrderedDict
import os.path as osp
import Polygon as plg

import numpy as np
from PIL import Image

art_eval_ann_file = 'data/ArT/annotations/sp_val_eval_labels.json'
art_submit_file = 'submit/art/submit.json'

def fast_hist(gt, prediction, n):
    k = (gt >= 0) & (gt < n)
    return np.bincount(
        n * gt[k].astype(int) + prediction[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    ious = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    ious[np.isnan(ious)] = 0
    return ious


def find_all_png(folder):
    paths = []
    for root, dirs, files in os.walk(folder, topdown=True):
        paths.extend([osp.join(root, f)
                      for f in files if osp.splitext(f)[1] == '.png'])
    return paths

def evaluate_segmentation(gt_dir, result_dir, num_classes, key_length):
    gt_dict = dict([(osp.split(p)[1][:key_length], p)
                    for p in find_all_png(gt_dir)])
    result_dict = dict([(osp.split(p)[1][:key_length], p)
                        for p in find_all_png(result_dir)])
    result_gt_keys = set(gt_dict.keys()) & set(result_dict.keys())
    if len(result_gt_keys) != len(gt_dict):
        raise ValueError('Result folder only has {} of {} ground truth files.'
                         .format(len(result_gt_keys), len(gt_dict)))
    print('Found', len(result_dict), 'results')
    print('Evaluating', len(gt_dict), 'results')
    hist = np.zeros((num_classes, num_classes))
    i = 0
    gt_id_set = set()
    for key in sorted(gt_dict.keys()):
        gt_path = gt_dict[key]
        result_path = result_dict[key]
        gt = np.asarray(Image.open(gt_path, 'r'))
        gt_id_set.update(np.unique(gt).tolist())
        prediction = np.asanyarray(Image.open(result_path, 'r'))
        hist += fast_hist(gt.flatten(), prediction.flatten(), num_classes)
        i += 1
        if i % 100 == 0:
            print('Finished', i, per_class_iu(hist) * 100)
    gt_id_set.remove(255)
    print('GT id set', gt_id_set)
    ious = per_class_iu(hist) * 100
    miou = np.mean(ious[list(gt_id_set)])
    return miou, list(ious)

def evaluate_drivable(gt_dir, result_dir):
    return evaluate_segmentation(gt_dir, result_dir, 3, 17)

def get_ap(recalls, precisions):
    # correct AP calculation
    # first append sentinel values at the end
    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))

    # compute the precision envelope
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(recalls[1:] != recalls[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
    return ap

def group_by_key(detections, key):
    groups = defaultdict(list)
    for d in detections:
        groups[d[key]].append(d)
    return groups

def get_union(pa, pb):
    pa_area = pa.area()
    pb_area = pb.area()
    # return pa_area + pb_area - get_intersection(pa, pb)
    # return pa_area + pb_area - get_intersection()
    return pa_area + pb_area - get_intersection(pa, pb)

def get_intersection(pa, pb):
    pInt = pa & pb
    if len(pInt) == 0:
        return 0
    else:
        return pInt.area()

def cat_best_hmean(gt, predictions, thresholds):
    """
    Implementation refers to https://github.com/rbgirshick/py-faster-rcnn
    """
    num_gts = len([g for g in gt if g['ingore'] == False])
    image_gts = group_by_key(gt, 'name')
    image_gt_boxes = {k: np.array([b['bbox'] for b in boxes]) 
                      for k, boxes in image_gts.items()}
    image_gt_ignored = {k: np.array([b['ingore'] for b in boxes])
                        for k, boxes in image_gts.items()}
    image_gt_checked = {k: np.zeros((len(boxes), len(thresholds)))
                        for k, boxes in image_gts.items()}
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)

    # go down dets and mark TPs and FPs
    nd = len(predictions)
    tp = np.zeros((nd, len(thresholds)))
    fp = np.zeros((nd, len(thresholds)))
    for i, p in enumerate(predictions):
        pred_polygon = plg.Polygon(np.array(p['bbox']))
        ovmax = -np.inf
        jmax = -1
        try:
            gt_boxes = image_gt_boxes[p['name']]
            gt_ignored = image_gt_ignored[p['name']]
            gt_checked = image_gt_checked[p['name']]
        except KeyError:
            gt_boxes = []
            gt_checked = None

        if len(gt_boxes) > 0:
            ovmax = 0
            jmax = 0
            for j, gt_box in enumerate(gt_boxes):
                gt_polygon = plg.Polygon(np.array(gt_box))
                union = get_union(pred_polygon, gt_polygon)
                inter = get_intersection(pred_polygon, gt_polygon)
                overlap = inter / union
                if overlap > ovmax:
                    ovmax = overlap
                    jmax = j

        for t, threshold in enumerate(thresholds):
            if ovmax > threshold:
                if gt_checked[jmax, t] == 0:
                    if gt_ignored[jmax]:
                        tp[i, t] = 0.
                    else:
                        tp[i, t] = 1.
                    gt_checked[jmax, t] = 1
                else:
                    fp[i, t] = 1.
            else:
                fp[i, t] = 1.

    # compute precision recall
    fp = np.cumsum(fp, axis=0)
    tp = np.cumsum(tp, axis=0)
    recalls = tp / float(num_gts)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    hmeans = 2 * precisions * recalls / (precisions + recalls)    
    best_i = np.argmax(hmeans)
    print('p: {:f}  r: {:f}  h: {:f}, best_score_th: {:f}'.format(
          float(precisions[best_i]), float(recalls[best_i]), float(hmeans[best_i]), predictions[best_i]['score']))

    # ap = np.zeros(len(thresholds))
    # for t in range(len(thresholds)):
    #     ap[t] = get_ap(recalls[:, t], precisions[:, t])

def trans_json_ic19_to_bdd100k(ic19, is_gt):
    bdd100k = []
    for key in ic19:
        ic19_i = ic19[key]
        for j in range(len(ic19_i)):
            bdd_ij = {
                'category': 'text',
                'timestamp': 1000,
                'name': key,
                'bbox': ic19_i[j]['points'],
            }
            if is_gt:
                bdd_ij['ingore'] = ic19_i[j]['illegibility']
                bdd_ij['score'] = 1
            else:
                bdd_ij['score'] = ic19_i[j]['confidence']
            bdd100k.append(bdd_ij)
    return bdd100k

def evaluate_detection(gt_path, result_path):
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt = json.loads(f.read(), object_pairs_hook=OrderedDict)
    gt = trans_json_ic19_to_bdd100k(gt, is_gt=True)
    with open(result_path, 'r', encoding='utf-8') as f:
        pred = json.loads(f.read(), object_pairs_hook=OrderedDict)
    pred = trans_json_ic19_to_bdd100k(pred, is_gt=False)
    
    cat_gt = group_by_key(gt, 'category')
    cat_pred = group_by_key(pred, 'category')
    cat_list = ['text']
    thresholds = [0.5]
    aps = np.zeros((len(thresholds), len(cat_list)))
    cat_best_hmean(cat_gt['text'], cat_pred['text'], thresholds)


def main():
    evaluate_detection(art_eval_ann_file, art_submit_file)

if __name__ == '__main__':
    main()