import os.path as osp
import os
import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         Numpy2Tensor)
from .utils import to_tensor, random_scale
from .extra_aug import ExtraAugmentation
from .custom import CustomDataset
import json
import cv2
from collections import OrderedDict

# first use the custom dataset to train.
#


def lvst_classes():
    return ['text']


label_ids = {name: i + 1 for i, name in enumerate(lvst_classes())}


class LsvtDataset(CustomDataset):
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
                'polygon_points': <list<np.ndarray>>
            }
        },
        ...
    ]

    The `ann` field is optional for testing.

    change the polygon points to
    """

    CLASSES = ('text')

    def getBboxesAndLabels(self, height, width, annotations):
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        # points_lists = [] # does not contain the ignored polygons.
        for annotation in annotations:
            points = np.array(annotation["points"])
            x, y, w, h = cv2.boundingRect(points)
            box = np.array([x, y, x + w, y + h])
            label = label_ids['text']
            if annotation["transcription"] == "###":
                bboxes_ignore.append(box)
                labels_ignore.append(label)
            else:
                bboxes.append(box)
                labels.append(label)
                # points_lists.append(points)
        if bboxes:
            bboxes = np.array(bboxes, dtype=np.float32)
            # filter the coordinates that overlap the image boundaries.
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, width - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, height - 1)
            labels = np.array(labels, dtype=np.int64)
            # nothing to do with points_lists
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.array([], dtype=np.int64)

        if bboxes_ignore:
            bboxes_ignore = np.array(bboxes_ignore, dtype=np.float32)
            # filter the coordinates that overlap the image boundaries
            bboxes_ignore[:, 0::2] = np.clip(bboxes_ignore[:, 0::2], 0, width - 1)
            bboxes_ignore[:, 1::2] = np.clip(bboxes_ignore[:, 1::2], 0, height - 1)
            labels_ignore = np.array(labels_ignore, dtype=np.int64)
        else:
            bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
            labels_ignore = np.array([], dtype=np.int64)
        return bboxes, labels, bboxes_ignore, labels_ignore, # polygon_points

    def load_annotations(self, ann_file):
        assert osp.isdir(self.img_prefix), 'Error: wrong img_prefix {}'.format(self.img_prefix)
        assert osp.isfile(ann_file), 'Error: wrong ann_file {}'.format(ann_file)
        detailed_annotation = None
        img_infos = []
        files = os.listdir(self.img_prefix)
        with open(ann_file, 'r', encoding='utf-8') as f:
            gt_annotations = json.loads(f.read(), object_pairs_hook=OrderedDict)
        detailed_ann_file = ann_file.replace('_full_', '_detail_')
        if osp.isfile(detailed_ann_file):
            with open(detailed_ann_file, 'r', encoding='utf-8') as f:
                detailed_annotation = json.loads(f.read(), object_pairs_hook=OrderedDict)
        for index, file in enumerate(files):
            name = osp.splitext(file)[0]
            if detailed_annotation is None:
                img = cv2.imread(osp.join(self.img_prefix, file))
                height, width = img.shape[:2]
            else:
                height, width = detailed_annotation[name]['height'], detailed_annotation[name]['width']
            annotations = gt_annotations[name]
            bboxes, labels, bboxes_ignore, labels_ignore, polygon_points = \
                self.getBboxesAndLabels(height, width, annotations)
            info = {
                'filename': file,
                'height': height,
                'width': width,
                'ann': {
                    'bboxes': bboxes,
                    'labels': labels,
                    'bboxes_ignore': bboxes_ignore,
                    'labels_ignore': labels_ignore,
                    'polygon_points': polygon_points,
                } # ann
            } # end
            img_infos.append(info)
            if index % 1000 == 0 or index % 500 == 0:
                print('{:d} % {:d}'.format(index, len(files)))
        return img_infos

    def generate_masks(self, height, width, polygon_points):
        gt_masks = []
        for polygon in polygon_points:
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask, [polygon], -1, 1, -1)
            gt_masks.append(mask)
        return np.array(gt_masks)

    # def generate_masks_show(self, img, height, width, polygon_points):
    #     show_masks = np.zeros((height, width, 3), dtype=np.uint8)
    #     for polygon in polygon_points:
    #         cv2.drawContours(show_masks, [polygon], -1, (255, 255, 255), -1)
    #         cv2.drawContours(img, [polygon], -1, (0, 0, 255), 2)
    #     return show_masks, img

    def get_ann_info(self, idx):
        ann_info = self.img_infos[idx]['ann']
        height = self.img_infos[idx]['height']
        width = self.img_infos[idx]['width']
        polygon_points = ann_info['polygon_points']
        if self.with_mask:
            gt_masks = self.generate_masks(height, width, polygon_points)
            ann_info['masks'] = gt_masks
        # filename = self.img_infos[idx]['filename']
        # img_show = cv2.imread(osp.join(self.img_prefix, filename))
        # masks, img = self.generate_masks_show(img_show, height, width, polygon_points)
        # cv2.imwrite('/home/data1/sxg/IC19/mmdetection-master/visualization/' + filename, masks)
        return ann_info
















