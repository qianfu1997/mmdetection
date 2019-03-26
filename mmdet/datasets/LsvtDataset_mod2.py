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
# change the box generation and mask generation from
# load annotation to get ann info


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
                'masks':
            }
        },
        ...
    ]

    The `ann` field is optional for testing.

    change the polygon points to
    """

    CLASSES = ('text')

    def getBboxesAndLabels(self, annotations):
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        # points_lists = [] # does not contain the ignored polygons.
        # avoid to copy object.
        for i in range(len(annotations)):
            points = np.asarray(annotations[i]["points"]).reshape(-1, 2).astype(np.int64)
            box = np.zeros(4)
            box[:2] = np.min(points, axis=0)
            box[2:] = np.max(points, axis=0)
            label = label_ids['text']
            if annotations[i]["illegibility"]:
                bboxes_ignore.append(box)
                labels_ignore.append(label)
            else:
                bboxes.append(box)
                labels.append(label)
                # do not store the ignored polygons.
        if bboxes:
            bboxes = np.array(bboxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.array([], dtype=np.int64)

        if bboxes_ignore:
            bboxes_ignore = np.array(bboxes_ignore, dtype=np.float32)
            labels_ignore = np.array(labels_ignore, dtype=np.int64)
        else:
            bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
            labels_ignore = np.array([], dtype=np.int64)
        return bboxes, labels, bboxes_ignore, labels_ignore

    # def load_annotations(self, ann_file):
    #     assert osp.isdir(self.img_prefix), 'Error: wrong img_prefix {}'.format(self.img_prefix)
    #     assert osp.isfile(ann_file), 'Error: wrong ann_file {}'.format(ann_file)
    #     detailed_annotation = None
    #     img_infos = []
    #     files = os.listdir(self.img_prefix)
    #     with open(ann_file, 'r', encoding='utf-8_sig') as f:
    #         gt_annotations = json.loads(f.read(), object_pairs_hook=OrderedDict)
    #     detailed_ann_file = ann_file.replace('_full_', '_detail_')
    #     if osp.isfile(detailed_ann_file):
    #         with open(detailed_ann_file, 'r', encoding='utf-8_sig') as f:
    #             detailed_annotation = json.loads(f.read(), object_pairs_hook=OrderedDict)
    #     for index in range(len(files)):
    #         name = osp.splitext(files[index])[0]
    #         if detailed_annotation is None:
    #             img = cv2.imread(osp.join(self.img_prefix, files[index]))
    #             height, width = img.shape[:2]
    #         else:
    #             height, width = detailed_annotation[name]['height'], detailed_annotation[name]['width']
    #         annotations = gt_annotations[name]
    #         bboxes, labels, bboxes_ignore, labels_ignore, new_anns = \
    #             self.getBboxesAndLabels(annotations)
    #         info = {
    #             'filename': files[index],
    #             'height': height,
    #             'width': width,
    #             'ann': {
    #                 'bboxes': bboxes,
    #                 'labels': labels,
    #                 'bboxes_ignore': bboxes_ignore,
    #                 'labels_ignore': labels_ignore,
    #                 'annotations': new_anns,
    #             } # ann
    #         } # end
    #         img_infos.append(info)
    #         if index % 1000 == 0 or index % 500 == 0:
    #             print('{:d} % {:d}'.format(index, len(files)))
    #     return img_infos
    def load_annotations(self, ann_file):
        assert osp.isdir(self.img_prefix), 'Error: wrong path'
        assert osp.isfile(ann_file), 'Error: wrong ann file'
        detailed_annotation = None
        img_infos = []
        files = os.listdir(self.img_prefix)
        detailed_ann_file = ann_file.replace('_full_', '_detail_')
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
                'height': height,
                'width': width,
                'img_annotation': gt_annotations[name]
            }# info
            img_infos.append(info)
            if indx % 1000 == 0 or indx % 500 == 0:
                print('{:d} % {:d}'.format(indx, len(files)))
        return img_infos

    def generate_masks(self, height, width, polygon_points):
        gt_masks = []
        for polygon in polygon_points:
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask, [polygon], -1, 1, -1)
            gt_masks.append(mask)
        return np.array(gt_masks)

    def generate_masks_ann(self, height, width, annotations):
        gt_masks = []
        for i in range(len(annotations)):
            polygon_points = np.asarray(annotations[i]["points"]).reshape(-1, 2).astype(np.int64)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask, [polygon_points], -1, 1, -1)
            gt_masks.append(mask)
        return np.array(gt_masks)

    # def generate_masks_show(self, img, height, width, polygon_points):
    #     show_masks = np.zeros((height, width, 3), dtype=np.uint8)
    #     for polygon in polygon_points:
    #         cv2.drawContours(show_masks, [polygon], -1, (255, 255, 255), -1)
    #         cv2.drawContours(img, [polygon], -1, (0, 0, 255), 2)
    #     return show_masks, img
    # def get_ann_info(self, idx):
    #     ann_info = self.img_infos[idx]['ann']
    #     height = self.img_infos[idx]['height']
    #     width = self.img_infos[idx]['width']
    #     # polygon_points = ann_info['polygon_points']
    #     annotations = self.img_infos[idx]['ann']['annotations']
    #     if self.with_mask:
    #         gt_masks = self.generate_masks(height, width, annotations)
    #         ann_info['masks'] = gt_masks
    #     # filename = self.img_infos[idx]['filename']
    #     # img_show = cv2.imread(osp.join(self.img_prefix, filename))
    #     # masks, img = self.generate_masks_show(img_show, height, width, polygon_points)
    #     # cv2.imwrite('/home/data1/sxg/IC19/mmdetection-master/visualization/' + filename, masks)
    #     return ann_info

    # def get_ann_info(self, idx):
    #     ann_info = self.img_infos[idx]['ann']
    #     height = self.img_infos[idx]['height']
    #     width = self.img_infos[idx]['width']
    #     annotations = ann_info['annotations']
    #     if self.with_mask:
    #         gt_masks = self.generate_masks_ann(height, width, annotations)
    #         ann_info['masks'] = gt_masks
    #     return ann_info

    def get_ann_info(self, idx):
        """ get bboxes and labels from annotations
            and generate gt_masks if with_mask is True
        """
        filename = self.img_infos[idx]['filename']
        height = self.img_infos[idx]['height']
        width  = self.img_infos[idx]['width']
        bboxes, labels, bboxes_ignore, labels_ignore = \
            self.getBboxesAndLabels(self.img_infos[idx]['img_annotation'])
        ann = {
            'bboxes': bboxes,
            'labels': labels,
            'bboxes_ignore': bboxes_ignore,
            'labels_ignore': labels_ignore
        }
        if self.with_mask:
            gt_masks = self.generate_masks_ann(height, width, self.img_infos[idx]['img_annotation'])
            ann['masks'] = gt_masks
        return ann

    """ for aug test """
    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)
            add filename to _img_meta.
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
            # _img_meta = dict(
            #     ori_shape=(img_info['height'], img_info['width'], 3),
            #     img_shape=img_shape,
            #     pad_shape=pad_shape,
            #     scale_factor=scale_factor,
            #     flip=flip)
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip,
                filename=img_info['filename'])

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
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        data = dict(img=imgs, img_meta=img_metas)
        if self.proposals is not None:
            data['proposals'] = proposals
        return data
















