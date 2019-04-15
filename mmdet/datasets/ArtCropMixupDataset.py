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
from .custom_crop import CustomCropDataset
import json
import cv2
from collections import OrderedDict

""" a crop version """

def art_classes():
    return ['text']

label_ids = {name: i + 1 for i, name in enumerate(art_classes())}

debug_path = './visualization/debug/'


class ArtCropMixupDataset(CustomCropDataset):
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
    if is mixup mode, then randomly select another img during preparing traing img.
    """

    CLASSES = ('text')

    def __init__(self, mixup_ratio=0, *args, **kwargs):
        super(ArtCropMixupDataset, self).__init__(*args, **kwargs)
        assert 0 <= mixup_ratio < 1
        self.mixup_ratio = mixup_ratio

    def getBboxesAndLabels(self, height, width, annotations):
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
        if len(bboxes) > 0:
            bboxes = np.array(bboxes, dtype=np.float32)
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, width - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, height - 1)
            labels = np.array(labels, dtype=np.int64)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.array([], dtype=np.int64)

        if bboxes_ignore:
            bboxes_ignore = np.array(bboxes_ignore, dtype=np.float32)
            bboxes_ignore[:, 0::2] = np.clip(bboxes_ignore[:, 0::2], 0, width - 1)
            bboxes_ignore[:, 1::2] = np.clip(bboxes_ignore[:, 1::2], 0, height - 1)
            labels_ignore = np.array(labels_ignore, dtype=np.int64)
        else:
            bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
            labels_ignore = np.array([], dtype=np.int64)
        return bboxes, labels, bboxes_ignore, labels_ignore

    def load_annotations(self, ann_file):
        assert osp.isdir(self.img_prefix), 'Error: wrong path'
        assert osp.isfile(ann_file), 'Error: wrong ann file'
        detailed_annotation = None
        img_infos = []
        files = os.listdir(self.img_prefix)
        detailed_ann_file = ann_file.replace('_art_', '_detail_')
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
            if indx % 1000 == 0:
                print('{:d} % {:d}'.format(indx, len(files)))

        return img_infos

    def generate_masks_ann(self, height, width, annotations):
        gt_masks = []
        for i in range(len(annotations)):
            if annotations[i]["illegibility"]:
                continue
            polygon_points = np.asarray(annotations[i]["points"]).reshape(-1, 2).astype(np.int64)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask, [polygon_points], -1, 1, -1)
            gt_masks.append(mask)
        return gt_masks

    def generate_ignore_masks_ann(self, height, width, annotations):
        gt_ignore_masks = []
        for i in range(len(annotations)):
            if annotations[i]["illegibility"]:
                polygon_points = np.array(annotations[i]["points"]).reshape(-1, 2).astype(np.int64)
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.drawContours(mask, [polygon_points], -1, 1, -1)
                gt_ignore_masks.append(mask)
        return gt_ignore_masks

    def generate_mask(self, height, width, annotations):
        gt_masks = []
        gt_ignore_masks = []
        for i in range(len(annotations)):
            mask = np.zeros((height, width), dtype=np.uint8)
            polygon_points = np.asarray(annotations[i]['points']).reshape(-1, 2).astype(np.int64)
            cv2.drawContours(mask, [polygon_points], -1, 1, -1)
            if annotations[i]["illegibility"]:
                gt_ignore_masks.append(mask)
            else:
                gt_masks.append(mask)
        return gt_masks, gt_ignore_masks

    def debug(self, idx, ann):
        """ check whether the masks are fit to bboxes """
        save_path = debug_path
        if not osp.isdir(save_path):
            os.mkdir(save_path)
        masks = ann['masks']
        bboxes = ann['bboxes']
        assert len(bboxes) == len(masks)
        for i in range(len(bboxes)):
            bbox, mask = bboxes[i], masks[i]
            instance_show = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            instance_show[mask > 0] = (255, 255, 255)
            top_left, bottom_right = bbox[:2], bbox[2:]
            cv2.rectangle(instance_show, tuple(top_left), tuple(bottom_right), (0, 255, 0), 2)
            cv2.imwrite(osp.join(save_path, str(i) + '_' + str(idx) + '.jpg'), instance_show)

    def get_ann_info(self, idx):
        """ get bboxes and labels from annotations
            and generate gt_masks if with_mask is True
        """
        height = self.img_infos[idx]['height']
        width  = self.img_infos[idx]['width']
        bboxes, labels, bboxes_ignore, labels_ignore = \
            self.getBboxesAndLabels(height=height, width=width, annotations=self.img_infos[idx]['img_annotation'])
        ann = {
            'bboxes': bboxes,
            'labels': labels,
            'bboxes_ignore': bboxes_ignore,
            'labels_ignore': labels_ignore
        }
        if self.with_mask:
            # gt_masks = self.generate_masks_ann(height, width, self.img_infos[idx]['img_annotation'])
            # gt_ignore_masks = self.generate_ignore_masks_ann(height, width, self.img_infos[idx]['img_annotation'])
            gt_masks, gt_ignore_masks = self.generate_mask(height, width, self.img_infos[idx]['img_annotation'])
            ann['masks'] = gt_masks
            ann['ignore_masks'] = gt_ignore_masks
            assert len(gt_masks) == bboxes.shape[0]
        # self.debug(idx, ann)
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

    def prepare_single_train_img(self, idx):
        """ inherit from crop dataset"""
        img_info = self.img_infos[idx]
        assert osp.isfile(osp.join(self.img_prefix, img_info['filename']))
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        assert len(img.shape) == 3
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        # get ann here
        ann = self.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']
        if len(gt_bboxes) == 0:
            return None

        flip = bool(np.random.rand() < self.flip_ratio)
        img_scale = random_scale(self.img_scales, mode=self.resize_mode)

        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        img = img.copy()
        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
            proposals = np.hstack(
                [proposals, scores]) if scores is not None else proposals
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)
        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            gt_masks = self.mask_transform(ann['masks'], pad_shape,
                                           scale_factor, flip)
            gt_ignore_masks = ann['ignore_masks']
            if gt_ignore_masks:
                gt_ignore_masks = self.mask_transform(ann['ignore_masks'], pad_shape,
                                                      scale_factor, flip)
        # extra augmentation
        # different from the original ones.
        # there use mask to generate img, gt_bboxes and gt_labels.
        # extra_aug only supports random crop.
        if self.extra_aug is not None and self.with_mask:
            """ crop by masks, select gt_masks and gt_bboxes. 
                the crop size is the ori_shape/img_shape/pad_shape
                scale_factor use the scale_factor.
                assert the input image shape: [3, H, W] and the output image shape
                [3, H, W]
            """
            if self.with_crowd:
                img, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks, img_shape, pad_shape = \
                    self.extra_aug(img, gt_bboxes, gt_labels, gt_masks, gt_bboxes_ignore,
                                   gt_ignore_masks, img_shape, pad_shape)
            else:
                img, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks, img_shape, pad_shape = \
                    self.extra_aug(img, gt_bboxes, gt_labels, gt_masks, np.zeros((0, 4), dtype=np.float32),
                                   gt_ignore_masks, img_shape, pad_shape)
            img = img.copy()
            # self.debug_rc(img, gt_bboxes, gt_masks, img_info["filename"])
        if len(gt_bboxes) == 0:
            return None

        if not (self.extra_aug is not None and self.with_mask):
            ori_shape = (img_info['height'], img_info['width'], 3)
        else:
            ori_shape = img_shape

        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)

        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)))
        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)
        return data

    def prepare_train_img(self, idx):
        mixup = bool(self.mixup_ratio > 0 and np.random.rand() < self.mixup_ratio)
        train_data = self.prepare_single_train_img(idx)
        if not mixup:
            return train_data
        else:
            mix_idx = np.random.choice(np.delete(np.arange(len(self)), idx))
            mix_data = self.prepare_single_train_img(mix_idx)
            if train_data is None or mix_data is None:
                return None

            # can be further changed to beta distribution after.
            # delta = 0.5
            delta = np.float32(np.random.beta(1.5, 1.5))
            # first mix the two image.
            pad_h1, pad_w1, _ = train_data['img_meta'].data['pad_shape']
            pad_h2, pad_w2, _ = mix_data['img_meta'].data['pad_shape']
            img_h1, img_w1, _ = train_data['img_meta'].data['img_shape']
            img_h2, img_w2, _ = mix_data['img_meta'].data['img_shape']
            ori_shape = (max(train_data['img_meta'].data['ori_shape'][0], mix_data['img_meta'].data['ori_shape'][0]),
                         max(train_data['img_meta'].data['ori_shape'][1], mix_data['img_meta'].data['ori_shape'][1]),
                         3)
            img_shape = (max(img_h1, img_h2), max(img_w1, img_w2), 3)
            pad_shape = (max(pad_h1, pad_h2), max(pad_w1, pad_w2), 3)

            mix_image = np.zeros((3, pad_shape[0], pad_shape[1]), dtype=np.float32)
            mix_image[:, :img_h1, :img_w1] += train_data['img'].data.numpy()[:, :img_h1, :img_w1] * delta * 1.0
            mix_image[:, :img_h2, :img_w2] += mix_data['img'].data.numpy()[:, :img_h2, :img_w2] * (1 - delta) * 1.0

            mix_img_meta = dict(
                ori_shape=ori_shape,
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=train_data['img_meta'].data['scale_factor'],
                flip=train_data['img_meta'].data['flip'])

            mix_gt_bboxes = np.concatenate((
                train_data['gt_bboxes'].data.numpy(), mix_data['gt_bboxes'].data.numpy()),
                0)
            mixup_data = dict(
                img=DC(to_tensor(mix_image), stack=True),
                img_meta=DC(mix_img_meta, cpu_only=True),
                gt_bboxes=DC(to_tensor(mix_gt_bboxes)))
            if self.proposals is not None:
                proposals = np.concatenate((
                    train_data['proposals'].data.numpy(), mix_data['proposals'].data.numpy()),
                    0)
                mixup_data['proposals'] = DC(to_tensor(proposals))
            if self.with_label:
                # smooth or not smooth
                gt_labels = np.concatenate((
                    train_data['gt_labels'].data.numpy().astype(np.float32),
                    mix_data['gt_labels'].data.numpy().astype(np.float32)),
                    0)
                mixup_data['gt_labels'] = DC(to_tensor(gt_labels))
            if self.with_crowd:
                gt_bboxes_ignore = np.concatenate((
                    train_data['gt_bboxes_ignore'].data.numpy(), mix_data['gt_bboxes_ignore'].data.numpy()),
                    0)
                mixup_data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
            if self.with_mask:
                gt_masks = np.concatenate((
                    train_data['gt_masks'].data, mix_data['gt_masks'].data), 0)
                mixup_data['gt_masks'] = DC(gt_masks, cpu_only=True)

            mix_name = self.img_infos[idx]['filename'].split('.')[0] + '_' + self.img_infos[mix_idx]['filename'].split('.')[0]
            # self.debug_mixup(mix_image, mix_gt_bboxes, gt_masks, mix_name)
            return mixup_data

    def debug_mixup(self, mix_img, mix_bboxes, mix_masks, mix_name=None):
        debug_img = mix_img.copy().transpose(1, 2, 0)
        debug_img = (debug_img * self.img_norm_cfg['std'] + self.img_norm_cfg['mean']).astype(np.uint8)
        if not osp.exists(debug_path):
            os.makedirs(debug_path)
        assert len(mix_masks) == len(mix_bboxes)
        instance_mask = np.zeros((debug_img.shape[0], debug_img.shape[1], 3), dtype=np.uint8)
        for i in range(len(mix_bboxes)):
            bbox, mask = mix_bboxes[i], mix_masks[i]
            instance_mask[mask > 0] = (255, 255, 255)
            topL, bottomR = bbox[:2], bbox[2:]
            cv2.rectangle(instance_mask, tuple(topL), tuple(bottomR), (0, 255, 0), 2)
        instance_mask = np.concatenate((debug_img, instance_mask), axis=1)
        cv2.imwrite(debug_path + mix_name + '.jpg', instance_mask)

















