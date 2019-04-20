import os.path as osp
import os
import cv2

import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         Numpy2Tensor)
from .utils import to_tensor, random_scale
from .extra_aug import ExtraAugmentation, ExtraAugmentationIC
""" a version with crop. 
    add support IC17.
"""
debug_path = '/data/sxg_workspace/mmdetection-master/visualization/debug/'
# debug_path = '/home/xieenze/sxg_workspace/mmdetection-master/visualization/debug/'


class CustomCropDataset(Dataset):
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
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    CLASSES = None

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=True,
                 with_crowd=True,
                 with_label=True,
                 extra_aug=None,            # here to add random crop for art.
                 resize_keep_ratio=True,
                 test_mode=False):
        # prefix of images path
        self.img_prefix = img_prefix

        # load annotations (and proposals)
        # need to be implement.
        # in test mode or not
        self.test_mode = test_mode
        self.img_infos = self.load_annotations(ann_file)
        if proposal_file is not None:
            self.proposals = self.load_proposals(proposal_file)
        else:
            self.proposals = None
        # filter images with no annotation during training
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]

        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # normalization configs
        self.img_norm_cfg = img_norm_cfg

        # max proposals per image
        self.num_max_proposals = num_max_proposals
        # flip ratio
        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor

        # with mask or not (reserved field, takes no effect)
        self.with_mask = with_mask
        # some datasets provide bbox annotations as ignore/crowd/difficult,
        # if `with_crowd` is True, then these info is returned.
        self.with_crowd = with_crowd
        # with label is False for RPN
        self.with_label = with_label

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # transforms
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.bbox_transform = BboxTransform()
        self.mask_transform = MaskTransform()
        self.numpy2tensor = Numpy2Tensor()

        # if use extra augmentation
        if extra_aug is not None:
            self.extra_aug = ExtraAugmentationIC(**extra_aug)
        else:
            self.extra_aug = None

        # image rescale if keep ratio
        self.resize_keep_ratio = resize_keep_ratio

        # random scale mode, if img scales > 2 choose value, else range
        self.resize_mode = 'value' if len(self.img_scales) > 2 else 'range'
        # self.resize_mode = 'range'

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, ann_file):
        return mmcv.load(ann_file)

    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):

        img_info = self.img_infos[idx]
        # load image
        assert osp.isfile(osp.join(self.img_prefix, img_info['filename']))
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        assert len(img.shape) == 3
        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
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
        # get ann here.
        ann = self.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        # extra augmentation for photometric
        if self.extra_aug is not None:
            img = self.extra_aug.first_transform(img)
        img = img.copy()

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        img_scale = random_scale(self.img_scales, mode=self.resize_mode)
        # img_scale = random_scale(self.img_scales, mode='value')  # sample a scale
        # here to select a rescale size, and pad the image.
        # scale_factor is used to guide the transformation of bboxes and masks.
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
            if len(gt_ignore_masks) > 1:
                gt_ignore_masks = self.mask_transform(gt_ignore_masks, pad_shape,
                                                      scale_factor, flip)
            else:
                gt_ignore_masks = np.array([], dtype=np.uint8)
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
                    self.extra_aug(img=img, boxes=gt_bboxes, labels=gt_labels, masks=gt_masks, ignore_bboxes=gt_bboxes_ignore, ignore_masks=gt_ignore_masks,
                                   img_shape=img_shape, pad_shape=pad_shape)

            else:
                img, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks, img_shape, pad_shape = \
                    self.extra_aug(img=img, boxes=gt_bboxes, labels=gt_labels, masks=gt_masks, ignore_bboxes=np.zeros((0, 4), dtype=np.float32), ignore_masks=gt_ignore_masks,
                                   img_shape=img_shape, pad_shape=pad_shape)
            img = img.copy()
            # self.debug_rc(img, gt_bboxes, gt_masks, img_info["filename"])

        if len(gt_bboxes) == 0:
            return None
        # the ori_shape will be changed to the crop size.
        # ori_shape = (img_info['height'], img_info['width'], 3)
        # if self.extra_aug is None:
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

    def debug_rc(self, img, gt_bboxes, gt_masks, img_name):
        """ a debug function for random crop """
        # identifier = 'art' if 'art' in self.img_prefix else 'lsvt'
        identifier = 'unknown'
        for t in ['art', 'LSVT', 'IC17', 'IC19']:
            if t in self.img_prefix:
                identifier = t
        save_path = osp.join(debug_path, identifier)
        debug_img = img.copy()
        debug_img = debug_img.transpose(1, 2, 0)
        debug_img = (debug_img * self.img_norm_cfg['std'] + self.img_norm_cfg['mean']).astype(np.uint8)
        if not osp.exists(save_path):
            os.makedirs(save_path)
        assert len(gt_masks) == len(gt_bboxes)
        instance_mask = np.zeros((debug_img.shape[0], debug_img.shape[1], 3), dtype=np.uint8)
        for i in range(len(gt_bboxes)):
            bbox, mask = gt_bboxes[i], gt_masks[i]
            instance_mask[mask > 0] = (255, 255, 255)
            topL, bottomR = bbox[:2], bbox[2:]
            cv2.rectangle(instance_mask, tuple(topL), tuple(bottomR), (0, 255, 0), 2)
        instance_mask = np.concatenate((debug_img, instance_mask), axis=1)
        # cv2.imwrite(save_path + img_name + '.jpg', instance_mask)
        cv2.imwrite(osp.join(save_path, img_name + '.jpg'), instance_mask)

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
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
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                # if flip_ratio, add an rotation img.
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        data = dict(img=imgs, img_meta=img_metas)
        if self.proposals is not None:
            data['proposals'] = proposals
        return data
