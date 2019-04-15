#
# @author:charlotte.Song
# @file: fcn_psp_mask_head.py
# @Date: 2019/3/31 20:33
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import mmcv
import pycocotools.mask as mask_util
import torch.nn as nn
import torch.nn.functional as F

from ..registry import HEADS
from ..utils import ConvModule
from mmdet.core import mask_cross_entropy, mask_target
""" implement a psp-like mask head 
    using 1x1, 2x2, 4x4, 8x8, 16x16 to concat, and use the concatenated feature map 
     to produce final prob map.
"""

@HEADS.register_module
class FCNPspMaskHead(nn.Module):
    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 pool_scales=(1, 2, 4, 8),
                 upsample_method='deconv',
                 upsample_ratio=2,
                 num_classes=81,
                 class_agnostic=False,
                 normalize=None):
        super(FCNPspMaskHead, self).__init__()
        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear"'.format(upsample_method))
        self.num_convs = num_convs
        self.roi_feat_size = roi_feat_size  # WARN: not used and reserved
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.normalize = normalize
        self.with_bias = normalize is None
        self.pool_scales = pool_scales

        self.pool_convs = nn.ModuleList()       # the four convs after concat.
        self.convs = nn.ModuleList()
        for i, scale in enumerate(self.pool_scales):
            out_channels_t = int(self.in_channels // len(self.pool_scales))
            padding = (self.conv_kernel_size - 1) // 2
            self.pool_convs.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    ConvModule(
                        self.in_channels,
                        out_channels_t,
                        self.conv_kernel_size,
                        padding=padding,
                        normalize=normalize,
                        bias=self.with_bias)))

        for i in range(self.num_convs):
            in_channels = 2 * self.in_channels if i == 0 else self.conv_out_channels
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    normalize=normalize,
                    bias=self.with_bias))

        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            self.upsample = nn.ConvTranspose2d(
                self.conv_out_channels,
                self.conv_out_channels,
                self.upsample_ratio,
                stride=self.upsample_ratio)
        else:
            self.upsample = nn.Upsample(
                scale_factor=self.upsample_ratio, mode=self.upsample_method)

        out_channels = 1 if self.class_agnostic else self.num_classes
        self.conv_logits = nn.Conv2d(self.conv_out_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

    def init_weights(self):
        for m in [self.upsample, self.conv_logits]:
            if m is None:
                continue
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)


    def forward(self, x):
        """ first store the size, """
        h, w = x.size()[-2:]
        # first split
        ppm = [conv(x) for conv in self.pool_convs]
        # rescale to ori size
        ppm = [
            F.interpolate(p, size=(h, w), mode='nearest')
            for p in ppm
        ]
        # concat.
        # x = ppm[0]
        for i in range(0, len(ppm)):
            x = torch.cat((x, ppm[i]), 1)

        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_logits(x)
        return mask_pred

    def get_target(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets

    def loss(self, mask_pred, mask_targets, labels):
        loss = dict()
        if self.class_agnostic:
            loss_mask = mask_cross_entropy(mask_pred, mask_targets,
                                           torch.zeros_like(labels))
        else:
            loss_mask = mask_cross_entropy(mask_pred, mask_targets, labels)
        loss['loss_mask'] = loss_mask
        return loss

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class+1, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid().cpu().numpy()
        assert isinstance(mask_pred, np.ndarray)

        cls_segms = [[] for _ in range(self.num_classes - 1)]
        bboxes = det_bboxes.cpu().numpy()[:, :4]
        labels = det_labels.cpu().numpy() + 1

        if rescale:     # keep the img size as the ori size.
            img_h, img_w = ori_shape[:2]
        else:
            # if not rescale, that means the output bboxes fit to
            # the size of input images.
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        for i in range(bboxes.shape[0]):
            # the bbox are are fit to the rescaled image.
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)

            if not self.class_agnostic:
                mask_pred_ = mask_pred[i, label, :, :]
            else:
                mask_pred_ = mask_pred[i, 0, :, :]
            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)

            bbox_mask = mmcv.imresize(mask_pred_, (w, h))   # expand bbox before.
            bbox_mask = (bbox_mask > rcnn_test_cfg.mask_thr_binary).astype(
                np.uint8)
            im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            cls_segms[label - 1].append(rle)

        return cls_segms



