#
# @author:charlotte.Song
# @file: maskScoring_fcn_pan_head.py
# @Date: 2019/4/14 14:40
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import mmcv
import pycocotools.mask as mask_util
from ..utils import ConvModule
from ..registry import HEADS
from mmdet.core import mask_cross_entropy, mask_target, smooth_l1_loss
""" an implementation of Mask Scoring and Path Aggregation Network 
    using MaskScoring as an assit loss, directly predict the IoU between the gt.
    using conv and fcs directly from conv3.
"""

@HEADS.register_module
class MSFCNPAN(nn.Module):
    def __init__(self,
                 num_convs=4,
                 roi_feats_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 upsample_method='deconv',
                 upsamle_ratio=2,
                 num_classes=81,
                 class_agnostic=False,
                 normalize=None,
                 num_fc_convs=2,
                 ms_fc_channels=512):
        super(MSFCNPAN, self).__init__()
        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear"'.format(upsample_method))
        self.num_convs = num_convs
        self.roi_feat_size = roi_feats_size
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = upsample_method
        self.upsample_ratio = upsamle_ratio
        self.num_classes = num_classes
        self.num_fc_convs = num_fc_convs
        self.ms_fc_channels = ms_fc_channels
        self.class_agnostic = class_agnostic
        self.normalize = normalize
        self.with_bias = normalize is None

        self.convs = nn.ModuleList()
        # standard branch
        for i in range(self.num_convs):
            in_channels = (self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    normalize=normalize,
                    bias=self.with_bias))
        self.fc_convs = nn.ModuleList()
        # share conv with classification and MaskIoU fcs.
        for i in range(self.num_fc_convs):
            out_channels_t = (self.conv_out_channels if i != self.num_fc_convs - 1 else
                              self.conv_out_channels // 2)
            padding = (self.conv_kernel_size - 1) // 2
            self.fc_convs.append(
                ConvModule(
                    self.conv_out_channels,
                    out_channels_t,
                    self.conv_kernel_size,
                    padding=padding,
                    normalize=normalize,
                    bias=self.with_bias))
        fc_inchannels = self.conv_out_channels // 2
        fc_inchannels *= (self.roi_feat_size * self.roi_feat_size)
        fc_outchannels = (self.roi_feat_size * self.roi_feat_size)
        self.mask_fc = nn.Linear(fc_inchannels, fc_outchannels)
        self.ms_fc0 = nn.Linear(fc_inchannels, self.ms_fc_channels)
        self.ms_fc1 = nn.Linear(self.ms_fc_channels, num_classes)

        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            self.upsample = nn.ConvTranspose2d(     # not relu.
                self.conv_out_channels,
                self.conv_out_channels,
                self.upsample_ratio,
                stride=self.upsample_ratio)
        else:
            self.upsample = nn.Upsample(
                scale_factor=self.upsample_ratio, mode=self.upsample_method)

        # out_channels == num_classes + (background)
        # and the output is the output used to sigmoid.
        out_channels = 1 if self.class_agnostic else self.num_classes
        self.conv_logits = nn.Conv2d(self.conv_out_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

    def init_weights(self):
        for m in [self.upsample_method, self.conv_logits]:
            if m is None:
                continue
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        for m in [self.mask_fc, self.ms_fc0, self.ms_fc1]:
            nn.init.kaiming_normal_(
                m.weigh, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = x
        for idx, conv in enumerate(self.convs):
            x = conv(x)
            if idx == len(self.convs) - 2:
                y = x
                for conv_ in self.fc_convs:
                    y = conv_(y)
        y = y.view(y.size(0), -1)
        y_mask_fc = self.mask_fc(y).view(y.size(0), 1, x.size(2), x.size(3))
        y_mask_fc = y_mask_fc.repeat(1, self.conv_out_channels, 1, 1)
        y_ms_fc = self.ms_fc1(self.ms_fc0(y)) # [N, 2]

        x = y_mask_fc + x
        if self.upsample_method is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_logits(x)
        return mask_pred, y_ms_fc

    def get_target(self, sampling_result, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_result]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_result
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                  gt_masks, rcnn_train_cfg)
        return mask_targets

    def mask_iou(self, pred, target, label):
        num_rois = pred.size()[0]
        inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
        pred_slice = pred[inds, label].squeeze(1).detach()
        pred_slice = torch.sigmoid(pred_slice).contiguou().view(num_rois, -1)
        target = target.contiguous().view(num_rois, -1)

        intersection = torch.sum((pred_slice > 0.5) * (target > 0.5), dim=1)
        union = torch.sum((pred_slice > 0.5) | (target > 0.5), dim=1) + 1e-5
        iou_target = intersection * 1.0 / union
        return iou_target

    # def get_ms_target(self, mask_pred, mask_targets, labels):
    #     if self.class_agnostic:
    #         iou_targets = self.mask_iou(mask_pred, mask_targets,
    #                                     torch.zeros_like(labels))
    #     else:
    #         iou_targets = self.mask_iou(mask_pred, mask_targets, labels)
    #     return iou_targets
    def get_ms_target(self, sampling_results, gt_masks, labels, rcnn_train_cfg,
                      mask_pred):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds, gt_masks,
                                   rcnn_train_cfg)
        if self.class_agnostic:
            iou_targets = self.mask_iou(mask_pred, mask_targets, torch.zeros_like(labels))
        else:
            iou_targets = self.mask_iou(mask_pred, mask_targets, labels)
        return mask_targets, iou_targets

    def loss(self, mask_pred, mask_targets, iou_pred, iou_targets, labels):
        loss = dict()
        if self.class_agnostic:
            loss_mask = mask_cross_entropy(mask_pred, mask_targets,
                                           torch.zeros_like(labels))
            iou_targets = self.mask_iou(mask_pred, mask_targets,
                                        torch.zeros_like(labels))
            # use smooth L1 loss instead of L2 loss.
            # iou_loss = smooth_l1_loss(iou_pred, iou_targets, reduction='mean')
        else:
            loss_mask = mask_cross_entropy(mask_pred, mask_targets, labels)
            iou_targets = self.mask_iou(mask_pred, mask_targets, labels)
            # iou_loss = smooth_l1_loss()
        loss_iou = smooth_l1_loss(iou_pred, iou_targets, labels)
        loss['loss_mask'] = loss_mask
        loss['loss_maskiou'] = loss_iou
        return loss

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid().cpu().numpy()
        assert isinstance(mask_pred, np.ndarray)

        cls_segms = [[] for _ in range(self.num_classes - 1)]
        bboxes = det_bboxes.cpu().numpy()[:, :4]
        labels = det_labels.cpu().numpy() + 1

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            # if not rescale, that means the output bboxes fit to
            # the size of input images.
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        for i in range(bboxes.shape[0]):
            #
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)

            if not self.class_agnostic:
                mask_pred_ = mask_pred[i, label, :, :]
            else:
                mask_pred_ = mask_pred[i, 0, :, :]
            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)

            bbox_mask = mmcv.imresize(mask_pred_, (w, h))
            bbox_mask = (bbox_mask > rcnn_test_cfg.mask_thr_binary).astype(
                np.uint8)
            im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            cls_segms[label - 1].append(rle)

        return cls_segms






















