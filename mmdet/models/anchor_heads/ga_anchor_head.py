#
# @author:charlotte.Song
# @file: ga_anchor_head.py
# @Date: 2019/3/26 22:28
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import (AnchorGenerator, anchor_target, delta2bbox,
                        multi_apply, weighted_cross_entropy, weighted_smoothl1,
                        weighted_binary_cross_entropy,
                        weighted_sigmoid_focal_loss, multiclass_nms)
from ..registry import HEADS

""" different from the standard anchor head,
    # ga_anchor_head, use standard 9 anchors to define the positive pixels
    ga_anchor_head assign groundtruth objects to different scales.
    that is, use different (w, h) to try the groundtruth, 
    and use a area threshold to filter the positive and negative regions.
    and the positive area of a object only occur on one feature map.
"""

@HEADS.register_module
class GAAnchorHead(nn.Module):
    """ GA-anchor-based head (GARPN)
    Args:
        in_channels (int) : Number of channels in the input feature map. (output from FPN)
        feat_channels (int): Number of channels in the feature map.
        # use anchor_scales and anchor_ratios to select positive regions.
        anchor_scales (Iterable): Anchor scales ( use to define whether a pixel is positive)
        anchor_ratios (Iterable): Anchor aspects.
        # for each point only select one gt with max IoU.
        anchor_strides (Iterable): Anchor strides.
        anchor_base_sizes (Iterable): Anchor base sizes (Used to assign gt to one feature map)
        # the regression gt use the gt bboxes size. that is the gt dw and dh.
        target_means (Iterable) : Mean values of regression targets.
        target_stds (Iterable): Std values of regression targets.
        use_sigmoid_cls (bool): Whether to use sigmoid loss for classification,
            (softmax by default)
        use_focal_loss (bool): Whether to use focal loss for classification.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 anchor_scales=[8, 16, 32],
                 anchor_ratios=[0.5, 1.0, 2.0],
                 anchor_strides=[4, 8, 16, 32, 64],
                 anchor_base_sizes=None,
                 target_means=(.0, .0, .0, .0), # for 4 points. for GA only regresses 2 points.
                 target_stds=(1., 1., 1., 1.),  # for 4 points.
                 use_sigmoid_cls=False,
                 use_focal_loss=False):
        super(GAAnchorHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(
            anchor_strides) if anchor_base_sizes is None else anchor_base_sizes
        self.target_means = target_means
        self.target_stds = target_stds
        self.use_sigmoid_cls = use_sigmoid_cls
        self.use_focal_loss = use_focal_loss

        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(
                # for different scales and ratios generate different anchors
                AnchorGenerator(anchor_base, anchor_scales, anchor_ratios))

        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes - 1
        else:
            self.cls_out_channels = self.num_classes

        self.__init_layers()

    def __init_layers(self):
        # num_anchors * cls_out_channels
        self.conv_cls = nn.Conv2d(self.feat_channels,
                                  self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        normal_init(self.conv_cls, std=0.01)
        normal_init(self.conv_reg, std=0.01)

    def forward_single(self, x):
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        return cls_score, bbox_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas):
        """
        Get anchors according to feature map sizes.

        :param featmap_sizes: (list[tuple]): Multi-level feature map sizes.
        :param img_metas: (list[dict]): Image meta info.
        :return:
            tuple: anchors of each image, valid flags of each image.
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since featre map sizes of all images are the same, we calcualte
        # all anchors for one time
        multi_level_anchors = []
        for i in range(num_levels):
            # generate anchor of different scales and aspects.
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i])
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                # select the min boundary for valid anchors
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                # just generate the valid flag.
                flags = self.anchor_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)
        return anchor_list, valid_flag_list


    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg):
        # classification loss
        if self.use_sigmoid_cls:
            labels = labels.reshape(-1, self.cls_out_channels)
            label_weights = label_weights.reshape(-1, self.cls_out_channels)
        else:
            labels = labels.reshape(-1)
            label_weights = label_weights.reshape(-1)
        # [N, anchors * cls_out_channles, H, W] -> [N, H, W, anchors * cls_out_channels]
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels)  # cls_out_channels = num_cls
        if self.use_sigmoid_cls:
            if self.use_focal_loss:
                cls_criterion = weighted_sigmoid_focal_loss
            else:
                # use binary_cross_entropy
                cls_criterion = weighted_binary_cross_entropy
        else:
            if self.use_focal_loss:
                raise NotImplementedError
            else:
                cls_criterion = weighted_cross_entropy
        # calculate loss
        if self.use_focal_loss:
            loss_cls = cls_criterion(
                cls_score,
                labels,
                label_weights,
                gamma=cfg.gamma,
                alpha=cfg.alpha,
                avg_factor=num_total_samples)
        else:
            loss_cls = cls_criterion(
                cls_score, labels, label_weights, avg_factor=num_total_samples)

        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)




