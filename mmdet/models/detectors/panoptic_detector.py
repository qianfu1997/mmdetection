#
# @author:charlotte.Song
# @file: ups_panoptic_detector.py
# @Date: 2019/4/19 20:38
# @description: implementation according to Panoptic FPN
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler


@DETECTORS.register_module
class PanopticDetector(BaseDetector, RPNTestMixin, BBoxTestMixin,
                          MaskTestMixin):
    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 panoptic_neck=None,
                 panoptic_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(PanopticDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        else:
            raise NotImplementedError

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)

        if mask_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(mask_roi_extractor)
            self.mask_head = builder.build_head(mask_head)

        if panoptic_head is not None:
            # self.semantic_neck = builder.build_neck(semantic_neck)
            # self.semantic_head = builder.build_head(semantic_head)
            self.panoptic_neck = builder.build_neck(panoptic_neck)
            self.panoptic_head = builder.build_head(panoptic_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_panoptic(self):
        return hasattr(self, 'panoptic_head') and self.panoptic_neck is not None and self.panoptic_head is not None

    def init_weights(self, pretrained=None):
        super(PanopticDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()

        if self.with_rpn:
            self.rpn_head.init_weights()

        if self.with_panoptic:
            self.panoptic_neck.init_weights()
            self.panoptic_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_roi_extractor.init_weights()
            self.mask_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x


    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_stuff_masks=None,
                      gt_stuff_labels=None,
                      proposals=None):
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            # outputs all stages of rpn
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.trai_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            # rpn_losses is a dict, and extend the key-value pair of rpn_losses
            # to losses.
            losses.update(rpn_losses)

            proposal_inputs = rpn_outs + (img_meta, self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        if self.with_bbox or self.with_mask:
            """ this branch allows to densely """








