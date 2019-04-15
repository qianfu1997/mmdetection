#
# @author:charlotte.Song
# @file: cbam_multi_level.py
# @Date: 2019/4/12 22:36
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init

from mmdet import ops
from ..registry import ROI_EXTRACTORS
from ..utils import ConvModule
""" add CBAM attention mechanism to select features. """


@ROI_EXTRACTORS.register_module
class CBAMMultiRoiExtractor(nn.Module):
    """Extract RoI features from multi-level feature maps.
    If there' re multiple input feature levels, each RoI is mapping to all feature maps,
    and select a RoI through a CBAM structure.
    """
    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 reduction=4,
                 with_cam=True,
                 with_sam=False):
        super(CBAMMultiRoiExtractor, self).__init__()
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.reduction = reduction
        self.with_cam = with_cam
        if self.with_cam:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.c_fc0 = nn.Linear(self.out_channels, self.out_channels // self.reduction)
            self.c_fc1 = nn.Linear(self.out_channels // self.reduction, self.out_channels)
            self.activation = nn.Sigmoid()
        # if self.with_sam:
        #     self.sam = ConvModule(
        #         out_channels,
        #         out_channels,
        #         3,
        #         padding=1,
        #         normalize=None,
        #         bias=True,
        #         inplace=False)

    @property
    def num_inputs(self):
        return len(self.featmap_strides)

    def init_weights(self):
        if self.with_cam:
            xavier_init(self.c_fc0, distribution='uniform')
            xavier_init(self.c_fc1, distribution='uniform')
        # if self.with_sam:
        #     xavier_init(self.sam, distribution='uniform')

    def build_roi_layers(self, layer_cfg, featmap_strides):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type, )
        layer_cls = getattr(ops, layer_type)
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    def cam(self, roi_feats):
        c_roi = self._pool(roi_feats)
        c_roi = c_roi.view(c_roi.size(0), -1)
        c_roi = self.activation(self.c_fc1(self.c_fc0(c_roi))).view(
            c_roi.size(0), self.out_channels, 1, 1)
        c_roi = c_roi.repeat(1, 1, roi_feats.size(2), roi_feats.size(3)) * roi_feats
        roi_feats = c_roi + roi_feats
        return roi_feats

    def forward(self, feats, rois):
        if len(feats) == 1:
            roi_feats = self.roi_layers[0](feats[0], rois)
            if self.with_cam:
                roi_feats = self.cam(roi_feats)
            return roi_feats

        out_size = self.roi_layers[0].out_size
        num_levels = len(feats)
        roi_feats = torch.cuda.FloatTensor(rois.size(0), self.out_channels,
                                           out_size, out_size)

        for i in range(num_levels):
            roi_feats_t = self.roi_layers[i](feats[i], rois)
            roi_feats_t = self.cam(roi_feats_t)
            roi_feats = torch.max(roi_feats, roi_feats_t)
        return roi_feats



            



