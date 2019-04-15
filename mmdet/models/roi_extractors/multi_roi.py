#
# @author:charlotte.Song
# @file: multi_roi.py
# @Date: 2019/4/5 21:01
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn

from mmdet import ops
from ..registry import ROI_EXTRACTORS
""" concat roi from 4 feature maps together to 
    make a better prediction.
"""

@ROI_EXTRACTORS.register_module
class MultiLayerRoIExtractor(nn.Module):
    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides):
        super(MultiLayerRoIExtractor, self).__init__()
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides

    @property
    def num_inputs(self):
        return len(self.featmap_strides)

    def init_weights(self):
        pass

    def build_roi_layers(self, layer_cfg, featmap_strides):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)  # use RoIAlign or RoIPooling.
        layer_cls = getattr(ops, layer_type)
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales"""
        scale = torch.sqrt( # the area of rois
            (rois[:, 3] - rois[:, 1] + 1) * (rois[:, 4] - rois[:, 2] + 1))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def forward(self, feats, rois):
        """ directly map all rois to all feature maps
            and concat all rois together.
        """
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        out_size = self.roi_layers[0].out_size
        num_levels = len(feats)
        roi_feats = torch.cuda.FloatTensor(
            rois.size()[0], self.out_channels * num_levels,
            out_size, out_size).fill_(0)

        for i in range(num_levels):
            roi_feats_t = self.roi_layers[i](feats[i], rois)
            roi_feats[:, i * self.out_channels:(i + 1) * self.out_channels, :, :] += roi_feats_t

        return roi_feats