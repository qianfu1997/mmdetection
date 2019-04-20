#
# @author:charlotte.Song
# @file: multi_level.py
# @Date: 2019/3/28 14:25
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn

from mmdet import ops
from ..registry import ROI_EXTRACTORS


@ROI_EXTRACTORS.register_module
class MultiRoIExtractor(nn.Module):
    """Extract RoI features from multi-level feature map.
    If there' re multiple input feature levels, each RoI is mapped to all level feature map.
    Args:
        roi_layer(dict): Specify RoI layer type and arguments.# such as RoIAlign
            and RoIPool
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.

    Map one roi to all level of feature maps, and then fuse and combine then
    through element-wise sum / max.
    """
    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides   # used to initialize roi_layer
                 ):
        super(MultiRoIExtractor, self).__init__()
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides

    @property
    def num_inputs(self):
        """int: Input feautre map levels."""
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

    # directly map all rois to all feature maps
    # def forward(self, feats, rois):
    #     if len(feats) == 1:
    #         return self.roi_layers[0](feats[0], rois)
    #
    #     out_size = self.roi_layers[0].out_size
    #     num_levels = len(feats)
    #     # map roi to each feature map
    #     # map roi to one feature map
    #     target_lvls = self.map_roi_levels(rois, num_levels)
    #     # init the container.
    #     roi_feats = torch.cuda.FloatTensor(rois.size()[0], self.out_channels,
    #                                        out_size, out_size).fill_(0)
    #     for i in range(num_levels):
    #         inds = target_lvls == i
    #         if inds.any():
    #             rois_ = rois[inds, :]
    #             # use the corresponding roi_lpanayer to
    #             roi_feats_t = self.roi_layers[i](feats[i], rois_)
    #             roi_feats[inds] += roi_feats_t
    #     return roi_feats
    def forward(self, feats, rois):
        """ directly map all rois to all feature maps
            and then get roi by max(roi, roi_)
        """
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        out_size = self.roi_layers[0].out_size
        num_levels = len(feats)
        roi_feats = torch.cuda.FloatTensor(rois.size()[0], self.out_channels,
                                           out_size, out_size).fill_(0)
        for i in range(num_levels):
            # calculate all rois, all rois are the same size.
            roi_feats_t = self.roi_layers[i](feats[i], rois)
            # element-wise max / can be transformed to element-wise sum
            roi_feats = torch.max(roi_feats, roi_feats_t)
        return roi_feats


