#
# @author:charlotte.Song
# @file: deepPAN.py
# @Date: 2019/4/2 12:40
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from ..utils import ConvModule
from ..registry import NECKS

""" an implementation of deepPAN, 
    add another top-down path after pan. 
"""


class stackNeck(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 layer_num,
                 normalize=None,
                 activation=None, **kwargs):
        super(stackNeck, self).__init__()
        """ a PAN(bottom-up path) + top-down path """
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.layer_num = layer_num
        self.normalize = normalize
        self.with_bias = normalize is None
        self.activation = activation

        self.pan_lateral_convs = nn.ModuleList()
        self.pan_convs = nn.ModuleList()
        self.deepan_convs = nn.ModuleList()

        for i in range(self.layer_num):
            in_channel = self.in_channel if i == 0 else self.out_channel
            pan_l_conv = ConvModule(
                in_channel,
                self.out_channel,
                3,
                stride=2,
                padding=1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)
            pan_conv = ConvModule(
                self.out_channel,
                self.out_channel,
                3,
                padding=1,
                normalize=normalize,
                activation=self.activation,
                inplace=False)
            dp_conv = ConvModule(
                self.out_channel,
                self.out_channel,
                3,
                padding=1,
                normalize=normalize,
                activation=activation,
                inplace=False)
            self.pan_lateral_convs.append(pan_l_conv)
            self.pan_convs.append(pan_conv)
            self.deepan_convs.append(dp_conv)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, outs):
        assert len(outs) == self.layer_num

        # build pan laterals bottom-up path
        for i in range(0, self.layer_num - 1):
            outs[i + 1] += F.interpolate(
                self.pan_lateral_convs[i](outs[i]), size=(outs[i + 1].size()[2], outs[i + 1].size()[3]),
                mode='nearest')

        # build mid-outputs
        outs = [
            self.pan_convs[i](outs[i]) for i in range(self.layer_num)
        ]

        # build top-bottom path
        for i in range(self.layer_num - 1, 0, -1):
            outs[i - 1] += F.interpolate(
                outs[i], scale_factor=2, mode='nearest')

        outs = [
            self.deepan_convs[i](outs[i]) for i in range(self.layer_num)
        ]
        return outs


@NECKS.register_module
class DeePAN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 normalize=None,
                 activation=None,
                 stack_num=1,
                 **kwargs):
        super(DeePAN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.with_bias = normalize is None

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level

        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)  # lateral_connection
            self.fpn_convs.append(fpn_conv)  # top-down path

        # make stack deepPan
        self.deepan_layers = []
        for i in range(stack_num):
            deepan_layer = stackNeck(
                out_channels,
                out_channels,
                layer_num=len(self.lateral_convs),
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                replace=False)
            layer_num = 'deepan{}'.format(i + 1)
            self.add_module(layer_num, deepan_layer)
            self.deepan_layers.append(layer_num)

        # add extra conv layers
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                in_channels = (self.in_channels[self.backbone_end_level - 1]
                               if i == 0 else out_channels)
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    normalize=normalize,
                    bias=self.with_bias,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build fpn laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path, the lower pic has a higher resolution.
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(  #
                laterals[i], scale_factor=2, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        #
        for i, layer_name in enumerate(self.deepan_layers):
            deepan_layer = getattr(self, layer_name)
            outs = deepan_layer(outs)

        # part 2 : add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                orig = inputs[self.backbone_end_level - 1]
                outs.append(self.fpn_convs[used_backbone_levels](orig))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    # BUG: we should add relu before each extra conv
                    outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)







