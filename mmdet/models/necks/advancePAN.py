#
# @author:charlotte.Song
# @file: advancePAN.py
# @Date: 2019/4/11 22:04
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
import torch
from ..utils import ConvModule
from ..registry import NECKS
""" according to TensorMask, add some modification to FPN 
    using averaging instead of addition, using 4 convs instead of 1 conv.
"""


@NECKS.register_module
class advancePAN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 normalize=None,
                 activation=None,
                 num_fpn_convs=4, **kwargs):
        super(advancePAN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.with_bias = normalize is None
        self.num_fpn_cons = num_fpn_convs

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.pan_lateral_convs = nn.ModuleList()
        self.pan_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            fpn_l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)
            pan_l_conv = ConvModule(        # stride 2 to reduce spatial size.
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)
            pan_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)
            self.lateral_convs.append(fpn_l_conv)
            self.pan_lateral_convs.append(pan_l_conv)
            self.pan_convs.append(pan_conv)

            for j in range(self.num_fpn_cons):
                fpn_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    normalize=normalize,
                    bias=self.with_bias,
                    activation=activation,
                    inplace=False)
                self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
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
        # using averaging instead of addition, according to TensorMask
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(#
                laterals[i], scale_factor=2, mode='nearest')
            laterals[i - 1] /= 2.0

        # build outputs
        # part 1: from original levels
        # outs = [
        #     self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        # ]
        outs = [
            laterals[i] for i in range(used_backbone_levels)
        ]
        for i in range(self.num_fpn_cons):
            outs = [
                self.fpn_convs[k * self.num_fpn_cons + i](outs[k]) for k in range(used_backbone_levels)
            ]

        # build pan laterals bottom-up path
        for i in range(0, used_backbone_levels - 1):
            outs[i + 1] += F.interpolate(
                self.pan_lateral_convs[i](outs[i]), size=(outs[i + 1].size()[2], outs[i + 1].size()[3]),
                mode='nearest')
        # build outputs
        outs = [
            self.pan_convs[i](outs[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
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
