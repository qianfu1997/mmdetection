#
# @author:charlotte.Song
# @file: DeepFPN.py
# @Date: 2019/3/25 15:37
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from ..utils import ConvModule
from ..registry import NECKS

@NECKS.register_module
class DeepFPN(nn.Module):
    def __init__(self,
                 in_channels,   # [256, 512, 1024, 2048]
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 normalize=None,
                 activation=None):
        """  implement deep FPN for RPN
            and the DeepFPN will cascade twice.
            DeepFPN but not use bn and relu.
        """
        super(DeepFPN, self).__init__()
        # FPN should get outputs from different depths
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
            # if end_level < inputs, no extra level is allowed.
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level

        self.lateral_convs = nn.ModuleList()        # the lateral path
        self.fpn_convs = nn.ModuleList()            # the fpn path.
        # for deep FPN add another lateral_path and top
        # cascade once.
        self.td_fpn_3x3convs = nn.ModuleList()
        self.td_fpn_1x1convs = nn.ModuleList()
        self.dt_fpn_3x3convs = nn.ModuleList()
        self.dt_fpn_1x1convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(        # for lateral only 1x1 conv
                in_channels[i],
                out_channels,
                1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False
            )
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)
            td_fpn_3x3conv = ConvModule(
                out_channels,
                out_channels,
                3, padding=1, normalize=normalize,
                bias=self.with_bias,
                activation=self.activation, inplace=False
            )
            td_fpn_1x1conv = ConvModule(
                out_channels, out_channels,
                1, padding=0, normalize=normalize,
                bias=self.with_bias, activation=self.activation, inplace=False
            )
            dt_fpn_3x3conv = ConvModule(
                out_channels, out_channels,
                3, stride=2, padding=1, normalize=normalize,
                bias=self.with_bias, activation=self.activation, inplace=False
            )
            dt_fpn_1x1conv = ConvModule(
                out_channels, out_channels,
                1, padding=0, normalize=normalize,
                bias=self.with_bias, activation=self.activation, inplace=False
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.td_fpn_3x3convs.append(td_fpn_3x3conv)
            self.td_fpn_1x1convs.append(td_fpn_1x1conv)
            self.dt_fpn_3x3convs.append(dt_fpn_3x3conv)
            self.dt_fpn_1x1convs.append(dt_fpn_1x1conv)

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
            self.fpn.convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        # first lateral then fpn
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        # first
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        # build media results.
        # part 1: from original levels
        outs = [
            # use fpn convs to calculate lateral layers.
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # deep FPN build bottom-up path
        for i in range(used_backbone_levels - 1, 0, -1):
            outs[i - 1] += F.interpolate(
                outs[i], scale_factor=2, mode='nearest')
        # bottom-up path
        #
        for i in range(1, used_backbone_levels):
            outs[i] = self.dt_fpn_1x1convs[i](self.dt_fpn_3x3convs[i](outs[i - 1]))

        # deep FPN build top-down path again.
        for i in range(used_backbone_levels - 1, 0, -1):
            outs[i - 1] += F.interpolate(
                outs[i], scale_factor=2, mode='nearest')
        # top-down path again
        outs = [self.td_fpn_1x1convs[i](self.td_fpn_1x1convs[i](outs[i]))
                for i in range(used_backbone_levels)]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps
            else:
                orig = inputs[self.backbone_end_level - 1]
                outs.append(self.fpn_convs[used_backbone_levels](orig))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)



