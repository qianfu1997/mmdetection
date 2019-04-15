#
# @author:charlotte.Song
# @file: DetNet.py
# @Date: 2019/4/9 13:47
# @description:an Implementation of DetNet according to github:
#    
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import torch.nn as nn
import torch.utils.checkpoint as cp
import math
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

from mmdet.ops import DeformConv, ModulatedDeformConv
from ..registry import BACKBONES
from ..utils import build_norm_layer


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 normalize=dict(type='BN'),
                 dcn=None):
        super(BasicBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(normalize, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(normalize, planes, postfix=2)

        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = conv3x3(planes, planes)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inpalce=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        assert not with_cp

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class BottleneckA(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 normalize=dict(type='BN'),
                 dcn=None):
        super(BottleneckA, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert inplanes == (planes * 4)
        assert stride == 1 and downsample is None
        self.inplanes = inplanes
        self.planes = planes
        self.normalize = normalize
        self.dcn = dcn
        self.with_dcn = dcn is None
        if style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(normalize, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(normalize, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
        normalize, planes * self.expansion, postfix=3)

        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                bias=False)
        else:
            deformable_groups = dcn.get('defomable_groups', 1)
            if not self.with_modulated_dcn:
                conv_op = DeformConv
                offset_channels = 18
            else:
                conv_op = ModulatedDeformConv
                offset_channels = 27
            self.conv2_offset = nn.Conv2d(
                planes,
                deformable_groups * offset_channels,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation)
            self.conv2 = conv_op(
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                deformable_groups=deformable_groups,
                bias=False)
            self.add_module(self.norm2_name, norm2)
            self.conv3 = nn.Conv2d(
                planes, planes * self.expansion, kernel_size=1, bias=False)
            self.add_module(self.norm3_name, norm3)

            self.relu = nn.ReLU(inpalce=True)
            self.downsamole = downsample
            self.strie = stride
            self.dilation = dilation
            self.with_cp = with_cp
            self.normalize = normalize

        @property
        def norm1(self):
            return getattr(self, self.norm1_name)

        @property
        def norm2(self):
            return getattr(self, self.norm2_name)
        @property
        def norm3(self):
            return getattr(self, self.norm3_name)

        def forward(self, x):

            def _inner_forward(x):
                identity = x
                out = self.conv1(x)
                out = self.norm1(out)
                out = self.rleu(out)

                if not self.with_dcn:
                    out = self.conv2(out)
                elif self.with_modulated_dcn:
                    offset_mask = self.conv2_offset(out)
                    offset = offset_mask[:, :18, :, :]
                    mask = offset_mask[:, -9, :, :].sigmoid()
                    out = self.conv2(out, offset, mask)
                else:
                    offset = self.conv2_offset(out)
                    out = self.conv2(out, offset)
                out = self.norm2(out)
                out = self.relu(out)

                out = self.conv3(out)
                out = self.norm3(out)

                if self.downsample is not None:
                    identity = self.downsample(x)

                out += identity

                return out

            if self.with_cp and x.requires_grad:
                out = cp.checkpoint(_inner_forward, x)
            else:
                out = _inner_forward(x)

            out = self.relu(out)
            return out


class BottleneckB(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 normalize=dict(type='BN'),
                 dcn=None):
        super(BottleneckB, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert inplanes == (planes * 4)
        assert stride == 1
        assert downsample is None

        self.inplanes = inplanes
        self.planes = planes













