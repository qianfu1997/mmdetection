#
# @author:charlotte.Song
# @file: fcn_deep_mask_head.py
# @Date: 2019/3/30 21:50
# @description:
# -*- coding: utf-8 -*-
import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn

from ..registry import HEADS
from ..utils import ConvModule
from mmdet.core import mask_cross_entropy, mask_target
""" fcn mask head add residual block version """


@HEADS.register_module
class FCNDeepMaskHead(nn.Module):
    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 upsample_method='deconv',
                 upsample_ratio=2,
                 num_classes=81,
                 class_agnostic=False,
                 normalize=None):
        super()
