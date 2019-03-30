#
# @author:charlotte.Song
# @file: ga_rpn_head.py
# @Date: 2019/3/26 22:26
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import delta2bbox
from mmdet.ops import nms
from .anchor_head import AnchorHead
from ..registry import HEADS