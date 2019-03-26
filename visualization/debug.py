#
# @author:charlotte.Song
# @file: debug.py
# @Date: 2019/3/4 21:31
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import argparse
import torch


""" used to find if there're 
    coordinates of a box overlap the image boundaries
"""

data_root = '../data/'
LSVT_gt_root = data_root + 'LSVT/annotations/'


