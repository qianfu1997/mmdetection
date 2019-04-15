#
# @author:charlotte.Song
# @file: custom_losses.py
# @Date: 2019/3/19 21:39
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn.functional as F

""" implement dice loss """


def mask_binary_dice_loss(pred, target, label):
    """ implement dice loss for mask
        use dice loss to train the mask branch.
        for dice loss the label is set to zero.
    """
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    # select the pred_masks.
    # for each roi only select the corresponding channel to train
    pred_slice = pred[inds, label].squeeze(1)

    pred_slice = torch.sigmoid(pred_slice)

    pred_slice = pred_slice.contiguous().view(num_rois, -1)
    target = target.contiguous().view(num_rois, -1)

    a = torch.sum(pred_slice * target, 1)
    b = torch.sum(pred_slice * pred_slice, 1) + 1e-4
    c = torch.sum(target * target, 1) + 1e-4
    d = (2.0 * a) / (b + c)
    return (1 - torch.mean(d))[None]


def weighted_mask_binary_dice_loss(pred, target, label, weight):
    """
    use weighted dice loss to train the mask branch
    """
    num_rois = pred.size()[0]
    assert num_rois == target.size()[0] == weight.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.divice)
    pred_slice = pred[inds, label].squeeze(1)

    pred_slice = torch.sigmoid(pred_slice)
    pred_slice = pred_slice.contiguous().view(num_rois, -1)
    target = target.contiguous().view(num_rois, -1)
    weight = weight.contiguous().view(num_rois, -1)

    pred_slice = pred_slice * weight
    target = target * weight

    a = torch.sum(pred_slice * target, 1)
    b = torch.sum(pred_slice * pred_slice, 1) + 1e-4
    c = torch.sum(target * target, 1) + 1e-4
    d = (2.0 * a) / (b + c)
    return (1 - torch.mean(d))[None]


def ohem_mask_binary_dice_loss(pred, target, label, ratio=3):
    """
    use ohem + dice loss to train the mask branch, select positive
    """
    pass

def combined_binary_dice_cross_entropy_loss(pred, target, label):
    """ combine the binary dice loss and binary cross entropy loss """
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)

    bce_loss = F.binary_cross_entropy_with_logits(
        pred_slice, target, reduction='mean')[None]

    # calculate dice loss
    pred_slice = torch.sigmoid(pred_slice).contiguous().view(num_rois, -1)
    target = target.contiguous().view(num_rois, -1)

    a = torch.sum(pred_slice * target, 1)
    b = torch.sum(pred_slice * pred_slice, 1) + 1e-4
    c = torch.sum(target * target, 1) + 1e-4
    d = (2.0 * a) / (b + c)
    dice_loss = (1 - torch.mean(d))[None]
    combined_loss = 0.5 * bce_loss + 0.5 * dice_loss
    return combined_loss


def bounded_iou_loss(pred, target, eps=1e-3, reduction='mean'):
    """ implemented from denet: bounded_iou_loss
    attention: only for guided anchoring.
        pred: [xc, yc, w, h]
        target: [x, y, w, h]
    """
    assert eps > 0
    assert pred.size() == target.size() and target.numel() > 0

    return None


""" implement GIoU according to paper """




