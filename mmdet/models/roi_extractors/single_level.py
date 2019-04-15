from __future__ import division

import torch
import torch.nn as nn

from mmdet import ops
from ..registry import ROI_EXTRACTORS


@ROI_EXTRACTORS.register_module
class SingleRoIExtractor(nn.Module):
    """Extract RoI features from a single level feature map.

    If there are mulitple input feature levels, each RoI is mapped to a level
    according to its scale.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0.
    """

    def __init__(self,  # RoI layer a dict of how to crop RoI region, the output size and out channels.
                 roi_layer,                 # RoI_layer how to crop RoI and the output size.
                 out_channels,              # RoI
                 featmap_strides,           #
                 finest_scale=56):          # 28
        super(SingleRoIExtractor, self).__init__()
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale

    @property
    def num_inputs(self):
        """int: Input feature map levels."""
        # the total num of feature maps.
        return len(self.featmap_strides)

    def init_weights(self):
        pass

    def build_roi_layers(self, layer_cfg, featmap_strides):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)     # use RoIAlign or RoIPooling.
        layer_cls = getattr(ops, layer_type)
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale: level 0
        - finest_scale <= scale < finest_scale * 2: level 1
        - finest_scale * 2 <= scale < finest_scale * 4: level 2
        - scale >= finest_scale * 4: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).        # 4 coordinates and 1 confidence.
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt( # roi [x, y, x, y]; bbox [xc, yc, w, h]
            # calculate the area of rois
            (rois[:, 3] - rois[:, 1] + 1) * (rois[:, 4] - rois[:, 2] + 1))
        # automatically select the corresponding feature levels.
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        # clip the boundaries.
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def forward(self, feats, rois):     # feats, rois.
        if len(feats) == 1:             # rois [x, y, x, y]
            # rois are proposals corresponding to original input image.
            return self.roi_layers[0](feats[0], rois)

        out_size = self.roi_layers[0].out_size      # the roi_layer's out size.
        num_levels = len(feats)
        # map roi to one feature map.
        target_lvls = self.map_roi_levels(rois, num_levels)
        roi_feats = torch.cuda.FloatTensor(rois.size()[0], self.out_channels,
                                           out_size, out_size).fill_(0)
        for i in range(num_levels):
            inds = target_lvls == i
            if inds.any():
                rois_ = rois[inds, :]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] += roi_feats_t
        return roi_feats
