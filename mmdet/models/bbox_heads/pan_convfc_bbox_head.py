import torch.nn as nn
import torch
from .bbox_head import BBoxHead
from ..registry import HEADS
from ..utils import ConvModule

@HEADS.register_module
class PANConvFCBBoxHead(BBoxHead):
    """More general bbox head, with shared conv and fc layers and two optional
    separated branches.
    designed for PAN version2.

                                /-> cls convs -> cls fcs -> cls
    shared convs -> shared fcs
                                \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,         # default use the same with in_channels.
                 fc_out_channels=1024,
                 normalize=None,
                 len_strides=4,
                 *args,
                 **kwargs):
        # init BBoxHead for PAN, the max will be selected in shared_fcs.
        super(PANConvFCBBoxHead, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs + num_cls_fcs
                + num_reg_convs + num_reg_fcs > 0)
        self.multi_roi = len_strides > 1
        self.len_strides = len_strides
        if self.multi_roi:
            """ if num_shared_convs > 0, then select will be happen after first convs
                else between two fcs.
            """
            assert num_shared_fcs > 0 or num_shared_convs > 0
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_convs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.normalize = normalize
        self.with_bias = normalize is None

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                                     True, len_stride=self.len_strides, multi_roi_flag=self.multi_roi)

        self.shared_out_channels = last_layer_dim
        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = self._add_conv_fc_branch(
            self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= (self.roi_feat_size * self.roi_feat_size)
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= (self.roi_feat_size * self.roi_feat_size)

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        # use another fc_cls to reconstruct the cls_num_classes.
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes)
        # same.
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else
                           4 * self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,     # fc in this branch.
                            in_channels,        # self.in_channels for each roi
                            is_shared=False,
                            len_stride=1,
                            multi_roi_flag=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        only shared branch can deal with multi-layer roi
        """
        if multi_roi_flag:
            assert len_stride > 1 and is_shared
        last_layer_dim = in_channels
        branch_convs = nn.ModuleList()
        """ if len_stride > 1, add convs in shared_convs """
        if num_branch_convs > 0:
            if multi_roi_flag:
                for i in range(len_stride):
                    conv_in_channels = last_layer_dim
                    branch_convs.append(
                        ConvModule(
                            conv_in_channels,
                            self.conv_out_channels,
                            3,
                            padding=1,
                            normalize=self.normalize,
                            bias=self.with_bias))
                for i in range(num_branch_convs - 1):
                    branch_convs.append(
                        ConvModule(
                            self.conv_out_channels,
                            self.conv_out_channels,
                            3,
                            padding=1,
                            normalize=self.normalize,
                            bias=self.with_bias))
                multi_roi_flag = False
            else:
                for i in range(num_branch_convs):
                    conv_in_channels = last_layer_dim if i == 0 else self.conv_out_channels
                    branch_convs.append(
                        ConvModule(
                            conv_in_channels,
                            self.conv_out_channels,
                            3,
                            padding=1,
                            normalize=self.normalize,
                            bias=self.with_bias))
            last_layer_dim = self.conv_out_channels
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            if (is_shared
                    or self.num_shared_convs == 0) and not self.with_avg_pool:
                last_layer_dim *= (self.roi_feat_size * self.roi_feat_size)
            if multi_roi_flag:
                for i in range(len_stride):
                    fc_in_channels = last_layer_dim
                    branch_fcs.append(
                        nn.Linear(fc_in_channels, self.fc_out_channels))
                for i in range(num_branch_fcs - 1):
                    branch_fcs.append(
                        nn.Linear(self.fc_out_channels, self.fc_out_channels))
            else:
                for i in range(num_branch_fcs):
                    fc_in_channels = (last_layer_dim  # 256 * 7 * 7
                                      if i == 0 else self.fc_out_channels)
                    branch_fcs.append(
                        nn.Linear(fc_in_channels, self.fc_out_channels))  # 1024
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(PANConvFCBBoxHead, self).init_weights()
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input = [x[:, i*self.in_channels:(i+1)*self.in_channels, :, :]
                 for i in range(self.len_strides)]
        len_stride = self.len_strides
        if self.num_shared_convs > 0:
            for i in range(len_stride):
                input[i] = self.shared_convs[i](input[i])
            for i in range(1, len_stride):
                input[0] = torch.max(input[0], input[i])
            for conv in self.shared_convs[len_stride:]:
                input[0] = conv(input[0])
            len_stride = 1

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                input = [self.avg_pool(input[i]) for i in range(len_stride)]
            for i in range(len_stride):
                input[i] = input[i].view(input[i].size(0), -1)
                input[i] = self.relu(self.shared_fcs[i](input[i]))
            for i in range(1, len_stride):
                input[0] = torch.max(input[0], input[i])
            for fc in self.shared_fcs[len_stride:]:
                input[0] = self.relu(fc(input[0]))

        x_cls = input[0]
        x_reg = input[0]

        for conv in self.cls_convs:     # convs of cls_branch.
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.view(x_cls.size(0), -1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred


@HEADS.register_module
class PANSharedFCBBoxHead(PANConvFCBBoxHead):

    def __init__(self, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        assert num_fcs >= 1
        super(PANSharedFCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)




