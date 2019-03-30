import torch
import torch.nn as nn

from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler


@DETECTORS.register_module
class TwoStageDetector(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)
        else:
            raise NotImplementedError

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)

        if mask_head is not None:
            self.mask_roi_extractor = builder.build_roi_extractor(
                mask_roi_extractor)
            self.mask_head = builder.build_head(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_roi_extractor.init_weights()
            self.mask_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_bboxes_ignore,
                      gt_labels,
                      gt_masks=None,
                      proposals=None):
        # x contains all level feature maps output by backbone.
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            # RPN uses the outputs of backbone and FPN.
            # rpn_outs: (rpn_cls_score, rpn_bbox_pred) of all levels.
            rpn_outs = self.rpn_head(x)         # rpn_cls_score and rpn_bbox_pred(delta)
            # rpn_outs is a tuple like:
            # ([cls_scores_1, cls_scores_2,...], [bbox_pred_1, bbox_pred_2...])
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            # rpn_reg_losses are calculated through anchors and gts.
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs)
            losses.update(rpn_losses)

            # generate proposal_inputs according to the test_cfg.rpn
            # the proposals generated by different rpn are transformed
            # to the original images.
            proposal_inputs = rpn_outs + (img_meta, self.test_cfg.rpn)
            # here to transform pred_delta to proposal(roi)
            # select proposals to assign gts and sample.
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        #
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    # proposals are generated by anchor settings.
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                # sampling_result is a class contains the pos_inds,
                # net_inds(the indexes of bboxes(anchor)), and the pos_assign_gt_inds
                # the pos_assign_gt_inds are the corresponding gt_index in gt_bboxes.
                # and pos_gt_bboxes are corresponding gt_bboxes...
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            # transform bbox(x, y, w, h) to roi(x, y, x, y) format.
            # each res represents an image. bboxes represent: cat(pos_bboxes, neg_bboxes)
            # pos_bboxes-pos_proposals, neg_bboxes-neg_proposals.
            # proposals are produced by rpn, based on anchors.
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            # single_level decide which feature map to use to extract feats
            # and the multi_level map rois to all feature maps and combines all feats
            # the roi region use the output of RPN.
            bbox_feats = self.bbox_roi_extractor(
                # select the needed feature map to extract roi feat.
                x[:self.bbox_roi_extractor.num_inputs], rois)
            # use feats to further predict the delta.
            # and the bbox_pred is corresponding to the rois(the sampling results)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            # bbox_head targets are generated through proposals output by rpn and gt_bboxes
            # target_deltas are calculated through proposals and gt.
            bbox_targets = self.bbox_head.get_target(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)

        # mask head forward and loss
        # if use the bboxes output by faster rcnn, then first should change the
        # bbox_pred to bboxes.
        if self.with_mask:
            # only use the positive proposals to train mask branch.
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], pos_rois)
            # mask pred generated by mask_head.
            # mask_feats are resized to the size of pos_rois ( 28 * 28 )
            mask_pred = self.mask_head(mask_feats)

            mask_targets = self.mask_head.get_target(
                sampling_results, gt_masks, self.train_cfg.rcnn)
            # sample the pos_gt_label, gt label of each sampling pos
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            # pos_label indicates the gt classes.
            # mask_pred: [N, class, H, W]
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
            losses.update(loss_mask)

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x = self.extract_feat(img)

        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0]. to the size of first img.
        """
        # recompute feats to save memory
        """ aug_test_rpn and aug_test_bboxes transform
            the results of multi-scale test images to fit the original image size.
        """
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        # det_bboxes are the result of different scale images.
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            # if not rescale, then transform the bbox result to the size of
            # input images.
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        #
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results
