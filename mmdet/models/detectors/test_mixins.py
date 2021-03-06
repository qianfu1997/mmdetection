from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_proposals,
                        merge_aug_bboxes, merge_aug_masks, multiclass_nms)

""" for ga_anchor here to add TestMixin """
class RPNTestMixin(object):

    def simple_test_rpn(self, x, img_meta, rpn_test_cfg):
        rpn_outs = self.rpn_head(x)
        proposal_inputs = rpn_outs + (img_meta, rpn_test_cfg)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        return proposal_list

    def aug_test_rpn(self, feats, img_metas, rpn_test_cfg):
        imgs_per_gpu = len(img_metas[0])
        aug_proposals = [[] for _ in range(imgs_per_gpu)]
        for x, img_meta in zip(feats, img_metas):
            # get proposals from different scales.
            proposal_list = self.simple_test_rpn(x, img_meta, rpn_test_cfg)
            for i, proposals in enumerate(proposal_list):
                aug_proposals[i].append(proposals)

        # after merging, proposals will be rescaled to the original image size
        merged_proposals = [
            merge_aug_proposals(proposals, img_meta, rpn_test_cfg)
            for proposals, img_meta in zip(aug_proposals, img_metas)
        ]
        return merged_proposals


class BBoxTestMixin(object):
    # the TestMixin for test bbox head.
    def simple_test_bboxes(self,
                           x,
                           img_meta,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)              # the proposals are the output of RPN
        roi_feats = self.bbox_roi_extractor(    # get corresponding feat of proposals.
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        cls_score, bbox_pred = self.bbox_head(roi_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        # det labels are the classes name.
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)      # filter bboxes according to test cfg.
        return det_bboxes, det_labels

    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            # for each img and img_meta.
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            # TODO more flexible
            # only use the proposal_list[0] of original image,
            # so the bboxes are corresponding. the proposal list
            # are merged from different scales.
            # RPN generate proposals for different scale images, then
            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip)
            rois = bbox2roi([proposals])
            # recompute feature maps to save GPU memory
            roi_feats = self.bbox_roi_extractor(
                x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            cls_score, bbox_pred = self.bbox_head(roi_feats)
            bboxes, scores = self.bbox_head.get_det_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        # after merging, bboxes will be rescaled to the original image size
        # here to calculate the mean box.
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(
            merged_bboxes, merged_scores, rcnn_test_cfg.score_thr,
            rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)
        return det_bboxes, det_labels

    def aug_test_cascadercnn_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        """ implement for cascade aug test. """
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            img_shape = img_meta[0]['img_shape']
            ori_shape = img_meta[0]['ori_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']

            # TODO more flexible
            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip)
            rois = bbox2roi([proposals])

            # get bbox from different stages
            ms_bbox_result = {}
            ms_segm_result = {}
            ms_scores = []
            rcnn_test_cfg = self.test_cfg_rcnn
            for i in range(self.num_stages):
                # for each image there are 3 stages to go through.
                bbox_roi_extractor = self.bbox_roi_extractor[i]
                bbox_head = self.bbox_head[i]
                bbox_feats = bbox_roi_extractor(
                    x[:len(bbox_roi_extractor.featmap_strides), rois])
                cls_score, bbox_pred = bbox_head(bbox_feats)
                # get bboxes scores from all num_stages
                ms_scores.append(cls_score)
                # the rois are the same, so the predicted bboxes can be
                # average between them
                if self.test_cfg.keep_all_stages:
                    det_bboxes, det_labels = bbox_head.get_det_bboxes(
                        rois,
                        cls_score,
                        bbox_pred,
                        img_shape,
                        scale_factor,
                        rescale=False,
                        cfg=None)
                    aug_bboxes.append(det_bboxes)

                if i < self.num_stages - 1:
                    bbox_label = cls_score.argmax(dim=1)
                    rois = bbox_head.regress_by_class(rois, bbox_label, bbox_pred,
                                                      img_meta[0])
            cls_score = sum(ms_scores) / self.num_stages
            bboxes, scores = self.bbox_head[-1].get_det_bboxes(
                rois,
                cls_score,
                bbox_pred,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(
            merged_bboxes, merged_scores, rcnn_test_cfg.score_thr,
            rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)
        return det_bboxes, det_labels


class MaskTestMixin(object):

    def simple_test_mask(self,
                         x,
                         img_meta,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        # image shape of the first image in the batch (only one)
        ori_shape = img_meta[0]['ori_shape']
        # the factor of rescaled img between the ori image.
        scale_factor = img_meta[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            _bboxes = (det_bboxes[:, :4] * scale_factor
                       if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_feats = self.mask_roi_extractor(
                x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            mask_pred = self.mask_head(mask_feats)
            segm_result = self.mask_head.get_seg_masks(
                mask_pred, _bboxes, det_labels, self.test_cfg.rcnn, ori_shape,
                scale_factor, rescale)
        return segm_result

    def aug_test_mask(self, feats, img_metas, det_bboxes, det_labels):
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            aug_masks = []
            for x, img_meta in zip(feats, img_metas):
                img_shape = img_meta[0]['img_shape']
                scale_factor = img_meta[0]['scale_factor']
                flip = img_meta[0]['flip']
                _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                       scale_factor, flip)
                mask_rois = bbox2roi([_bboxes])
                mask_feats = self.mask_roi_extractor(
                    x[:len(self.mask_roi_extractor.featmap_strides)],
                    mask_rois)
                mask_pred = self.mask_head(mask_feats)
                # convert to numpy array to save memory
                aug_masks.append(mask_pred.sigmoid().cpu().numpy())
            # here is to flip back the masks.
            merged_masks = merge_aug_masks(aug_masks, img_metas,
                                           self.test_cfg.rcnn)

            ori_shape = img_metas[0][0]['ori_shape']
            # sxg: for aug test, the masks are always fixed to
            # the size of img[0]
            segm_result = self.mask_head.get_seg_masks(
                merged_masks,
                det_bboxes,
                det_labels,
                self.test_cfg.rcnn,
                ori_shape,                  # the mask size is the ori ones.
                scale_factor=1.0,           # and won't rescale to the rescale img.
                rescale=False)
        return segm_result

    def aug_test_cascade_mask(self, feats, img_metas, det_bboxes, det_labels):
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head[-1].num_classes - 1)]
        else:
            aug_masks = []
            aug_img_metas = []
            if self.mask_head.mask_score:
                aug_mask_ious = []
                for x, img_meta in zip(feats, img_metas):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']
                    _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                           scale_factor, flip)
                    mask_rois = bbox2roi([_bboxes])
                    for i in range(self.num_stages):
                        mask_roi_extractor = self.mask_roi_extractor[i]
                        mask_feats = mask_roi_extractor(
                            x[:len(mask_roi_extractor.featmap_strides)],
                            mask_rois)
                        mask_pred = self.mask_head[i](mask_feats)
                        aug_masks.append(mask_pred)
                        aug_img_metas.append(img_meta)
                merged_masks = merge_aug_masks(aug_masks, aug_img_metas,
                                               self.test_cfg.rcnn)
                ori_shape = img_metas[0][0]['ori_shape']
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks,
                    det_bboxes,
                    det_labels,
                    self.test_cfg.rcnn,
                    ori_shape,
                    scale_factor=1.0,
                    rescale=False)
        return segm_result

