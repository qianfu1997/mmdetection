import mmcv
import numpy as np
from numpy import random
import cv2

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps

""" in this version, will check the crop mask and do not 
    use the small mask.
"""


class PhotoMetricDistortion(object):

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, img, boxes, labels):
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        return img, boxes, labels


class Expand(object):

    def __init__(self, mean=(0, 0, 0), to_rgb=True, ratio_range=(1, 4)):
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range

    def __call__(self, img, boxes, labels):
        if random.randint(2):
            return img, boxes, labels

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img
        img = expand_img
        boxes += np.tile((left, top), 2)
        return img, boxes, labels


class RandomCrop(object):

    def __init__(self,
                 min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                 min_crop_size=0.3):
        # 1: return ori img
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, img, boxes, labels):
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            if mode == 1:
                return img, boxes, labels

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array((int(left), int(top), int(left + new_w),
                                  int(top + new_h)))
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = (center[:, 0] > patch[0]) * (
                    center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (
                        center[:, 1] < patch[3])
                if not mask.any():
                    continue
                boxes = boxes[mask]
                labels = labels[mask]

                # adjust boxes
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                boxes -= np.tile(patch[:2], 2)

                return img, boxes, labels


def get_rect_from_cnt(cnt):
    cnt_points = np.array(cnt).reshape(-1, 2).astype(np.int64)
    box = np.zeros(4)
    box[:2] = np.min(cnt_points, axis=0) - 4
    box[2:] = np.max(cnt_points, axis=0) + 4
    return box


def get_cnt_from_mask(mask):
    points = np.array(np.where(mask > 0)).transpose((1, 0))[:, ::-1]
    if points.shape[0] > 0:
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        if cnt.shape[0] <= 3:
            return None
        else:
            return cnt
    else:
        return None


def clip_box(bboxes, img_w, img_h):
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_w - 1)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_h - 1)
    return bboxes


def bbox_overlap_ic(bboxes1, bboxes2, mode='iob'):
    """
    :param bboxes1:np.ndarray (n, 4)
    :param bboxes2: np.ndarray (k, 4)
    :param mode: iou (intersection over union) or iof (intersection over foreground)
                 iob (intersection over box)
    :returns:
        ious(ndarray): shape (n, k)
    """
    assert mode in ['iou', 'iof', 'iob']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
        bboxes1[:, 3] - bboxes2[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
        bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[:, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[:, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[:, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[:, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(
            y_end - y_start + 1, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        elif mode == 'iof':
            union = area1[i] if not exchange else area2
        else:
            union = area2 if not exchange else area1[i]
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious


class PhotoMetricDistortionIC(object):

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, img):
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        return img


class advanceRandomRotationIC(object):
    def __init__(self,
                 max_angle=20,
                 ver_flip_ratio=0.0,
                 angle_flip=0):
        self.max_angle = max_angle
        self.ver_flip_ratio = ver_flip_ratio
        self.angle_flip = angle_flip

    # input: img, gt_bboxes, gt_labels, gt_masks, gt_ignore_bboxes,
    # gt_ignore_masks, img_shape, pad_shape, ratio
    def __call__(self, img, gt_bboxes, gt_labels, gt_masks, gt_ignore_bboxes,
                 gt_ignore_masks, img_shape, pad_shape, return_angle=False):
        """ keep the img_shape and pad_shape still """
        pad_h, pad_w, _ = pad_shape
        img_h, img_w, _ = img_shape
        img = img.transpose(1, 2, 0).copy()
        assert img.shape[0] == pad_h and img.shape[1] == pad_w
        angle = np.random.rand() * 2 * self.max_angle - self.max_angle
        random_anchor = [90, 270]
        angle += random.choice(random_anchor) if bool(self.angle_flip > 0 and
                                                      np.random.rand() < self.angle_flip) else 0
        # first ver_flip
        ver_flip_f = bool(self.ver_flip_ratio > 0 and np.random.rand() < self.ver_flip_ratio)
        rotation_matrix = cv2.getRotationMatrix2D((int(img_w / 2), int(img_h / 2)), angle, 1)
        img_rotation = img[:img_h, :img_w, :].copy()
        if ver_flip_f:
            img_rotation = np.flip(img_rotation, 0)
        img_rotation = cv2.warpAffine(img_rotation, rotation_matrix, (img_w, img_h))
        img[:img_h, :img_w, :] = img_rotation.copy()

        new_gt_bbox, new_gt_mask, new_gt_labels = [], [], []
        for idx in range(len(gt_masks)):
            mask_rotation = gt_masks[idx][:img_h, :img_w].copy()
            if ver_flip_f:
                mask_rotation = np.flip(mask_rotation, 0)
            mask_rotation = cv2.warpAffine(
                mask_rotation, rotation_matrix, (img_w, img_h))
            if np.max(mask_rotation) > 0:
                cnt = get_cnt_from_mask(mask_rotation)
                if cnt is not None:
                    # generate the bounding boxes
                    box = get_rect_from_cnt(cnt)
                    new_gt_bbox.append(box.copy())
                    new_mask = gt_masks[idx].copy()
                    new_mask[:img_h, :img_w] = mask_rotation.copy()
                    new_gt_mask.append(new_mask)
                    new_gt_labels.append(gt_labels[idx])
        new_gt_ignore_bbox = []
        new_gt_ignore_mask = []
        for idx in range(len(gt_ignore_masks)):
            mask_rotation = gt_ignore_masks[idx][:img_h, :img_w].copy()
            if ver_flip_f:
                mask_rotation = np.flip(mask_rotation, 0)
            mask_rotation = cv2.warpAffine(
                mask_rotation, rotation_matrix, (img_w, img_h))
            if np.max(mask_rotation) > 0:
                cnt = get_cnt_from_mask(mask_rotation)
                if cnt is not None:
                    box = get_rect_from_cnt(cnt)
                    new_gt_ignore_bbox.append(box.copy())
                    new_mask = gt_ignore_masks[idx].copy()
                    new_mask[:img_h, :img_w] = mask_rotation.copy()
                    new_gt_ignore_mask.append(new_mask)
        if new_gt_bbox:
            new_gt_bbox = np.array(new_gt_bbox, dtype=np.float32)
            new_gt_bbox = clip_box(new_gt_bbox, img_w=img_w, img_h=img_h)
            new_gt_labels = np.array(new_gt_labels, dtype=np.int64)
            new_gt_mask = np.stack(new_gt_mask, axis=0)
        else:
            new_gt_bbox = np.zeros((0, 4), dtype=np.float32)
            new_gt_labels = np.array([], dtype=np.int64)
        if new_gt_ignore_bbox:
            new_gt_ignore_bbox = np.array(new_gt_ignore_bbox, dtype=np.float32)
            new_gt_ignore_bbox = clip_box(new_gt_ignore_bbox, img_w=img_w, img_h=img_h)
            new_gt_ignore_mask = np.stack(new_gt_ignore_mask, axis=0)
        else:
            new_gt_ignore_bbox = np.zeros((0, 4), dtype=np.float32)

        img = img.tranpose(2, 0, 1)
        assert len(img.shape) == 3 and img.shape[0] == 3
        if return_angle:
            return img, new_gt_bbox, new_gt_labels, new_gt_ignore_bbox, \
            new_gt_mask, new_gt_ignore_mask, img_shape, pad_shape, angle
        else:
            return img, new_gt_bbox, new_gt_labels, new_gt_ignore_bbox, \
            new_gt_mask, new_gt_ignore_mask, img_shape, pad_shape


class RandomCropIC(object):
    def __init__(self,
                 crop_size=(800, 800),
                 pad=True):
        """ if the img shape > crop_size then crop.
        else pad to the crop size
        A new version, crop according to the box center.
        """
        self.crop_size = crop_size if crop_size is None or isinstance(crop_size, tuple) else \
            (crop_size, crop_size)
        self.pad = self.pad

        assert len(crop_size) == 2
        crop_size = tuple(map(
            lambda x: x + (32 - x % 32) if x % 32 != 0 else x, crop_size))
        self.crop_size = crop_size

    """ input: img, gt_box, gt_label, gt_mask, gt_ignore_box, gt_ignore_mask,
                img_shape, pad_shape, ratio(randomrotate)
        output: img, gt_box, gt_label, gt_mask, gt_ignore_box, gt_ignore_mask,
                img_shape, pad_shape, ratio(randomrotate)
    """
    def __call__(self, img, gt_bboxes, gt_labels, gt_masks,
                 gt_ignore_bboxes, gt_ignore_masks, img_shape, pad_shape):
        pad_h, pad_w = pad_shape[:2]
        img_h, img_w = img_shape[:2]
        crp_h, crp_w = self.crop_size
        # img [c, H, W] -> [H, W, c]
        img = img.transpose(1, 2, 0).copy()
        assert img.shape[0] == pad_h and img_shape[1] == pad_w
        if pad_h == crp_h and pad_w == crp_w:
            img = img.transpose(2, 0, 1)
            return img, gt_bboxes, gt_labels, gt_ignore_bboxes, \
                   gt_masks, gt_ignore_masks, img_shape, pad_shape

        tar_size = tuple(map(lambda x, y: x if x < y else y, img_shape[:2], self.crop_size[:2]))
        label = np.zeros(img_shape[:2], dtype=np.uint8)
        for idx in range(len(gt_masks)):
            mask = gt_masks[idx][:img_h, :img_w]
            label[mask > 0] = 1
        count = 50
        while count > 0:
            # randomly select a patch.
            if random.random() > 3.0 / 8.0 and np.max(label) > 0:
                tl = tuple(np.maximum(np.min(np.where(label > 0), axis=1) - tar_size, 0))
                br = tuple(np.maximum(np.max(np.where(label > 0), axis=1) - tar_size, 0))
                br[0] = min(br[0], img_h - tar_size[0])
                br[1] = min(br[1], img_w - tar_size[1])

                i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
                j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 1
            else:
                i = random.randint(img_h - tar_size[0]) if img_h > tar_size[0] else 0
                j = random.randint(img_w - tar_size[1]) if img_w > tar_size[1] else 0

            patch = np.array((int(j), int(i), int(j + tar_size[1]), int(i + tar_size[0])))
            overlaps = bbox_overlaps(
                patch.reshape(-1, 4), gt_bboxes.reshape(-1, 4), mode='iob').reshape(-1)
            if len(gt_masks) > 0 and (0 < overlaps.min() < 0.3):
                count -= 1
                continue
            else:
                break

        """ do not select the center..."""
        new_gt_bbox, new_gt_mask, new_gt_labels = [], [], []
        for idx in range(len(gt_masks)):
            mask = gt_masks[idx]
            mask = mask[i:i + tar_size[0], j:j + tar_size[1]]
            if np.max(mask) > 0:
                cnt = get_cnt_from_mask(mask)
                if cnt is not None:
                    box = get_rect_from_cnt(cnt)
                    new_gt_bbox.append(box.copy())
                    mask_p = cv2.copyMakeBorder(mask, 0, crp_h - tar_size[0],
                                                0, crp_w - tar_size[1], borderType=cv2.BORDER_CONSTANT, value=(0,))
                    new_gt_mask.append(mask_p.copy())
                    new_gt_labels.append(gt_labels[idx])
        new_gt_ignore_bbox = []
        new_gt_ignore_mask = []
        for idx in range(len(gt_ignore_masks)):
            mask = gt_ignore_masks[idx]
            mask = mask[i:i + tar_size[0], j:j + tar_size[1]]
            if np.max(mask) > 0:
                cnt = get_cnt_from_mask(mask)
                if cnt is not None:
                    box = get_rect_from_cnt(cnt)
                    new_gt_ignore_bbox.append(box.copy())
                    mask_p = cv2.copyMakeBorder(mask, 0, crp_h - tar_size[0], 0, crp_w - tar_size[1],
                                                borderType=cv2.BORDER_CONSTANT,
                                                value=(0,))
                    new_gt_ignore_mask.append(mask_p.copy())

        if new_gt_bbox:
            new_gt_bbox = np.array(new_gt_bbox, dtype=np.float32)
            new_gt_bbox = clip_box(new_gt_bbox, img_w=tar_size[1], img_h=tar_size[0])
            new_gt_labels = np.array(new_gt_labels, dtype=np.int64)
            new_gt_mask = np.stack(new_gt_mask, axis=0)
        else:
            new_gt_bbox = np.zeros((0, 4), dtype=np.float32)
            new_gt_labels = np.array([], dtype=np.int64)
        if new_gt_ignore_mask:
            new_gt_ignore_bbox = np.array(new_gt_ignore_bbox, dtype=np.float32)
            new_gt_ignore_bbox = clip_box(new_gt_ignore_bbox, img_w=tar_size[1], img_h=tar_size[0])
            new_gt_ignore_mask = np.stack(new_gt_ignore_mask, axis=0)
        else:
            new_gt_ignore_bbox = np.zeros((0, 4), dtype=np.float32)
        img_crp = img[i:i + tar_size[0], j:j + tar_size[1]]
        img_shape_new = img_crp.shape
        img_p = cv2.copyMakeBorder(img_crp, 0, crp_h - tar_size[0], 0, crp_w - tar_size[1], borderType=cv2.BORDER_CONSTANT,
                                   value=(0, 0, 0))
        pad_shape_new = img_p.shape
        img_p = img_p.transpose(2, 0, 1)
        assert len(img_p.shape) == 3 and img_p.shape[0] == 3
        return img_p, new_gt_bbox, new_gt_labels, new_gt_ignore_bbox, \
               new_gt_mask, new_gt_ignore_mask, img_shape_new, pad_shape_new


# add color jitter and photoMetric for data augmentation
class ExtraAugmentationIC(object):
    """ add for custom random crop method
        first for photoMetric.
    """
    def __init__(self,
                 random_rotate=None,
                 random_crop=None,
                 photo_metric_distortion=None):
        self.first_transforms = []
        self.transforms = []
        if photo_metric_distortion is not None:
            self.first_transforms.append(PhotoMetricDistortionIC(**photo_metric_distortion))
        if random_rotate is not None:
            self.transforms.append(advanceRandomRotationIC(**random_rotate))
        if random_crop is not None:
            self.transforms.append(RandomCropIC(**random_crop))

    def first_transform(self, img):
        if len(self.first_transforms) < 1:
            return img
        else:
            img = img.astype(np.float32)
            for transform in self.first_transforms:
                img = transform(img)
            return img

    def __call__(self, img, boxes, labels, masks, ignore_bboxes, ignore_masks, img_shape, pad_shape):
        img = img.astype(np.float32)
        for transform in self.transforms:
            img, boxes, labels, ignore_bboxes, masks, ignore_masks, img_shape, pad_shape = \
                transform(img=img, gt_bboxes=boxes, gt_labels=labels, gt_masks=masks, gt_ignore_bboxes=ignore_bboxes, gt_ignore_masks=ignore_masks, img_shape=img_shape, pad_shape=pad_shape)
        return img, boxes, labels, ignore_bboxes, masks, img_shape, pad_shape

