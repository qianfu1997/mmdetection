import mmcv
import numpy as np
from numpy import random
import cv2

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps


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


# def get_rect_from_cnt(cnt):
#     # cnt_points = np.array(cnt).reshape(-1, 2).astype(np.int64)
#     # box = np.zeros(4)
#     # box[:2] = np.min(cnt_points, axis=0)
#     # box[2:] = np.max(cnt_points, axis=0)
#     rect = cv2.minAreaRect(cnt)   # ori
#     rect = cv2.boxPoints(rect)
#     rect = np.array(np.int0(rect)).reshape(-1, 2).astype(np.int64)
#     box = np.zeros(4)
#     box[:2] = np.min(rect, axis=0)
#     box[2:] = np.max(rect, axis=0)
#     return box

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


class RandomRotationIC(object):
    def __init__(self,
                 max_angle=10,
                 ver_flip_ratio=0.0,
                 angle_flip=0):
        self.max_angle = max_angle
        self.ver_flip_ratio = ver_flip_ratio
        self.angle_flip = angle_flip

        """ 
        """

    def __call__(self, img, gt_bboxes, gt_labels, gt_masks, gt_ignore_bboxes,
                 gt_ignore_masks, img_shape, pad_shape):
        """ keep the img_shape and pad_shape still """
        pad_h, pad_w, _ = pad_shape
        img_h, img_w, _ = img_shape
        # img: [3, H, W] -> [H, W, 3]
        img = img.transpose(1, 2, 0).copy()
        assert img.shape[0] == pad_h and img.shape[1] == pad_w
        # select a random rotate, and generate bbox from masks.
        angle = np.random.rand() * 2 * self.max_angle - self.max_angle
        random_anchor = [90, 270]
        angle += random.choice(random_anchor) if \
            bool(self.angle_flip > 0 and np.random.rand() < self.angle_flip) else 0

        # first ver_flip
        ver_flip_f = bool(self.ver_flip_ratio > 0 and np.random.rand() < self.ver_flip_ratio)
        rotation_matrix = cv2.getRotationMatrix2D((int(img_w / 2), int(img_h / 2)), angle, 1)
        img_rotation = img[:img_h, :img_w, :].copy()
        if ver_flip_f:
            img_rotation = np.flip(img_rotation, 0)
        img_rotation = cv2.warpAffine(      # do not use reflect.
            img_rotation, rotation_matrix, (img_w, img_h))
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
            mask_rotation = cv2.warpAffine(     # do not reflect.
                mask_rotation, rotation_matrix, (img_w, img_h))
            if np.max(mask_rotation) > 0:
                cnt = get_cnt_from_mask(mask_rotation)
                if cnt is not None:
                    box = get_rect_from_cnt(cnt)
                    new_gt_ignore_bbox.append(box.copy())
                    new_mask = gt_ignore_masks[idx].copy()
                    new_mask[:img_h, :img_w] = mask_rotation.copy()
                    new_gt_ignore_mask.append(new_mask)

        if len(new_gt_bbox) > 0:
            new_gt_bbox = np.array(new_gt_bbox, dtype=np.float32)
            new_gt_bbox = clip_box(new_gt_bbox, img_w=img_w, img_h=img_h)
            new_gt_labels = np.array(new_gt_labels, dtype=np.int64)
            new_gt_mask = np.stack(new_gt_mask, axis=0)
        else:
            new_gt_bbox = np.zeros((0, 4), dtype=np.float32)
            new_gt_labels = np.array([], dtype=np.int64)
        if len(new_gt_ignore_bbox) > 0:
            new_gt_ignore_bbox = np.array(new_gt_ignore_bbox, dtype=np.float32)
            new_gt_ignore_bbox = clip_box(new_gt_ignore_bbox, img_w=img_w, img_h=img_h)
            new_gt_ignore_mask = np.stack(new_gt_ignore_mask, axis=0)
        else:
            new_gt_ignore_bbox = np.zeros((0, 4), dtype=np.float32)

        # img [H, W, 3] -> [3, H, W]
        img = img.transpose(2, 0, 1)
        assert len(img.shape) == 3 and img.shape[0] == 3
        return img, new_gt_bbox, new_gt_labels, new_gt_ignore_bbox,\
               new_gt_mask, new_gt_ignore_mask, img_shape, pad_shape


class RandomCropIC(object):
    def __init__(self,
                 crop_size=(800, 800),
                 pad=True):
        """ if the img shape > crop_size then crop.
        else pad to the crop size """
        # 1: return ori img
        self.crop_size = crop_size if crop_size is None or isinstance(crop_size, tuple) else \
            (crop_size, crop_size)
        self.pad = pad

        """ check the crop_size """
        h, w = self.crop_size
        h = h + (32 - h % 32) if h % 32 != 0 else h
        w = w + (32 - w % 32) if w % 32 != 0 else w
        self.crop_size = (h, w)

    def __call__(self, img, gt_bboxes, gt_labels, gt_masks, gt_ignore_bboxes,
                 gt_ignore_masks, img_shape, pad_shape):
        pad_h, pad_w, _ = pad_shape
        img_h, img_w, _ = img_shape
        crp_h, crp_w = self.crop_size
        # img [c, H, W] -> [H, W, c]
        img = img.transpose(1, 2, 0).copy()
        assert img.shape[0] == pad_h and img.shape[1] == pad_w
        if pad_h == crp_h and pad_w == crp_w:
            img = img.transpose(2, 0, 1)
            return img, gt_bboxes, gt_labels, gt_ignore_bboxes, \
                   gt_masks, gt_ignore_masks, img_shape, pad_shape

        tar_h = crp_h if img_h > crp_h else img_h
        tar_w = crp_w if img_w > crp_w else img_w

        label = np.zeros((img_h, img_w), dtype=np.uint8)
        for idx in range(len(gt_masks)):
            mask = gt_masks[idx][:img_h, :img_w]
            label[mask > 0] = 1
        # randomly select a region
        if random.random() > 3.0 / 8.0 and np.max(label) > 0:
            tl = np.min(np.where(label > 0), axis=1) - (tar_h, tar_w)
            tl[tl < 0] = 0
            br = np.max(np.where(label > 0), axis=1) - (tar_h, tar_w)
            br[br < 0] = 0
            br[0] = min(br[0], img_h - tar_h)
            br[1] = min(br[1], img_w - tar_w)

            i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
            j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
        else:
            i = random.randint(0, img_h - tar_h) if img_h - tar_h > 0 else 0
            j = random.randint(0, img_w - tar_w) if img_w - tar_w > 0 else 0

        new_gt_bbox, new_gt_mask, new_gt_labels = [], [], []
        for idx in range(len(gt_masks)):
            """ find the cc, and generate box for cc. """
            mask = gt_masks[idx]
            mask = mask[i:i + tar_h, j:j + tar_w]
            if np.max(mask) > 0:
                cnt = get_cnt_from_mask(mask)
                if cnt is not None:
                    # generate the bounding boxes
                    box = get_rect_from_cnt(cnt)
                    new_gt_bbox.append(box.copy())
                    # pad the masks
                    mask_p = cv2.copyMakeBorder(mask, 0, crp_h - tar_h, 0, crp_w - tar_w, borderType=cv2.BORDER_CONSTANT,
                                               value=(0,))
                    new_gt_mask.append(mask_p.copy())
                    new_gt_labels.append(gt_labels[idx])

        new_gt_ignore_bbox = []
        new_gt_ignore_mask = []
        for idx in range(len(gt_ignore_masks)):
            mask = gt_ignore_masks[idx]
            mask = mask[i:i + tar_h, j:j + tar_w]   # mask crop.
            if np.max(mask) > 0:
                cnt = get_cnt_from_mask(mask)
                if cnt is not None:
                    box = get_rect_from_cnt(cnt)
                    new_gt_ignore_bbox.append(box.copy())
                    mask_p = cv2.copyMakeBorder(mask, 0, crp_h - tar_h, 0, crp_w - tar_w,
                                                borderType=cv2.BORDER_CONSTANT,
                                                value=(0,))
                    new_gt_ignore_mask.append(mask_p.copy())

        if len(new_gt_bbox) > 0:
            new_gt_bbox = np.array(new_gt_bbox, dtype=np.float32)
            new_gt_bbox = clip_box(new_gt_bbox, img_w=tar_w, img_h=tar_h)
            new_gt_labels = np.array(new_gt_labels, dtype=np.int64)
            new_gt_mask = np.stack(new_gt_mask, axis=0)
        else:
            new_gt_bbox = np.zeros((0, 4), dtype=np.float32)
            new_gt_labels = np.array([], dtype=np.int64)
        if len(new_gt_ignore_bbox) > 0:
            new_gt_ignore_bbox = np.array(new_gt_ignore_bbox, dtype=np.float32)
            new_gt_ignore_bbox = clip_box(new_gt_ignore_bbox, img_w=tar_w, img_h=tar_h)
            new_gt_ignore_mask = np.stack(new_gt_ignore_mask, axis=0)
        else:
            new_gt_ignore_bbox = np.zeros((0, 4), dtype=np.float32)

        # crop and pad the img
        img_crp = img[i:i + tar_h, j:j + tar_w, :]
        img_shape_new = img_crp.shape
        img_p = cv2.copyMakeBorder(img_crp, 0, crp_h - tar_h, 0, crp_w - tar_w, borderType=cv2.BORDER_CONSTANT,
                                   value=(0, 0, 0))
        pad_shape_new = img_p.shape
        # img_shape_new = img_p.shape
        img_p = img_p.transpose(2, 0, 1)
        assert len(img_p.shape) == 3 and img_p.shape[0] == 3
        return img_p, new_gt_bbox, new_gt_labels, new_gt_ignore_bbox,\
               new_gt_mask, new_gt_ignore_mask, img_shape_new, pad_shape_new


class ExtraAugmentation(object):

    def __init__(self,
                 photo_metric_distortion=None,
                 expand=None,
                 random_crop=None):
        self.transforms = []
        if photo_metric_distortion is not None:
            self.transforms.append(
                PhotoMetricDistortion(**photo_metric_distortion))
        if expand is not None:
            self.transforms.append(Expand(**expand))
        if random_crop is not None:
            self.transforms.append(RandomCrop(**random_crop))

    def __call__(self, img, boxes, labels):
        img = img.astype(np.float32)
        for transform in self.transforms:
            img, boxes, labels = transform(img, boxes, labels)
        return img, boxes, labels


# class ExtraAugmentationIC(object):
#     """ add for custom random crop method.
#         first random rescale the img, and store
#     """
#
#     def __init__(self,
#                  photo_metric_distortion=None,
#                  random_rotate=None,
#                  random_crop=None):
#         self.transforms = []
#         if random_rotate is not None:
#             self.transforms.append(RandomRotationIC(**random_rotate))
#         if random_crop is not None:
#             self.transforms.append(RandomCropIC(**random_crop))
#
#     def __call__(self, img, boxes, labels, masks,
#                  ignore_bboxes, ignore_masks, img_shape, pad_shape):
#         img = img.astype(np.float32)
#         for transform in self.transforms:
#             img, boxes, labels, ignore_bboxes, masks, ignore_masks, img_shape, pad_shape = \
#                 transform(img, boxes, labels, masks, ignore_bboxes,
#                           ignore_masks, img_shape, pad_shape)
#         return img, boxes, labels, ignore_bboxes, \
#                masks, img_shape, pad_shape


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
            self.transforms.append(RandomRotationIC(**random_rotate))
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

