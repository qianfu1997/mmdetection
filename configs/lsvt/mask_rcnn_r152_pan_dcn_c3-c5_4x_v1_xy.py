# model settings
# GPUs:8
# a very deep model(152 layers) with dcn and pan
model = dict(
    type='MaskRCNN',
    pretrained='modelzoo://resnet152',
    backbone=dict(
        type='ResNet',
        depth=152,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        dcn=dict(
            modulated=False,
            deformable_groups=1,
            fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='PAN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 0.2, 1, 2, 5, 8],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        use_sigmoid_cls=True),
    bbox_roi_extractor=dict(
        type='MultiRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=2,  # for text detection, there are only two classes text/non-text
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False),
    mask_roi_extractor=dict(
        type='MultiRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    mask_head=dict(
        type='FCNMaskHeadPAN',
        num_convs=4,
        in_channels=256,
        conv_out_channels=256,
        num_classes=2  # for text detection, there are only two classes text/non-text
    ))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        smoothl1_beta=1 / 9.0,
        debug=False),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='OHEMSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        mask_size=28,
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_thr=0.5),
        max_per_img=100,  # max number of boxes to keep
        mask_thr_binary=0.5))
# dataset settings
dataset_type = 'ArtCropDataset'
data_root = 'data/ArT/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/sp_train_full_labels_new.json',
        img_prefix=data_root + 'sp_train_full_images_new/',
        img_scale=[(3200, 560), (3200, 1040)],
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=True,
        with_crowd=True,
        with_label=True,
        extra_aug=dict(
            random_rotate=dict(
                max_angle=5,
                ver_flip_ratio=0.0,  # the flip ratio
                angle_flip=0),  # default: False
            random_crop=dict(
                crop_size=(800, 800),
                pad=True),
            photo_metric_distortion=dict(
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18)
        ),
        resize_keep_ratio=True,
        test_mode=False,
        # img_scale_mode='range'
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/sp_val_full_labels_new.json',
        img_prefix=data_root + 'sp_val_full_images_new/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=True,
        with_crowd=True,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=None,
        # img_prefix='/home/xieenze/R4D/whai_workspace/mmdetection-art/data/LSVT/weak/train_weak_images/',
        # img_prefix='/home/xieenze/DataSet/LSVT/sp_val_full_images/',
        img_prefix='/home/xieenze/DataSet/LSVT/sp_val_full_images/',
        img_scale=(3200, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',  # lr updated policy can be 'cosine'
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[32, 43]
)
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        #dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 48
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/home/data1/xdata/mmdet_lsvt_work_dirs/mask_rcnn_r152_pan_dcn_c3-c5_4x_v1'
load_from = None
resume_from = None
workflow = [('train', 1)]
