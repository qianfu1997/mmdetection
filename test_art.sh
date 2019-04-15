# CUDA_VISIBLE_DEVICES=7 python ./tools/art_test.py /home/data1/sxg/IC19/mmdetection-master/configs/art/cascade_mask_rcnn_r50_fpn_1x_ic17_v2.py /home/data1/sxg/IC19/mmdetection-master/work_dirs/art_cascade_mrcnn_r50_test_a/epoch_24.pth /home/data1/sxg/IC19/mmdetection-master/configs/evaluation/art/art_eval_setting.json --gpus 1 --save_json /home/data1/sxg/IC19/mmdetection-master/submit/art/submit.json --show_path /home/data1/sxg/IC19/mmdetection-master/visualization/eval_result/art/ --gt_ann_path /home/data1/IC19/ArT/annotations/sp_val_eval_labels.json  --show --debug

# CUDA_VISIBLE_DEVICES=0 python ./tools/art_test.py /home/data1/sxg/IC19/mmdetection-master/configs/art/mask_rcnn_r50_fpn_1x_ic17_dice.py /home/data1/sxg/IC19/mmdetection-master/work_dirs/art_mrcnn_r50_test_dice_a/epoch_24.pth /home/data1/sxg/IC19/mmdetection-master/configs/evaluation/art/art_eval_setting.json --gpus 1 --save_json /home/data1/sxg/IC19/mmdetection-master/submit/art/submit.json --show_path /home/data1/sxg/IC19/mmdetection-master/visualization/eval_result/art/ --show


# CUDA_VISIBLE_DEVICES=5 python ./tools/art_test.py /home/data1/sxg/IC19/mmdetection-master/configs/art_crop/mask_rcnn_r50_fpn_1x_ic17_msda.py /home/data1/sxg/IC19/mmdetection-master/work_dirs/art_crop_mrcnn_r50_test_msda_b/epoch_24.pth /home/data1/sxg/IC19/mmdetection-master/configs/evaluation/art_crop/art_eval_setting.json --gpus 1 --save_json /home/data1/sxg/IC19/mmdetection-master/submit/art/submit.json --show_path /home/data1/sxg/IC19/mmdetection-master/visualization/eval_result/art/ --gt_ann_path /home/data1/IC19/ArT/annotations/sp_val_eval_labels.json  --show --debug


CUDA_VISIBLE_DEVICES=6 python ./tools/art_test.py /home/data1/sxg/IC19/mmdetection-master/configs/art_crop/mask_rcnn_r50_fpn_1x_ic17_msda_pan.py /home/data1/sxg/IC19/mmdetection-master/work_dirs/art_crop_mrcnn_r50_fpn_1x_ic17_msda_pan_rerun/epoch_23.pth /home/data1/sxg/IC19/mmdetection-master/configs/evaluation/art_crop/art_eval_setting.json --gpus 1 --save_json /home/data1/sxg/IC19/mmdetection-master/submit/art/submit.json --show_path /home/data1/sxg/IC19/mmdetection-master/visualization/eval_result/art/ --show