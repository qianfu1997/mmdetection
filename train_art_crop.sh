# CUDA_VISIBLE_DEVICES=1,2,0 bash ./tools/m_dist_train.sh /home/data1/sxg/IC19/mmdetection-master/configs/art/mask_rcnn_r50_fpn_1x_ic17.py 3 --work_dir ./work_dirs/art_mrcnn_r50_test_f/
# CUDA_VISIBLE_DEVICES=2,3 bash ./tools/m_dist_train.sh /home/data1/sxg/IC19/mmdetection-master/configs/art/mask_rcnn_r50_fpn_1x_ic17_v2.py 2 --work_dir ./work_dirs/art_mrcnn_r50_test_b/
# CUDA_VISIBLE_DEVICES=7,6,4,5 bash ./tools/m_dist_train.sh /home/data1/sxg/IC19/mmdetection-master/configs/art_crop/mask_rcnn_r50_fpn_1x_ic17_msda.py 4 --work_dir ./work_dirs/art_crop_mrcnn_r50_test_msda_b/
# CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/m_dist_train.sh /home/data1/sxg/IC19/mmdetection-master/configs/art_crop/mask_rcnn_r50_fpn_1x_ic17_msda_2xepochs.py 4 --work_dir ./work_dirs/art_crop_mrcnn_r50_test_msda_2xepochs_rrvfaf/

# CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/m_dist_train.sh /home/data1/sxg/IC19/mmdetection-master/configs/art_crop/mask_rcnn_r50_fpn_1x_ic17_msda.py 4 --work_dir ./work_dirs/art_crop_mrcnn_r50_test_msda_c/	# for scale around 928

# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh /home/data1/sxg/IC19/mmdetection-master/configs/art_crop/mask_rcnn_r50_fpn_1x_ic17_msda_shortv2.py 4  --work_dir ./work_dirs/art_crop_mrcnn_r50_test_msda_d/	# for scale around 1024

# CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_train.sh /home/data1/sxg/IC19/mmdetection-master/configs/art_crop/mask_rcnn_r50_fpn_1x_ic17_msda_shortv3.py 4 --work_dir ./work_dirs/art_crop_mrcnn_r50_test_msda_e/	# for scale around 1024

# CUDA_VISIBLE_DEVICES=1,2,3,0 bash ./tools/m_dist_train.sh ./configs/art_crop/mask_rcnn_r50_fpn_1x_ic17_PMTD_msda.py 4 --work_dir ./work_dirs/art_crop_mrcnn_r50_fpn_1x_ic17_PMTD_msda_a/   # data augmentation using PMTD settings.

# CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/m_dist_train.sh /home/data1/sxg/IC19/mmdetection-master/configs/art_crop/mask_rcnn_r50_fpn_1x_ic17_multiroiextractor_msda.py 4 --work_dir ./work_dirs/art_crop_mrcnn_r50_fpn_1x_ic17_multiroiextractor_msda_a/	# for multi_roi_extractor


# CUDA_VISIBLE_DEVICES=1,2,3,0 bash ./tools/m_dist_train.sh ./configs/art_crop/mask_rcnn_r50_fpn_1x_ic17_msda_pan.py 4 --work_dir ./work_dirs/art_crop_mrcnn_r50_fpn_1x_ic17_msda_pan_b/   # pan setttings

# CUDA_VISIBLE_DEVICES=3,2,1,0 bash ./tools/m_dist_train.sh ./configs/art_crop/mask_rcnn_r50_fpn_1x_ic17_msda.py 4 --work_dir ./work_dirs/art_crop_mrcnn_r50_fpn_1x_ic17_msda_rerun/   # pan setttings rerun

# CUDA_VISIBLE_DEVICES=7,6,5,4 bash ./tools/m_dist_train.sh ./configs/art_crop/mask_rcnn_r50_fpn_1x_ic17_msda_deepan.py 4 --work_dir ./work_dirs/art_crop_mrcnn_r50_fpn_1x_ic17_msda_deepan_a/ --seed 123456 # pan setttings rerun
 
# CUDA_VISIBLE_DEVICES=2,3,4,5 bash ./tools/m_dist_train.sh ./configs/art_crop/mask_rcnn_r50_fpn_2x_ic17_msda_2xanchorscales.py 4 --work_dir ./work_dirs/art_crop_mrcnn_r50_fpn_2x_ic17_msda_2xanchorscales_b/ --seed 123456 # baseline setting with anchor-base-scale 8, 16, pad 4


# CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/m_dist_train.sh ./configs/art_crop/mask_rcnn_r50_fpn_2x_ic17_msda_pan.py 4 --work_dir ./work_dirs/art_crop_mrcnn_r50_fpn_2x_ic17_msda_pan_a/ --seed 123456 # pan with 2x epochs

# CUDA_VISIBLE_DEVICES=4,5,6,7,0,1,2,3 bash ./tools/dist_train.sh ./configs/art_crop/mask_rcnn_r50_fpn_2x_ic17_msda_psppan.py 8 --work_dir ./work_dirs/art_crop_mrcnn_r50_fpn_2x_ic17_msda_psppan_a/ --seed 123456 # pan with 2x epoch

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./tools/m_dist_train.sh ./configs/art_crop/mask_rcnn_r50_fpn_2x_ic17_msda_pan.py 8 --work_dir ./work_dirs/art_crop_mrcnn_r50_fpn_2x_ic17_msda_pan_b/  # pan with 2x epochs and different initialization and 8 GPUs.
