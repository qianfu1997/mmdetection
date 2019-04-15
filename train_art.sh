# CUDA_VISIBLE_DEVICES=1,2,0 bash ./tools/m_dist_train.sh /home/data1/sxg/IC19/mmdetection-master/configs/art/mask_rcnn_r50_fpn_1x_ic17.py 3 --work_dir ./work_dirs/art_mrcnn_r50_test_f/
# CUDA_VISIBLE_DEVICES=2,3 bash ./tools/m_dist_train.sh /home/data1/sxg/IC19/mmdetection-master/configs/art/mask_rcnn_r50_fpn_1x_ic17_v2.py 2 --work_dir ./work_dirs/art_mrcnn_r50_test_b/
# CUDA_VISIBLE_DEVICES=5,4,6,7 bash ./tools/m_dist_train.sh /home/data1/sxg/IC19/mmdetection-master/configs/art/cascade_mask_rcnn_r50_fpn_1x_ic17_v2.py 4 --work_dir ./work_dirs/art_cascade_mrcnn_r50_test_b/
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh /home/data1/sxg/IC19/mmdetection-master/configs/art/mask_rcnn_r50_fpn_1x_ic17_dice.py 4 --work_dir ./work_dirs/art_mrcnn_r50_test_dice_a/

# CUDA_VISIBLE_DEVICES=3,2,1,0 bash ./tools/m_dist_train.sh /home/data1/sxg/IC19/mmdetection-master/configs/art/mask_rcnn_r50_fpn_2x_gn_ic17_deepHead.py 4 --work_dir ./work_dirs/art_mrcnn_r50_test_deepHead_a/

# CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_train.sh /home/data1/sxg/IC19/mmdetection-master/configs/art/mask_rcnn_r50_fpn_1x_ic17_ohem.py 4 --work_dir ./work_dirs/art_mrcnn_r50_test_ohem_a/

# CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/m_dist_train.sh /home/data1/sxg/IC19/mmdetection-master/configs/art/mask_rcnn_r50_fpn_2x_gn_ic17_deepHead.py 4 --work_dir ./work_dirs/art_mrcnn_r50_test_deepHead_b/

# CUDA_VISIBLE_DEVICES=3,2,1,0 bash ./tools/m_dist_train.sh /home/data1/sxg/IC19/mmdetection-master/configs/art/mask_rcnn_r50_fpn_2x_gn_ic17_deepHead.py 4 --work_dir ./work_dirs/art_mrcnn_r50_test_deepHead_c/ --resume_from ./work_dirs/art_mrcnn_r50_test_deepHead_c/epoch_6.pth  # without gn

# CUDA_VISIBLE_DEVICES=7,6,5,4 bash ./tools/dist_train.sh /home/data1/sxg/IC19/mmdetection-master/configs/art/mask_rcnn_r50_fpn_1x_ic17_2xmask.py 4 --work_dir ./work_dirs/art_mrcnn_r50_test_2xmask_b/   # 8 conv + 2x mask

CUDA_VISIBLE_DEVICES=7,6,5,4 bash ./tools/dist_train.sh ./configs/art/mask_rcnn_r50_fpn_2x_gn_ic17_v2.py 4 --work_dir ./work_dirs/art_mrcnn_r50_fpn_2x_gn_ic17_v2_a/    # gn in backbone/neck/mask_head without deep mask head

