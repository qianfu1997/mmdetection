import argparse
from mmdet.core import tensor2imgs, get_classes
import torch
import numpy as np
import mmcv
import os
import cv2
import os.path as osp
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel
from mmdet import datasets
from mmdet.core import results2json, coco_eval
from mmdet.datasets import build_dataloader
from mmdet.models import build_detector, detectors
import pycocotools.mask as maskUtils
from ohem import m_NMS_py

def show_result(data,
                result,
                img_norm_cfg,
                index,
                outpath='./outputs/',
                dataset='IcdarDataset',
                score_thr=0.3):
    # result = bbox_result + segm_result
    # both of them are list
    # the result already completes the shape rescale
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    img_tensor = data['img'][0]
    img_metas  = data['img_meta'][0].data[0]
    imgs = tensor2imgs(img_tensor, **img_norm_cfg)
    assert len(imgs) == len(img_metas)

    if isinstance(dataset, str):
        class_names = get_classes(dataset)
    elif isinstance(dataset, (list, tuple)) or dataset is None:
        class_names = dataset
    else:
        raise TypeError(
            'dataset must be a valid dataset name or a sequence'
            ' of class names, not {}'.format(type(dataset)))

    for img, img_meta in zip(imgs, img_metas):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]
        # bboxes 鐨勯『搴忓拰mask椤哄簭鐩稿悓锛屼娇鐢ㄧ疆淇″害
        # 瓒呰繃闃堝€肩殑bbox鍙婂叾鐩稿簲鐨刴ask
        # vstack change the list of bboxes to
        # np.ndarray[N, 5], (x1, y1, x2, y2) and confidence
        bboxes = np.vstack(bbox_result)
        if segm_result is not None:
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            for i in inds:
                color_mask = np.random.randint(
                    # with a shape of (1, 3) RGB color
                    0, 256, (1, 3), dtype=np.uint8)
                # cocoapi use RLE to encode and decode mask
                # can use bool value to decode the mask
                mask = maskUtils.decode(segms[i]).astype(np.bool)
                img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
        # draw bounding boxes
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)]
        labels = np.concatenate(labels)
        if outpath is not None and osp.exists(outpath):
            mmcv.imshow_det_bboxes(
                img_show,
                bboxes,
                labels,
                class_names=class_names,
                score_thr=score_thr,
                show=False,
                out_file=osp.join(outpath, str(index) + '.jpg')
            )


def write_result_to_txt(image_name, bboxes, path='./submit/'):
    file_name = os.path.join(path, 'res_%s.txt' % (image_name))
    lines = []
    for idx, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox]
        line = "%d, %d, %d, %d, %d, %d, %d, %d\n" % tuple(values)
        lines.append(line)
    with open(file_name, 'w+') as f:
        f.writelines(lines)


def generate_text_mask(data,
                       result,
                       img_norm_cfg,
                       ori_shape,
                       filename,
                       outpath='./outputs/',
                       dataset='icdar',
                       score_thr=0.3):
    # result = bbox_result + segm_result
    # both of them are list
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    img_tensor = data['img'][0]
    img_metas = data['img_meta'][0].data[0]
    imgs = tensor2imgs(img_tensor, **img_norm_cfg)
    assert len(imgs) == len(img_metas)

    if isinstance(dataset, str):
        class_names = get_classes(dataset)
    elif isinstance(dataset, (list, tuple)) or dataset is None:
        class_names = dataset
    else:
        raise TypeError(
            'dataset must be a valid dataset name or a sequence'
            ' of class names, not {}'.format(type(dataset)))

    for img, img_meta in zip(imgs, img_metas):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]
        scale    = (ori_shape[0] * 1.0 / h, ori_shape[1] * 1.0 / w)
        # vstack change the list of bboxes to
        # np.ndarray[N, 5], (x1, y1, x2, y2) and confidence
        bboxes = np.vstack(bbox_result)
        quad_bboxes = []
        standard_bboxes = []
        bbox_probs = []
        if segm_result is not None:
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            num_instance = 0
            for i in inds:
                bbox_probs.append(bboxes[i, -1])
                instance_mask = np.zeros((h, w), dtype=np.uint8)
                color_mask = np.random.randint(
                    # with a shape of (1, 3) RGB color
                    0, 256, (1, 3), dtype=np.uint8)
                # cocoapi use RLE to encode and decode mask
                # can use bool value to decode the mask
                mask = maskUtils.decode(segms[i]).astype(np.bool)
                img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
                instance_mask[mask] = 1
                (k_label_num, k_label, k_stats, _) = cv2.connectedComponentsWithStats(instance_mask.astype(np.uint8), connectivity=4)
                lblareas = k_stats[:, cv2.CC_STAT_AREA]
                max_index = np.argmax(lblareas[1:]) + 1
                # for idx in range(1, k_label_num):
                points = np.array(np.where(k_label == max_index)).transpose((1, 0))[:, ::-1]
                rect = cv2.minAreaRect(points)
                std_rect = cv2.boxPoints(rect) * scale
                std_rect = std_rect.astype(np.int32)
                rect = cv2.boxPoints(rect).astype(np.int32)
                quad_bboxes.append(rect.reshape(-1))
                standard_bboxes.append(std_rect.reshape(-1))
        assert len(quad_bboxes) == len(standard_bboxes) == len(bbox_probs)
        quad_bboxes = m_NMS_py(quad_bboxes, bbox_probs)
        standard_bboxes = m_NMS_py(standard_bboxes, bbox_probs)
        img_show = cv2.resize(img_show, dsize=None, fx=scale[1], fy=scale[0])
        for box in standard_bboxes:
            cv2.drawContours(img_show, [box.reshape(4, 2)], -1, (0, 0, 255), 1)
        # for box in quad_bboxes:
        #     cv2.drawContours(img_show, [box.reshape(4, 2)], -1, (0, 0, 255), 4)
        cv2.imwrite(osp.join(outpath, filename), img_show)
        write_result_to_txt(filename.split('.')[0], standard_bboxes)
        # # draw bounding boxes
        # labels = [
        #     np.full(bbox.shape[0], i, dtype=np.int32)
        #     for i, bbox in enumerate(bbox_result)]
        # labels = np.concatenate(labels)
        # if outpath is not None and osp.exists(outpath):
        #     mmcv.imshow_det_bboxes(
        #         img_show,
        #         bboxes,
        #         labels,
        #         class_names=class_names,
        #         score_thr=score_thr,
        #         show=False,
        #         out_file=osp.join(outpath, filename)
        #     )


def single_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        imgname = dataset.img_infos[i]['filename']
        height, width = dataset.img_infos[i]['height'], dataset.img_infos[i]['width']
        shape = [height, width]
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            # model.module.show_result(data, result, dataset.img_norm_cfg,
            #                          dataset.CLASSES)
            # show_result(data, result, dataset.img_norm_cfg, filename=imgname)
            generate_text_mask(data, result, dataset.img_norm_cfg, ori_shape=shape, filename=imgname)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()

    return results


def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))
    if args.gpus == 1:
        model = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        load_checkpoint(model, args.checkpoint)
        model = MMDataParallel(model, device_ids=[0])

        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            num_gpus=1,
            dist=False,
            shuffle=False)
        outputs = single_test(model, data_loader, args.show)
    else:
        model_args = cfg.model.copy()
        model_args.update(train_cfg=None, test_cfg=cfg.test_cfg)
        model_type = getattr(detectors, model_args.pop('type'))
        outputs = parallel_test(
            model_type,
            model_args,
            args.checkpoint,
            dataset,
            _data_func,
            range(args.gpus),
            workers_per_gpu=args.proc_per_gpu)

    # if args.out:
    #     print('writing results to {}'.format(args.out))
    #     mmcv.dump(outputs, args.out)
    #     eval_types = args.eval
    #     if eval_types:
    #         print('Starting evaluate {}'.format(' and '.join(eval_types)))
    #         if eval_types == ['proposal_fast']:
    #             result_file = args.out
    #             coco_eval(result_file, eval_types, dataset.coco)
    #         else:
    #             if not isinstance(outputs[0], dict):
    #                 result_file = args.out + '.json'
    #                 results2json(dataset, outputs, result_file)
    #                 coco_eval(result_file, eval_types, dataset.coco)
    #             else:
    #                 for name in outputs[0]:
    #                     print('\nEvaluating {}'.format(name))
    #                     outputs_ = [out[name] for out in outputs]
    #                     result_file = args.out + '.{}.json'.format(name)
    #                     results2json(dataset, outputs_, result_file)
    #                     coco_eval(result_file, eval_types, dataset.coco)


if __name__ == '__main__':
    main()