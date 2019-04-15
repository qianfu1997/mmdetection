import multiprocessing

import torch

import mmcv
from .checkpoint import load_checkpoint

import cv2
import os.path as osp
import json
import numpy as np


def worker_func(model_cls, model_kwargs, checkpoint, dataset, data_func,
                gpu_id, idx_queue, result_queue):
    model = model_cls(**model_kwargs)
    load_checkpoint(model, checkpoint, map_location='cpu')
    torch.cuda.set_device(gpu_id)
    model.cuda()
    model.eval()
    with torch.no_grad():
        while True:
            idx = idx_queue.get()
            data = dataset[idx]
            result = model(**data_func(data, gpu_id))
            result_queue.put((idx, result))


def parallel_test(model_cls,
                  model_kwargs,
                  checkpoint,
                  dataset,
                  data_func,
                  gpus,
                  workers_per_gpu=1):
    """Parallel testing on multiple GPUs.

    Args:
        model_cls (type): Model class type.
        model_kwargs (dict): Arguments to init the model.
        checkpoint (str): Checkpoint filepath.
        dataset (:obj:`Dataset`): The dataset to be tested.
        data_func (callable): The function that generates model inputs.
        gpus (list[int]): GPU ids to be used.
        workers_per_gpu (int): Number of processes on each GPU. It is possible
            to run multiple workers on each GPU.

    Returns:
        list: Test results.
    """
    ctx = multiprocessing.get_context('spawn')
    idx_queue = ctx.Queue()
    result_queue = ctx.Queue()
    num_workers = len(gpus) * workers_per_gpu
    workers = [
        ctx.Process(
            target=worker_func,
            args=(model_cls, model_kwargs, checkpoint, dataset, data_func,
                  gpus[i % len(gpus)], idx_queue, result_queue))
        for i in range(num_workers)
    ]
    for w in workers:
        w.daemon = True
        w.start()

    for i in range(len(dataset)):
        idx_queue.put(i)

    results = [None for _ in range(len(dataset))]
    prog_bar = mmcv.ProgressBar(task_num=len(dataset))
    for _ in range(len(dataset)):
        idx, res = result_queue.get()
        results[idx] = res
        prog_bar.update()
    print('\n')
    for worker in workers:
        worker.terminate()

    return results


def worker_func_art(model_cls, model_kwargs, checkpoint, dataset, data_func,
                    gpu_id, idx_queue, result_queue, post_processor,
                    img_prefix, show=True, show_path=None):
    """ store the img_name, img_shape, ori_shape in data_queue
        return single_pred_results.
    """
    model = model_cls(**model_kwargs)
    load_checkpoint(model, checkpoint, map_location='cpu')
    torch.cuda.set_device(gpu_id)
    model.cuda()
    model.eval()
    with torch.no_grad():
        while True:
            idx = idx_queue.get()
            data = dataset[idx]
            data_dict = data_func(data, gpu_id)
            img_metas = data_dict['img_meta'][0]
            img_metas_0 = img_metas[0]
            filename = img_metas_0['filename']
            img_name = osp.splitext(filename)[0].replace('gt_', 'res_')
            result = model(**data_func(data, gpu_id))
            if isinstance(result, tuple):
                bbox_result, segm_result = result
            else:
                bbox_result, segm_result = result, None
            vs_bbox_result = np.vstack(bbox_result)
            if segm_result is None:
                pred_bboxes, pred_bbox_scores = [], []
            else:
                if isinstance(segm_result, tuple):
                    """ changed """
                    segm_scores = mmcv.concat_list(segm_result[-1])
                    segms = mmcv.concat_list(segm_result[0])
                else:
                    segm_scores = np.asarray(vs_bbox_result[:, -1])
                    segms = mmcv.concat_list(segm_result)

                pred_bboxes, pred_bbox_scores = post_processor.process(segms, segm_scores,
                                                                       mask_shape=img_metas_0['ori_shape'],
                                                                       scale_factor=(1.0, 1.0))
        # save the results.
            single_pred_results = []
            for pred_bbox, pred_bbox_score in zip(pred_bboxes, pred_bbox_scores):
                pred_bbox = np.asarray(pred_bbox).reshape((-1, 2)).astype(np.int32)
                pred_bbox = pred_bbox.tolist()
                single_bbox_dict = {
                    "points": pred_bbox,
                    "confidence": float(pred_bbox_score),
                }
                single_pred_results.append(single_bbox_dict)
            pred_result = {
                "img_name": img_name,
                "single_pred_results": single_pred_results
            }
            result_queue.put((idx, pred_result))

            if show:
                img = cv2.imread(osp.join(img_prefix, filename))
                for idx in range(len(single_pred_results)):
                    bbox = np.asarray(single_pred_results[idx]["points"]).reshape(-1, 2).astype(np.int64)
                    cv2.drawContours(img, [bbox], -1, (0, 255, 0), 2)
                cv2.imwrite(osp.join(show_path, filename), img)


def parallel_test_art(model_cls,
                      model_kwargs,
                      checkpoint,
                      dataset,
                      data_func,
                      gpus,
                      post_processor,
                      save_json_file,
                      workers_per_gpu=1,        # in configs there are 2.
                      show=True,
                      show_path=None):
    """ use the worker_func to generate results
        and do post-processing for each result.
    """
    """Parallel testing on multiple GPUs.

        Args:
            model_cls (type): Model class type.
            model_kwargs (dict): Arguments to init the model.
            checkpoint (str): Checkpoint filepath.
            dataset (:obj:`Dataset`): The dataset to be tested.
            data_func (callable): The function that generates model inputs.
            gpus (list[int]): GPU ids to be used.
            workers_per_gpu (int): Number of processes on each GPU. It is possible
                to run multiple workers on each GPU.

        Returns:
            list: Test results.
    """
    ctx = multiprocessing.get_context('spawn')
    idx_queue = ctx.Queue()
    result_queue = ctx.Queue()
    num_workers = len(gpus) * workers_per_gpu
    img_prefix = dataset.img_prefix
    workers = [
        ctx.Process(
            target=worker_func_art,
            args=(model_cls, model_kwargs, checkpoint, dataset, data_func,
                  gpus[i % len(gpus)], idx_queue, result_queue,
                  post_processor,
                  img_prefix,
                  show, show_path))
        for i in range(num_workers)
    ]
    for w in workers:
        w.daemon = True
        w.start()

    for i in range(len(dataset)):
        idx_queue.put(i)

    results = [None for _ in range(len(dataset))]
    imgs_bboxes_results = {}
    prog_bar = mmcv.ProgressBar(task_num=len(dataset))
    for _ in range(len(dataset)):
        idx, res = result_queue.get()
        results[idx] = res

        img_name = res['img_name']
        single_pred_results = res['single_pred_results']
        imgs_bboxes_results[img_name] = single_pred_results
        prog_bar.update()
    print('\n')
    for worker in workers:
        worker.terminate()
    with open(save_json_file, 'w+', encoding='utf-8') as f:
        json.dump(imgs_bboxes_results, f)
    return results




