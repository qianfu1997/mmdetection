#
# @author:charlotte.Song
# @file: test_best_epoch.py
# @Date: 2019/3/23 11:33
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import argparse
import subprocess


program_root = '/home/data1/sxg/IC19/mmdetection-master/'
cd_root_cmd = 'cd %s'%program_root

process = []
best_f = 0
best_ep = 0
best_line = ''

def parse_arg():
    parser = argparse.ArgumentParser('automatically test best epoch')
    parser.add_argument('--train_config', type=str, default=None)
    parser.add_argument('--eval_config', type=str, default=None)
    parser.add_argument('--work_dir', type=str, default=None)
    parser.add_argument('--min_epoch', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=0)
    args = parser.parse_args()
    return args


def art_eval(ep, args):
    run_cmd = 'CUDA_VISIBLE_DEVICES=4,5,6,7 ' \
              'python ./tools/art_test.py /home/data1/sxg/IC19/mmdetection-master/configs/%s /home/data1/sxg/IC19/mmdetection-master/work_dirs/%s/epoch_%d.pth /home/data1/sxg/IC19/mmdetection-master/configs/evaluation/%s --gpus 4 --save_json /home/data1/sxg/IC19/mmdetection-master/submit/art/submit.json --show_path /home/data1/sxg/IC19/mmdetection-master/visualization/eval_result/art/ --show'%(
        args.train_config, args.work_dir, ep, args.eval_config)
    run_cmd = cd_root_cmd + " && " + run_cmd
    print('ep: %d'% ep)
    p = subprocess.Popen(run_cmd, shell=True, stderr=subprocess.STDOUT)
    p.wait()

    eval_cmd = cd_root_cmd + '&&' + 'sh art_eval.sh'
    p = subprocess.Popen(eval_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p.wait()
    f = 0
    line = ''
    for line in iter(p.stdout.readline, ''):
        if 'h:' in line:
            f = float(line.split('h:')[-1])
            global best_f, best_ep, best_line
            if f >= best_f:
                best_f = f
                best_ep = ep
                best_line = line
            break
    p.stdout.close()
    print('f-measure %f at ep %d' % (f, ep))
    print(line)
    print('best f-measure %f at ep %d' % (best_f, best_ep))
    print(best_line)


if __name__ == '__main__':
    args = parse_arg()
    for ep in range(47, 48):
        art_eval(ep, args)

