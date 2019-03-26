import subprocess
import argparse

parser = argparse.ArgumentParser(description='Hyperparams')
# parser.add_argument('--gpus', nargs='?', type=int, default=1)
parser.add_argument('--resume_root', nargs='?', type=str, default=None)
parser.add_argument('--dataset', nargs='?', type=str, default='resnet50')
args = parser.parse_args()

# root = '/home/xieenze/R4D/whai_workspace/Text-Det_Recog/embeding/wenhai/psenet_pp.pytorch/'
root = '/home/data1/sxg/IC19/Text-Det_Recog-master/embeding/wenhai/psenet_pp.pytorch/'
cd_root_cmd = 'cd %s'%root

processes = []
best_f = 0
best_ep = 0
best_line = ''

def ctw1500_eval(ep):
    # run_cmd = 'python test_ctw1500.py --arch resnet18_half --scale 4 --short_size 320 --min_kernel_area 0.5 --min_area 50 --min_score 0.84 --resume %s/checkpoint_%dep.pth.tar'%(args.resume_root, ep)
    # run_cmd = 'python test_ctw1500.py --arch resnet18_half --scale 4 --short_size 512 --min_kernel_area 1.3 --min_area 130 --min_score 0.86 --resume %s/checkpoint_%dep.pth.tar'%(args.resume_root, ep)
    run_cmd = 'python test_ctw1500.py --arch resnet18_half --scale 4 --short_size 640 --min_kernel_area 2.0 --min_area 200 --min_score 0.88 --resume %s/checkpoint_%dep.pth.tar'%(args.resume_root, ep)
    # run_cmd = 'python test_ctw1500.py --arch resnet18_half --scale 4 --short_size 736 --min_kernel_area 2.6 --min_area 260 --min_score 0.88 --resume %s/checkpoint_%dep.pth.tar'%(args.resume_root, ep)
    run_cmd = cd_root_cmd + " && " + run_cmd
    print('ep: %d'%ep)
    p = subprocess.Popen(run_cmd, shell=True, stderr=subprocess.STDOUT)
    p.wait()

    eval_cmd = cd_root_cmd + " && " + 'cd eval && python eval_ctw1500.py'
    p = subprocess.Popen(eval_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p.wait()
    f = 0
    line = ''
    for line in iter(p.stdout.readline, b''):
        if 'f:' in line:
            f = float(line.split('f: ')[-1])
            global best_f, best_ep, best_line
            if f >= best_f:
                best_f = f
                best_ep = ep
                best_line = line
            break
    p.stdout.close()
    print('f-measure %f at ep %d'%(f, ep))
    print(line)
    print('best f-measure %f at ep %d'%(best_f, best_ep))
    print(best_line)


def art_eval(ep):
    run_cmd = 'CUDA_VISIBLE_DEVICES=3 python test_art.py --arch resnet50 --scale 1 --short_size 928  --min_kernel_area 2.6 --min_area 260 --min_score 0.86 --resume /home/data1/sxg/IC19/Text-Det_Recog-master/embeding/wenhai/psenet_pp.pytorch/checkpoints/art_resnet50_bs_8_ep_200_imsz_736_kscale_70_short_size_800_loss_w_100_50_25_use_polylr_v3/checkpoint_%dep.pth.tar'%(ep)
    run_cmd = cd_root_cmd + " && " + run_cmd
    print('ep: %d'%ep)
    p = subprocess.Popen(run_cmd, shell=True, stderr=subprocess.STDOUT)
    p.wait()

    eval_cmd = cd_root_cmd + '&&' + 'sh art_eval.sh'
    p = subprocess.Popen(eval_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p.wait()
    f = 0
    line = ''
    for line in iter(p.stdout.readline, b''):
        if 'Hmean:_' in line:
            f = float(line.split('Hmean:_')[-1])
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


def ic15_eval(ep):
    # run_cmd = 'python test_ic15.py --arch resnet18_half --scale 4 --short_size 720 --min_kernel_area 2.6 --min_area 260 --min_score 0.85 --resume %s/checkpoint_%dep.pth.tar'%(args.resume_root, ep)
    run_cmd = 'python test_ic15.py --arch resnet18_half --scale 4 --short_size 736 --min_kernel_area 2.6 --min_area 260 --min_score 0.85 --resume %s/checkpoint_%dep.pth.tar'%(args.resume_root, ep)
    run_cmd = cd_root_cmd + " && " + run_cmd
    print('ep: %d'%ep)
    p = subprocess.Popen(run_cmd, shell=True, stderr=subprocess.STDOUT)
    p.wait()

    eval_cmd = cd_root_cmd + " && " + 'cd eval && sh eval_ic15.sh'
    p = subprocess.Popen(eval_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p.wait()
    f = 0
    line = ''
    for line in iter(p.stdout.readline, b''):
        if '\"hmean\": ' in line:
            f = float(line.split(', ')[-2].split('\"hmean\": ')[-1])
            global best_f, best_ep, best_line
            if f >= best_f:
                best_f = f
                best_ep = ep
                best_line = line
            break
    p.stdout.close()
    print('f-measure %f at ep %d'%(f, ep))
    print(line)
    print('best f-measure %f at ep %d'%(best_f, best_ep))
    print(best_line)

def tt_eval(ep):
    # run_cmd = 'python test_tt.py --arch resnet18_half --scale 4 --short_size 320 --min_kernel_area 0.5 --min_area 50 --min_score 0.82 --resume %s/checkpoint_%dep.pth.tar'%(args.resume_root, ep)
    # run_cmd = 'python test_tt.py --arzch resnet18_half --scale 4 --short_size 512 --min_kernel_area 1.3 --min_area 130 --min_score 0.84 --resume %s/checkpoint_%dep.pth.tar'%(args.resume_root, ep)
    run_cmd = 'python test_tt.py --arch resnet18_half --scale 4 --short_size 640 --min_kernel_area 2.0 --min_area 200 --min_score 0.86 --resume %s/checkpoint_%dep.pth.tar'%(args.resume_root, ep)
    run_cmd = cd_root_cmd + " && " + run_cmd
    print('ep: %d'%ep)
    p = subprocess.Popen(run_cmd, shell=True, stderr=subprocess.STDOUT)
    p.wait()

    eval_cmd = cd_root_cmd + " && " + 'cd eval && sh eval_tt.sh'
    p = subprocess.Popen(eval_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p.wait()
    f = 0
    line = ''
    for line in iter(p.stdout.readline, b''):
        if 'Hmean:_' in line:
            f = float(line.split('Hmean:_')[-1])
            global best_f, best_ep, best_line
            if f >= best_f:
                best_f = f
                best_ep = ep
                best_line = line
            break
    p.stdout.close()
    print('f-measure %f at ep %d'%(f, ep))
    print(line)
    print('best f-measure %f at ep %d'%(best_f, best_ep))
    print(best_line)

def msra_eval(ep):
    run_cmd = 'python test_msra.py --arch resnet18_half --scale 4 --short_size 736 --min_kernel_area 2.6 --min_area 260 --min_score 0.86 --resume %s/checkpoint_%dep.pth.tar'%(args.resume_root, ep)
    run_cmd = cd_root_cmd + " && " + run_cmd
    print('ep: %d'%ep)
    p = subprocess.Popen(run_cmd, shell=True, stderr=subprocess.STDOUT)
    p.wait()

    eval_cmd = cd_root_cmd + " && " + 'cd eval && python eval_msra.py'
    p = subprocess.Popen(eval_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p.wait()
    f = 0
    line = ''
    for line in iter(p.stdout.readline, b''):
        if 'f:' in line:
            f = float(line.split('f: ')[-1])
            global best_f, best_ep, best_line
            if f >= best_f:
                best_f = f
                best_ep = ep
                best_line = line
            break
    p.stdout.close()
    print('f-measure %f at ep %d'%(f, ep))
    print(line)
    print('best f-measure %f at ep %d'%(best_f, best_ep))
    print(best_line)


if args.dataset == 'ctw1500':
    eval_ = ctw1500_eval
elif args.dataset == 'ic15':
    eval_ = ic15_eval
elif args.dataset == 'tt':
    eval_ = tt_eval
elif args.dataset == 'msra':
    eval_ = msra_eval
elif args.dataset == 'art':
    eval_ = art_eval

# for ep in range(550, 601, 10):
#     eval_(ep)
for ep in range(180, 200, 2):
    eval_(ep)