#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$2  --master_port=9900  $(dirname "$0")/train.py $1 --launcher pytorch ${@:3}
