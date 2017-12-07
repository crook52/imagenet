#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess as sp
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--rootdir','--r',default='~/chainer-master')
parser.add_argument('--o',default='result')
parser.add_argument('--cuDNN',default=False)

args = parser.parse_args()

if args.cuDNN:
    sp.check_output('export CHAINER_CUDNN=1',shell=True)
else:
    sp.check_output('export CHAINER_CUDNN=0',shell=True)

workspace = args.rootdir + '/examples/imagenet'

os.chdir(workspace)

cmd = 'python3 train_imagenet_data_parallel.py train.txt val.txt' \
      ' -a resnet50 -B 64 -j 8 -m train_mean.npy -R /home/cs28/workspace/ -o 026 -E 1'

sp.run(cmd,shell=True)
#sp.check_output('cd'+workspace)
#sp.check_output('python3 train_imagenet_data_parallel.py \
#train.txt val.txt -a resnet50 -B 64 -j 8 -m train_mean.npy -R ~/workspace/ -o 1205 -E 1')

