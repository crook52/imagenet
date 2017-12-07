#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess as sp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--rootdir','--r',default='~/chainer-master')
parser.add_argument('--o',default='result')
parser.add_argument('--cuDNN',default=False)
sp.check_output('python3 train_imagenet_data_parallel.py \
train.txt val.txt -a resnet50 -B 64 -j 8 -m train_mean.npy -R ~/workspace/ -o 1205 -E 1')

test