#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess as sp
import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('--rootdir', '--r', default='/home/dnn/git/imagenet/chainer-master2')
parser.add_argument('--o', default='testooooooooo')
parser.add_argument('--cuDNN', default=False)
parser.add_argument('--maxEpoch', default=20 ,type=int)
args = parser.parse_args()

if args.cuDNN:
    sp.check_output('export CHAINER_CUDNN=1',shell=True)
else:
    sp.check_output('export CHAINER_CUDNN=0',shell=True)

workspace = args.rootdir + '/examples/imagenet'

os.chdir(workspace)

print(os.getcwd())

lr_list = [1.0, 0.5, 1.1]
print(os.getcwd())
sp.check_output('pwd',shell=True)

main_output = args.o
sub_output = args.o
globalLR = 0.01 ##init

for epoch in range(args.maxEpoch):
    best_err = float('inf')
    bestLR = 1000000
    for lr_num in range(len(lr_list)):

        sub_cmd = 'python3 ymd_sub.py' \
                  ' --out ' + sub_output + \
                  ' --LR ' + str(lr_list[lr_num]) + \
                  ' --gLR ' + str(globalLR) + \
                  ' --epoch ' + str(epoch) + \
                  ' --iteration ' + str(1000)
        print(sub_cmd)
        sp.run(sub_cmd,shell=True)
        filename = workspace+'/'+args.o+'/'+str(epoch)+'_'+str(lr_list[lr_num])
        print(filename)
        data = json.load(open(filename))
        new_err = data[len(data)-1]['main/loss']
        if best_err > new_err:
            bestLR=lr_list[lr_num]
            best_err = new_err

    print('I chose',bestLR)
    main_cmd = 'python3 ymd_main.py' \
               ' --out ' + main_output +\
               ' --LR ' + str(bestLR*globalLR)+ \
               ' --epoch ' + str(epoch)
    print(main_cmd)
    sp.run(main_cmd,shell=True)
    globalLR = bestLR*globalLR


#sp.check_output('cd'+workspace)
#sp.check_output('python3 train_imagenet_data_parallel.py \
#train.txt val.txt -a resnet50 -B 64 -j 8 -m train_mean.npy -R ~/workspace/ -o 1205 -E 1')

