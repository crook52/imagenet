#!/usr/bin/env python
"""Example code of learning a large scale convnet from LSVRC2012 dataset
with multiple GPUs using data parallelism.

Prerequisite: To run this example, crop the center of ILSVRC2012 training and
validation images, scale them to 256x256 and convert them to RGB, and make
two lists of space-separated CSV whose first column is full path to image and
second column is zero-origin label (this format is same as that used by Caffe's
ImageDataLayer).

You need to install chainer with NCCL to run this example.
Please see https://github.com/nvidia/nccl#build--run .

"""
from __future__ import print_function
import argparse

import numpy as np

import chainer
from chainer import training
from chainer.training import extensions
from chainer.training import updaters

import alex
import googlenet
import googlenetbn
import nin
import resnet50
import train_imagenet
import csv
import re
def set_random_seed(seed):
    # set Python random seed
   # random.seed(seed)

    # set NumPy random seed
    np.random.seed(seed)

    # set Chainer(CuPy) random seed
   # cp.random.seed(seed)

def main():
    archs = {
        'alex': alex.Alex,
        'alex_fp16': alex.AlexFp16,
        'googlenet': googlenet.GoogLeNet,
        'googlenetbn': googlenetbn.GoogLeNetBN,
        'googlenetbn_fp16': googlenetbn.GoogLeNetBNFp16,
        'nin': nin.NIN,
        'resnet50': resnet50.ResNet50
    }


    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('--train', default='train.txt', help='Path to training image-label list file')
    parser.add_argument('--val', default='val.txt', help='Path to validation image-label list file')
    parser.add_argument('--arch', '-a', choices=archs.keys(),
                        default='resnet50', help='Convnet architecture')
    parser.add_argument('--batchsize', '-B', type=int, default=64,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=0,
                        help='Number of epochs to train')
    parser.add_argument('--gpus', '-g', type=int, nargs="*",
                        default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--loaderjob', '-j', type=int, default=8,
                        help='Number of parallel data loading processes')
    parser.add_argument('--mean', '-m', default='train_mean.npy',
                        help='Mean file (computed by compute_mean.py)')
    parser.add_argument('--resume', '-r', default='ymd_trainer',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--root', '-R', default='/home/dnn/',
                        help='Root directory path of image files')
    parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                        help='Validation minibatch size')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--LR',default=0.01, type=float)
    parser.set_defaults(test=False)
    args = parser.parse_args()
    best_lr = args.LR
    resume_trainer = args.out + '/' + args.resume

    set_random_seed(0)  ####固定

    # Initialize the model to train
    model = archs[args.arch]()
    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)

    # Load the datasets and mean file
    mean = np.load(args.mean)
    train = train_imagenet.PreprocessedDataset(
        args.train, args.root, mean, model.insize, args.epoch) ##Falseを追加でも固定できる
    val = train_imagenet.PreprocessedDataset(
        args.val, args.root, mean, model.insize, args.epoch, random=False)
    # These iterators load the images with subprocesses running in parallel to
    # the training/validation.
    devices = tuple(args.gpus)

    train_iters = [
        chainer.iterators.MultiprocessIterator(i,
                                               args.batchsize,
                                               n_processes=args.loaderjob)
        for i in chainer.datasets.split_dataset_n_random(train, len(devices),seed=1)]
    val_iter = chainer.iterators.MultiprocessIterator(
        val, args.val_batchsize, repeat=False, n_processes=args.loaderjob)

    # Set up an optimizer
    #optimizer = chainer.optimizers.MomentumSGD(lr=args.LR*len(devices), momentum=0.9)
    optimizer = chainer.optimizers.MomentumSGD(lr=args.LR, momentum=0.9)
    optimizer.setup(model)

    # Set up a trainer
    updater = updaters.MultiprocessParallelUpdater(train_iters, optimizer,
                                                   devices=devices)
    #trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)
    trainer = training.Trainer(updater, (args.epoch + 1, 'epoch'), args.out)

    if args.test:
        val_interval = 5, 'epoch'
        log_interval = 1, 'epoch'
    else:
        val_interval = 1, 'epoch'
        log_interval = 100, 'iteration'
        sub_log_interval = 1, 'iteration'
        snapshot_interval = 1, 'epoch'

    trainer.extend(extensions.Evaluator(val_iter, model, device=args.gpus[0]),
                   trigger=val_interval)
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(filename='ymd_trainer'), trigger=snapshot_interval)
    # trainer.extend(extensions.snapshot_object(
    #     model, 'model_iter_{.updater.iteration}'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'ymd_model'), trigger=snapshot_interval)
    # Be careful to pass the interval directly to LogReport
    # (it determines when to emit log rather than when to read observations)
    trainer.extend(extensions.LogReport(trigger=log_interval, log_name='main_log'))
    trainer.extend(extensions.LogReport(trigger=sub_log_interval, log_name='iteration_log'))
    trainer.extend(extensions.LogReport(trigger=val_interval, log_name='epoch_log'))
    trainer.extend(extensions.observe_lr(), trigger=sub_log_interval)
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'lr'
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=2))

    if not args.epoch == 0:
        if args.resume:
            chainer.serializers.load_npz(resume_trainer, trainer)
            trainer.updater.get_optimizer('main').lr = best_lr
    trainer.run()
    # print(trainer.observation)
    # loss = trainer.observation['main/loss']
    # pattern=r'([0-9]+.[0-9]+)'
    # loss=re.findall(pattern,str(loss))
    # filename = args.out+'/loss.csv'
    # f = open(filename,'a',newline='')
    # writer = csv.writer(f, lineterminator='\n')
    # writer.writerow(loss)
    # f.close()

if __name__ == '__main__':
    main()
