#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: config.py

import numpy as np

# model flags ---------------------
RESNET_NUM_BLOCKS = [3, 4, 6, 3]     # for resnet50
# RESNET_NUM_BLOCKS = [3, 4, 23, 3]    # for resnet101
MODE_FPN=False
FPN_NUM_CHANNEL=256
NUM_ATT_CONV=1
# tasks ---------------------
MODE_SEMANTIC=False
LABEL_SMOOTH=False
MODE_MULTI_SEG=0
MODE_CLS=0
MODE_SEG=0.5
MODE_ATT=0.5
NUM_ATT = 312
NUM_CLASS = 150
# dataset -----------------------
IS_ZSL=True
BBOX=False
BASEDIR = 'dataset/CUB'
TRAIN_DATASET = 'zsl_train_3'
VAL_DATASET = 'zsl_test_3'
#TRAIN_DATASET = 'train_0.6'
#VAL_DATASET = 'val'
TRAIN_NUM = 0
VAL_NUM=0
NUM_CATEGORY_TEST = 50
NUM_CATEGORY_TRAIN = 150
NUM_CATEGORY=200
INPUT_SIZE=224
TRAIN_IMAGE_SIZE=224
CLASS_NAMES = []  # NUM_CLASS strings

ATT_DATA=[]
BATCH_SIZE_ONE_GPU=32

# train -----------------------
WEIGHT_DECAY = 1e-5
BASE_LR = 1e-3 # defined for total batch size=128. Otherwise it will be adjusted automatically
#BASE_LR = 1e-4  #fine-tune
WARMUP = 100   # in terms of iterations. This is not affected by #GPUs
WARMUP_INIT_LR = 1e-4  # defined for total batch size=128. Otherwise it will be adjusted automatically
STEPS_PER_EPOCH = 50
STARTING_EPOCH = 1  # the first epoch to start with, useful to continue a training

# LR_SCHEDULE=3x is the same as LR_SCHEDULE=[420000, 500000, 540000]
# which means to decrease LR at steps 420k and 500k and stop training at 540k.
# When the total bs!=128, the actual iterations to decrease learning rate, and
# the base learning rate are computed from BASE_LR and LR_SCHEDULE.
# Therefore, there is *no need* to modify the config if you only change the number of GPUs.
LR_SCHEDULE = [4200,5000,5400]
#LR_SCHEDULE = [2100,2500]
EVAL_TIMES =20 # eval times during training