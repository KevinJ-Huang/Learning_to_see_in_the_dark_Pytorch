#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import sys
sys.path.insert(0, '/home/nwpu-ustc/PycharmProjects/Dark-pytorch-memory/')


# =================== config for train flow ============================================================================
DESC = "L1 "
experiment_name = os.path.splitext(__file__.split('/')[-1])[0]

# BASE_ROOT = '/3T/sr_group'   # 1080a
BASE_ROOT = '/home/nwpu-ustc/PycharmProjects/Dark-pytorch-memory'  # server

TRAIN_ROOT       = '/home/nwpu-ustc/PycharmProjects/Dark-pytorch-memory/'+experiment_name

MODEL_FOLDER        = os.path.join(TRAIN_ROOT, 'models')
TRAIN_OUT_FOLDER    = os.path.join(TRAIN_ROOT, 'train_out')
PEEK_OUT_FOLDER     = os.path.join(TRAIN_ROOT, 'peek_out')
TEST_OUT_FOLDER     = os.path.join(TRAIN_ROOT, 'test_out')

DATASET_DIR = '/3T/images'
DATASET_ID = 'dataset20171030_gauss_noise03_random_ds_random_kernel_jpeg'
DATASET_TXT_DIR = './imgs/'

IMAGE_SITE_URL  = 'http://172.16.3.247:8000/image-site/dataset/{dataset_name}?page=1&size=50'
IMAGE_SITE_DATA_DIR = '/home/nwpu-ustc/PycharmProjects/Dark-pytorch/dataset'

peek_images = ['./imgs/44.png',  './imgs/36.png']
test_input_dir = "/3T/images/ftt-png/"

INPUT_DIR = '/home/nwpu-ustc/PycharmProjects/Dark-pytorch-memory/dataset/Sony/short/'
GT_DIR = '/home/nwpu-ustc/PycharmProjects/Dark-pytorch-memory/dataset/Sony/long/'


GPU_ID = 0
epochs = 4000
batch_size = 1
start_epoch = 1
save_snapshot_interval_epoch = 1
peek_interval_epoch = 1000000
save_train_hr_interval_epoch = 1
loss_average_win_size = 1
validate_interval_epoch = 1000000
plot_loss_start_epoch = 1
only_validate = False  #

from visdom import Visdom
# vis = Visdom(server='http://localhost', port=8097)
vis = None

# =================== config for model =================================================================================
from squid.model import SuperviseModel
import torch
import torch.nn as nn
from net import DarkNet


target_net = DarkNet()

model = SuperviseModel({
    'net': target_net,
    'optimizer': torch.optim.Adam([{'name': 'net_params', 'params': target_net.parameters(), 'base_lr': 1e-3,'warm_epoch':2, 'total_epoch': epochs}],
                                  betas=(0.9, 0.999), weight_decay=0.0005),
    'not_show_gradient': True,
    'supervise': {
        'out':  {
            'L1_loss': {'obj': nn.L1Loss(size_average=True),  'factor': 1.0, 'weight': 1.0}
        },
    },
    'metrics': {}
})

# =================== dataset ==========================================================================================
from data import DarkPipeline

train_num_workers = 24
PATCH_SIZE = 512

train_dataset = DarkPipeline({
            'input_dir':INPUT_DIR,
            'gt_dir':GT_DIR,
            'patch_size':PATCH_SIZE
})

valid_dataset = DarkPipeline({
            'input_dir':INPUT_DIR,
            'gt_dir':GT_DIR,
            'patch_size':PATCH_SIZE
})