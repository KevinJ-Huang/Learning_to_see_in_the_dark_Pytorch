#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created by xuchongbo at 20171130 in Meitu.
"""

import argparse
import os
import shutil
import time
import rawpy
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torch
import random
from PIL import Image
import sys
import os
from torch.autograd import Variable
from squid import utils
from squid import inference
import numpy as np

config = utils.load_config(sys.argv[1])


# config = utils.load_config('./configs/xcb_20171031_srgan_dataset20171030_gauss_noise03_random_ds_random_kernel_jpeg_skip_1664_all_loss_L1_1_gan_5e2_v3.py')


def prepare_dirs():
    # assure
    utils.touch_dir(config.MODEL_FOLDER)
    dirs = (
    ("TrainOut", config.TRAIN_OUT_FOLDER), ("PeekOut", config.PEEK_OUT_FOLDER), ("TestOut", config.TEST_OUT_FOLDER))
    txt = ""
    for title, dirpath in dirs:
        utils.touch_dir(dirpath)
        dirname = dirpath.rstrip("/").split('/')[-1]
        name = config.experiment_name + "-" + dirname
        utils.create_link(src=dirpath, dst=os.path.join(config.IMAGE_SITE_DATA_DIR, name))
        url = config.IMAGE_SITE_URL.format(dataset_name=name)
        txt += """ %s <a href='%s'> %s </a></br>""" % (title, url, url)

    if config.vis is not None:
        config.vis.text(txt, win='links', env=config.experiment_name)


def main():
    raw_input("press anykey ")
    start_epoch = config.start_epoch  # epoch start from 1
    if config.GPU_ID is not None:
        print("use cuda")
        # cudnn.benchmark = True
        torch.cuda.set_device(config.GPU_ID)
        config.model.cuda(config.GPU_ID)
    if config.only_validate:
        print("only validate it")
        valid_loader = torch.utils.data.DataLoader(dataset=config.valid_dataset, batch_size=config.batch_size,
                                                   shuffle=True, num_workers=2, drop_last=False)
        loss_dict = validate(valid_loader, config.model)
        print("validate:", loss_dict)
    else:
        # prepare dirs
        print ("prepare dirs and links")
        prepare_dirs()
        # Datasets
        print ("init data loader...")
        train_loader = torch.utils.data.DataLoader(dataset=config.train_dataset, batch_size=config.batch_size,
                                                   shuffle=True, num_workers=getattr(config, 'train_num_workers', 16),
                                                   pin_memory=True, drop_last=False)
        valid_loader = torch.utils.data.DataLoader(dataset=config.valid_dataset, batch_size=config.batch_size,
                                                   shuffle=True, num_workers=getattr(config, 'val_num_workers', 4),
                                                   pin_memory=True, drop_last=False)
        iters = len(train_loader)
        print("begin train..")

        input_images = {}
        gt_images = [None] * 6000
        input_images['300'] = [None] * iters
        input_images['250'] = [None] * iters
        input_images['100'] = [None] * iters

        for epoch in range(start_epoch, config.epochs + 1):
            train(epoch, train_loader, config.model,input_images,gt_images)

            if epoch % config.validate_interval_epoch == 0:
                loss_dict, score_dict = validate(valid_loader, config.model)
                utils.print_loss(config, "valid_loss", loss_dict, epoch, iters, iters, need_plot=True)
                utils.print_loss(config, "valid_score", score_dict, epoch, iters, iters, need_plot=True)

            if epoch % config.peek_interval_epoch == 0:
                for item in config.peek_images:
                    peek(config.target_net, item, epoch)

            if epoch % config.save_snapshot_interval_epoch == 0:
                # Save the Models
                config.model.save_snapshot(os.path.join(config.MODEL_FOLDER, 'snapshot_%d' % (epoch)))

    # test model after train has completed.
    if getattr(config, 'is_inference', True):
        inference.run(config.test_input_dir, config.TEST_OUT_FOLDER, config.target_net, config.GPU_ID)

    print "train and inference are completed."


def train(epoch, train_loader, model, input_images, gt_images):
    loss_accumulator = utils.DictAccumulator(config.loss_average_win_size)
    grad_accumulator = utils.DictAccumulator(config.loss_average_win_size)
    score_accumulator = utils.DictAccumulator(config.loss_average_win_size)
    iters = len(train_loader)


    for i, (ratio, index, in_path, gt_path) in enumerate(train_loader):
        ratio = int(ratio)
        index = int(index)
        in_path = "".join(in_path)
        gt_path = "".join(gt_path)
        if input_images[str(ratio)[0:3]][index] is None:
            print('Yes')
            raw = rawpy.imread(in_path)
            input_images[str(ratio)[0:3]][index] = np.expand_dims(pack_raw(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_images[index] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        H = input_images[str(ratio)[0:3]][index].shape[1]
        W = input_images[str(ratio)[0:3]][index].shape[2]

        ps = 512
        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)

        input_patch = input_images[str(ratio)[0:3]][index][:, yy:yy + ps, xx:xx + ps, :]
        gt_patch = gt_images[index][:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=0)
            gt_patch = np.flip(gt_patch, axis=0)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))
        input_patch = np.minimum(input_patch, 1.0)



        inputs = np.transpose(input_patch, (0, 3, 2, 1))
        targets = np.transpose(gt_patch, (0, 3, 2, 1)).copy()

        inputs = torch.from_numpy(inputs).float()
        targets = torch.from_numpy(targets).float()


        inputs = inputs.cuda()
        targets = targets.cuda()
        inputs = Variable(inputs)
        targets = Variable(targets)

        net_outputs, loss, grad, lr_dict, score = model.fit(inputs, targets, update=True, epoch=epoch,
                                                            cur_iter=i + 1, iter_one_epoch=iters)
        loss_accumulator.update(loss)
        grad_accumulator.update(grad)
        score_accumulator.update(score)


        if (i + 1) % config.loss_average_win_size == 0:
            need_plot = True
            if hasattr(config, 'plot_loss_start_iter'):
                need_plot = (i + 1 + (epoch - 1) * iters >= config.plot_loss_start_iter)
            elif hasattr(config, 'plot_loss_start_epoch'):
                need_plot = (epoch >= config.plot_loss_start_epoch)

            utils.print_loss(config, "train_loss", loss_accumulator.get_average(), epoch=epoch, iters=iters,
                             current_iter=i + 1, need_plot=need_plot)
            utils.print_loss(config, "grad", grad_accumulator.get_average(), epoch=epoch, iters=iters,
                             current_iter=i + 1, need_plot=need_plot)
            utils.print_loss(config, "learning rate", lr_dict, epoch=epoch, iters=iters, current_iter=i + 1,
                             need_plot=need_plot)

            utils.print_loss(config, "train_score", score_accumulator.get_average(), epoch=epoch, iters=iters,
                             current_iter=i + 1, need_plot=need_plot)




    if epoch % config.save_train_hr_interval_epoch == 0:
        k = random.randint(0, net_outputs['output'].size(0) - 1)
        for name, out in net_outputs.items():

            utils.save_tensor(out.data[k],
                              os.path.join(config.TRAIN_OUT_FOLDER, 'epoch_%d_k_%d_%s.png' % (epoch, k, name)))



def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out




def validate(valid_loader, model):
    loss_accumulator = utils.DictAccumulator()
    score_accumulator = utils.DictAccumulator()

    # loss of the whole validation dataset
    for i, (inputs, targets) in enumerate(valid_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        inputs = Variable(inputs, volatile=True)
        targets = Variable(targets)

        loss, score = model.fit(inputs, targets, update=False)

        loss_accumulator.update(loss)
        score_accumulator.update(score)

    return loss_accumulator.get_average(), score_accumulator.get_average()


def peek(target_net, img_path, epoch):
    # open image
    img = Image.open(img_path)

    # save raw peek images for first time
    if epoch == config.peek_interval_epoch:
        img.save(os.path.join(config.PEEK_OUT_FOLDER, os.path.basename(img_path) + '_0.png'))

    # do inference
    img = img.convert('RGB')
    trans = transforms.Compose([transforms.ToTensor(), ])
    input_tensor = trans(img)
    inputs = input_tensor.view(1, input_tensor.size(0), input_tensor.size(1), input_tensor.size(2))

    print("inference...")
    inputs = Variable(inputs, volatile=True)
    target_net.eval()
    net_outputs = target_net(inputs.cuda())

    # save net_outputs
    for name, out in net_outputs.items():
        utils.save_tensor(out.data[0], os.path.join(config.PEEK_OUT_FOLDER,
                                                    os.path.basename(img_path) + '_%s_%d.png' % (name, epoch)))


if __name__ == '__main__':
    main()

