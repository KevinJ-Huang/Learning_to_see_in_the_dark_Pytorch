import numpy as np
import os
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import glob
import rawpy
import torch

class Dataset(data.Dataset):


    def __init__(self, args):
        self.args = args
        self.toTensor = transforms.ToTensor()


    def get_id(self):
        train_fns = glob.glob(self.args['gt_dir'] + '0*.ARW')
        train_ids = []
        for i in range(len(train_fns)):
            _, train_fn = os.path.split(train_fns[i])
            train_ids.append(int(train_fn[0:5]))
        return train_ids


    def __len__(self):
        train_ids = self.get_id()
        return len(train_ids)


    def __getitem__(self, index):
        train_ids = self.get_id()
        train_id = train_ids[index]


        in_files = glob.glob(self.args['input_dir']+'%05d_00*.ARW'%train_id)
        in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
        _, in_fn = os.path.split(in_path)

        gt_files = glob.glob(self.args['gt_dir'] + '%05d_00*.ARW' % train_id)
        gt_path = gt_files[0]
        _, gt_fn = os.path.split(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)
        print(ratio)

        return ratio,index,in_path,gt_path







