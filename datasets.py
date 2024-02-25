"""
Author  : Xu fuyong
Time    : created by 2019/7/16 19:49

"""
import h5py
import numpy as np
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        file = './' + self.h5_file
        with h5py.File(self.h5_file, 'r') as f:
            return f['lr'][idx] / 255. , f['hr'][idx] / 255. 

    def __len__(self):
        file = './' + self.h5_file
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr_data = f['lr'][str(idx)][:, :] / 255.
            hr_data = f['hr'][str(idx)][:, :] / 255.
            return lr_data, hr_data

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


