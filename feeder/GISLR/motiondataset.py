#!/usr/bin/env python
# encoding: utf-8
'''
@project : MSRGCN
@file    : dataset.py
@author  : Droliven
@contact : droliven@163.com
@ide     : PyCharm
@time    : 2021-07-27 20:16
'''
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
# from .. import data_utils
# from ..multi_scale import downs_from_22
# from ..dct import get_dct_matrix, dct_transform_numpy

def data_load(data_path):
    # 规整后的数据的读取，数据规整方式见test_idea
    data = np.load(data_path + "data_all.npy")
    label = np.load(data_path + "label_all_onehot.npy")
    return data, label

class GislrDataset(Dataset):

    def __init__(self, path_to_data, mode_name="train", kflod_random_state=42, kflod_test_size=0.2, device="cuda:0", **dic):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        self.path_to_data = path_to_data

        self.kflod_random_state = kflod_random_state
        self.kflodtest_size = kflod_test_size
        self.data, self.label = data_load(self.path_to_data)
        train_data, test_data, train_label, test_label = train_test_split(self.data, self.label, test_size=self.kflodtest_size, random_state=self.kflod_random_state)
        if mode_name == "train":
            self.input_data = train_data
            self.input_label = train_label
        elif mode_name == "test":
            self.input_data = test_data
            self.input_label = test_label
        print(self.input_data.shape)

    def __len__(self):
        return self.input_label.shape[0]
    
    def __iter__(self):
        return self

    def __getitem__(self, index):
        data = self.input_data[index]
        label = self.input_label[index]
        return data, label, index

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    feeder = GislrDataset(path_to_data="/share/kaggle/asl-signs/")
    data_loader = DataLoader(dataset=feeder, batch_size=128, shuffle=True, num_workers=8, drop_last=True)
    process = tqdm(data_loader, ncols=40)
    for batch_idx, (data, label, index) in enumerate(process):
        print(data.shape)
        print(label.shape)


