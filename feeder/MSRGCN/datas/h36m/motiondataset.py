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

from torch.utils.data import Dataset
import numpy as np
from .. import data_utils
from ..multi_scale import downs_from_22
from ..dct import get_dct_matrix, dct_transform_numpy

class MotionDataset(Dataset):

    def __init__(self, path_to_data, actions, mode_name="train", train_manner="all", test_manner="all", train_split=0, teat_split=1, validation_split=2, input_n=20, output_n=10, dct_used=15, sample_rate=2, Index2212=[], Index127=[], Index74=[], global_max=0, global_min=0, device="cuda:0", debug_step=100, subs=[]):
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
        # self.load_mode = load_mode # 设置加载模式，0是训练，1是测试，2是验证
        if mode_name == 'train':
            actions = train_manner
            split = train_split
        elif mode_name == 'test':
            # actions = test_manner
            actions = actions
            split = teat_split

        down_key=[('p22', 'p12', Index2212), ('p12', 'p7', Index127), ('p7', 'p4', Index74)]
        # subs = [[1, 6, 7, 8, 9], [5], [11]]
        # subs = [[1], [5], [11]] # 训练集和测试集选择
        # subs = [[1, 6, 7, 8, 9, 11], [5]]
        # print(actions)
        acts = data_utils.define_actions(actions)

        subjs = subs[split]
        all_seqs, dim_ignore, dim_used = data_utils.load_data_3d(path_to_data, subjs, acts, sample_rate, input_n + output_n, test_manner=test_manner, device=device)
        gt_32 = all_seqs.transpose(0, 2, 1)  # b, 96, 35
        gt_22 = gt_32[:, dim_used, :]

        gt_all_scales = {'p32': gt_32, 'p22': gt_22}
        gt_all_scales = downs_from_22(gt_all_scales, down_key=down_key)
        # print("gt_all_scales.keys", gt_all_scales.keys[0]) # 看一下关键字
        # 重复已知最后一帧
        input_all_scales = {}
        for k in gt_all_scales.keys():
            input_all_scales[k] = np.concatenate((gt_all_scales[k][:, :, :input_n], np.repeat(gt_all_scales[k][:, :, input_n-1:input_n], output_n, axis=-1)), axis=-1)

        # DCT *********************
        self.dct_used = dct_used
        self.dct_m, self.idct_m = get_dct_matrix(input_n + output_n)

        for k in input_all_scales:
            input_all_scales[k] = dct_transform_numpy(input_all_scales[k], self.dct_m, dct_used)

        # Max min norm to -1 -> 1 ***********
        self.global_max = global_max
        self.global_min = global_min

        # 求全局最大最小值
        if mode_name == 'train':
            gt_max = []
            gt_min = []
            for k in gt_all_scales.keys():
                gt_max.append(np.max(gt_all_scales[k]))
                gt_min.append(np.min(gt_all_scales[k]))
            for k in input_all_scales.keys():
                gt_max.append(np.max(input_all_scales[k]))
                gt_min.append(np.min(input_all_scales[k]))

            self.global_max = np.max(np.array(gt_max))
            self.global_min = np.min(np.array(gt_min))

        for k in input_all_scales.keys():
            input_all_scales[k] = (input_all_scales[k] - self.global_min) / (self.global_max - self.global_min)
            input_all_scales[k] = input_all_scales[k] * 2 - 1

        # todo 加速调试 *********************************
        little = np.arange(0, input_all_scales[list(input_all_scales.keys())[0]].shape[0], debug_step)
        for k in input_all_scales:
            input_all_scales[k] = input_all_scales[k][little]
            gt_all_scales[k] = gt_all_scales[k][little]

        self.gt_all_scales = gt_all_scales
        self.input_all_scales = input_all_scales

    def __len__(self):
        return self.gt_all_scales[list(self.gt_all_scales.keys())[0]].shape[0]

    def __getitem__(self, item):
        gts = {}
        inputs = {}
        for k in ['p32', 'p22', 'p12', 'p7', 'p4']:
            gts[k] = self.gt_all_scales[k][item]
            inputs[k] = self.input_all_scales[k][item]
        return inputs, gts

if __name__ == '__main__':
    pass


