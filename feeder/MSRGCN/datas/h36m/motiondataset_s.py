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

class MotionDataset_s(Dataset):

    def __init__(self, path_to_data, actions, mode_name="train", input_n=10, output_n=25, dct_used=35, split=0, sample_rate=2, down_key=[('p22', 'p12', []), ('p12', 'p7', []), ('p7', 'p4', [])], test_manner="all", global_max=0, global_min=0, device="cuda:0", debug_step=100):
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
        self.split = split

        subs = [[1, 6, 7, 8, 9], [5], [11]]
        # subs = [[1], [5], [11]] # 训练集和测试集选择
        acts = data_utils.define_actions(actions)

        subjs = subs[split]
        all_seqs, dim_ignore, dim_used = data_utils.load_data_3d(path_to_data, subjs, acts, sample_rate, input_n + output_n, test_manner=test_manner, device=device)
        gt_32 = all_seqs.transpose(0, 2, 1)  # b, 96, 35
        gt_22 = gt_32[:, dim_used, :]

        gt_all_scales = {'p32': gt_32, 'p22': gt_22}
        gt_all_scales = downs_from_22(gt_all_scales, down_key=down_key)
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
        gts_s = {}
        inputs = {}
        inputs_s = {}
        for k in ['p32', 'p22', 'p12', 'p7', 'p4']:
            gts[k] = self.gt_all_scales[k][item]
            # print("gts[k]", gts[k].shape)
            # gts_s[k] = gts[k][:, :20] # 先10帧，后15帧
            gts_s[k] = gts[k][:, :25] # 先15帧，后10帧
            inputs[k] = self.input_all_scales[k][item]
            # inputs_s[k] = inputs[k][:, :20] # 先10帧，后15帧
            inputs_s[k] = inputs[k][:, :25] # 先15帧，后10帧
        return inputs, inputs_s, gts, gts_s

if __name__ == '__main__':
    pass


