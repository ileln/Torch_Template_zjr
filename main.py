import os
import torch
import numpy as np
import pprint
import argparse

from runner import Runner

from tools import get_args, seed_torch, save_args

# 读取参数
parser = argparse.ArgumentParser(description='Arguments for running the scripts')
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--config', type=str, default="config/MSRGCN/h36m/demo.yaml", help="配置文件")
parser.add_argument('--sample_rate', type=int, default=2, help="抽帧")
parser.add_argument('--lr', type=float, default=2e-4, help="")
parser.add_argument('--lr_decay', type=float, default=0.98, help="")
parser.add_argument('--n_epoch', type=int, default=5000)
parser.add_argument('--leaky_c', type=float, default=0.2)
parser.add_argument('--p_dropout',type=float, default=0.1, help="")
parser.add_argument('--train_batch_size', type=int, default=16, help="")
parser.add_argument('--test_batch_size', type=int, default=128, help="")
parser.add_argument('--input_n', type=int, default=10, help="")
parser.add_argument('--output_n', type=int, default=25, help="")
parser.add_argument('--seq_len', type=int, default=35, help="")
parser.add_argument('--dct_n', type=int, default=35, help="")
parser.add_argument('--device', type=str, default='0', help="")
parser.add_argument('--num_works', type=int, default=8, help="")
parser.add_argument('--seed', type=int, default=3450, help="")

args = get_args(parser) # 读取yaml文件中的参数，并以字典形式保存在args变量中
print("\n================== Arguments =================")
pprint.pprint(vars(args), indent=4)
print("==========================================\n")
save_args(args) # 保存参数

# 系统设置
seed_torch(args.seed) # 初始cudnn加速，也可设置随机种子，此处没有设置随机种子
os.environ["CUDA_VISIBLE_DEVICES"] = args.device # 全局修改模型占用的显卡

# 运行器初始化
# runner = Runner(args)