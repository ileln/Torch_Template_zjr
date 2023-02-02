import os
import torch
import numpy as np
import pprint
from runner import Runner

from tools import get_args, seed_torch

# 读取参数
args = get_args("config/MSRGCN/h36m/demo.yaml") # 读取yaml文件中的参数，并以字典形式保存在args变量中
print("\n================== Arguments =================")
pprint(args, indent=4)
print("==========================================\n")


# 系统设置
seed_torch(args.seed) # 初始cudnn加速，也可设置随机种子，此处没有设置随机种子
os.environ["CUDA_VISIBLE_DEVICES"] = args.device # 全局修改模型占用的显卡

runner = Runner(args)