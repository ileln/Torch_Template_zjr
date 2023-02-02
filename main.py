import os
import torch
import numpy as np
import pprint
from runner import Runner

from tools import get_args, seed_torch, save_args

# 读取参数
args = get_args("config/MSRGCN/h36m/demo.yaml") # 读取yaml文件中的参数，并以字典形式保存在args变量中
print("\n================== Arguments =================")
pprint.pprint(vars(args), indent=4)
print("==========================================\n")
save_args(args) # 保存参数

# 系统设置
seed_torch(args.seed) # 初始cudnn加速，也可设置随机种子，此处没有设置随机种子
os.environ["CUDA_VISIBLE_DEVICES"] = args.device # 全局修改模型占用的显卡

# 运行器初始化
runner = Runner(args)