import os
import torch
import numpy as np
import pprint
import argparse
import pandas as pd
import wandb
import datetime

from runner import Runner

from tools import get_args, seed_torch, save_args
from feeder.MSRGCN.datas import define_actions, define_actions_cmu

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
parser.add_argument('--test_batch_size', type=int, default=256, help="")
parser.add_argument('--input_n', type=int, default=10, help="")
parser.add_argument('--output_n', type=int, default=25, help="")
parser.add_argument('--seq_len', type=int, default=35, help="")
parser.add_argument('--dct_n', type=int, default=35, help="")
parser.add_argument('--device', type=str, default='cuda:0', help="")
parser.add_argument('--num_works', type=int, default=8, help="")
parser.add_argument('--seed', type=int, default=3450, help="")
parser.add_argument('--modle_path', type=str, default="", help="")

args = get_args(parser) # 读取yaml文件中的参数，并以字典形式保存在args变量中
print("\n================== Arguments =================")
pprint.pprint(vars(args), indent=4)
print("==========================================\n")
save_args(args) # 保存参数

# 系统设置
seed_torch(args.seed) # 初始cudnn加速，也可设置随机种子，此处没有设置随机种子

cuda = args.device.split(':')
os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_cuda # 全局修改模型占用的显卡
os.environ['CUDA_LAUNCH_BLOCKING'] = args.visible_cuda
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['WANDB_API_KEY'] = args.wandb_key # 设置wandb的API的key

# wandb登录
wandb.login()

# 运行器初始化
runner = Runner(args)

if args.exp_name == "h36m":
    acts = define_actions(args.test_manner)
elif args.exp_name == "cmu":
    acts = define_actions_cmu(args.test_manner)

if args.is_load:
    runner.restore(args.modle_path)
    file = open(args.modle_path+"run_id.txt", 'r')
    run_id = file.readlines()
    run = wandb.init(project=args.wandb_project, id=run_id, resume='must')

if args.is_train:
    # wandb设置
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    wandb.init(project=args.wandb_project, config=vars(args), name=nowtime, save_code=True)
    run_id = wandb.run.id
    with open(args.work_dir+"run_id.txt", 'w') as f:
        f.write(run_id)
    
    # wandb版本管理
    # wandb保存配置文件
    arti_config = wandb.Artifact("config", type='yaml')
    arti_config.add_file(args.work_dir + "/config.yaml")
    wandb.log_artifact(arti_config)
    # wandb保存模型文件
    arti_modle = wandb.Artifact("modle", type='code')
    x = args.model.split('.')
    arti_modle.add_file(x[0] + '/' + x[1] + '/' + x[2] + '.py')
    wandb.log_artifact(arti_modle)
    # wandb保存运行文件
    arti_runner = wandb.Artifact("runner", type='code')
    arti_config.add_file("/runner/runner.py")
    wandb.log_artifact(arti_runner)
    
    # 开始训练
    runner.run()
    
    # wandb结束
    wandb.finish()
else:
    errs = runner.test()

    col = args.frame_ids
    d = pd.DataFrame(errs, index=acts, columns=col)
    d.to_csv(f"{args.exp_name}_in{args.input_n}out{args.output_n}dctn{args.dct_n}_{args.test_manner}.csv", line_terminator="\n")