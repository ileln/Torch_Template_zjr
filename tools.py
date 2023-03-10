import sys
import traceback
import yaml
import argparse
import torch
import os

# 代码工具
def import_class(import_str):
    # 用字符形式更加方便地调用自己声明的类
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

class DictAction(argparse.Action):
    # CTR中的覆盖词典操作
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(DictAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        input_dict = eval(f'dict({values})')  #pylint: disable=W0123
        output_dict = getattr(namespace, self.dest)
        for k in input_dict:
            output_dict[k] = input_dict[k]
        setattr(namespace, self.dest, output_dict)

# 参数工具
def read_yaml(path):
    # 读取yaml文件并转化为字典
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)
    return dict

# def read_args():
#     # 读取命令行中的参数
#     parser = argparse.ArgumentParser(description='Arguments for running the scripts')

#     parser = argparse.ArgumentParser(description='manual to this script')
#     parser.add_argument('--config', type=str, default="config/MSRGCN/h36m/demo.yaml", help="配置文件")
#     parser.add_argument('--work_dir', type=str, default="./workdir/MSRGCN/h36m/test1", help="实验文件保存路径")
#     parser.add_argument('--exp_name', type=str, default="h36m", help="h36m / cmu")
#     parser.add_argument('--h36m', type=str, action=DictAction, default=dict(), help="the arguments of data loader for h36m")
#     parser.add_argument('--cmu', type=str, action=DictAction, default=dict(), help="the arguments of data loader for cmu")
#     parser.add_argument('--subs', type=list, default=[[1, 6, 7, 8, 9], [5], [11]], help="数据集使用")
#     parser.add_argument('--train_split', type=int, default=0, help="对应subs中的选项")
#     parser.add_argument('--teat_split', type=int, default=1, help="")
#     parser.add_argument('--validation', type=int, default=2, help="")
#     parser.add_argument('--sample_rate', type=int, default=2, help="抽帧")
#     parser.add_argument('--down_key', type=list, default=[], help="")
#     parser.add_argument('--global_max', type=int, default=0, help="")
#     parser.add_argument('--global_min', type=int, default=0, help="")
#     parser.add_argument('--debug_step', type=int, default=1, help="")
#     parser.add_argument('--modle', type=str, default='models.MSRGCN.MSRGCN', help="模型选择")
#     parser.add_argument('--lr', type=float, default=2e-4, help="")
#     parser.add_argument('--lr_decay', type=float, default=0.98, help="")
#     parser.add_argument('--n_epoch', type=int, default=5000)
#     parser.add_argument('--leaky_c', type=float, default=0.2)
#     parser.add_argument('--optimizer', type=str, default='optim.Adam')
#     parser.add_argument('--ml_weight',type=float, default=0.3)
#     parser.add_argument('--p_dropout',type=float, default=0.1, help="")
#     parser.add_argument('--train_batch_size', type=int, default=16, help="")
#     parser.add_argument('--test_batch_size', type=int, default=128, help="")
#     parser.add_argument('--input_n', type=int, default=10, help="")
#     parser.add_argument('--output_n', type=int, default=25, help="")
#     parser.add_argument('--seq_len', type=int, default=35, help="")
#     parser.add_argument('--dct_n', type=int, default=35, help="")
#     parser.add_argument('--device', type=str, default='0', help="")
#     parser.add_argument('--num_works', type=int, default=8, help="")
#     parser.add_argument('--seed', type=int, default=3450, help="")


#     # args = parser.parse_args()
#     return parser

def get_args(parser):
    # 合并命令行和yaml文件中的参数，命令行中的参数优先
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
            print(default_arg)
        dic_p = vars(p)
        default_arg.update(dic_p)
    args = argparse.Namespace(**default_arg)
    return args

def save_args(args):
    # 保存参数
    arg_dict = vars(args)
    if not os.path.exists(str(args.work_dir)):
        os.makedirs(str(args.work_dir))
    with open('{}/config.yaml'.format(str(args.work_dir)), 'w') as f:
        f.write(f"# command line: {' '.join(sys.argv)}\n\n")
        yaml.dump(arg_dict, f)

# 训练工具
def seed_torch(seed=3450):
    # 设置torch随机种子和cudnn加速
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

def lr_decay(optimizer, lr_now, gamma):
    # 衰减学习率函数
    lr = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def lr_decay(optimizer, lr_now, gamma):
    lr = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr