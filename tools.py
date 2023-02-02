import sys
import traceback
import yaml
import argparse
import torch

def import_class(import_str):
    # 用字符形式更加方便地调用自己声明的类
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def read_yaml(path):
    # 读取yaml文件并转化为字典
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)
    return dict

def read_args():
    # 读取命令行中的参数
    parser = argparse.ArgumentParser(description='Arguments for running the scripts')

    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--exp_name', type=str, default="h36m", help="h36m / cmu")
    parser.add_argument('--input_n', type=int, default=10, help="")
    parser.add_argument('--output_n', type=int, default=25, help="")
    parser.add_argument('--dct_n', type=int, default=35, help="")
    parser.add_argument('--device', type=str, default="0", help="")
    parser.add_argument('--num_works', type=int, default=0)
    # parser.add_argument('--train_manner', type=str, default="all", help="all / 8")
    parser.add_argument('--test_manner', type=str, action="append")
    parser.add_argument('--train_manner', type=str, action="append")
    # parser.add_argument('--train_manner', type=str)
    # parser.add_argument('--debug_step', type=int, default=1, help="")
    parser.add_argument('--is_train', type=bool, default=True, help="")
    parser.add_argument('--is_load', type=bool, default='', help="")
    parser.add_argument('--model_path', type=str, default="", help="")

    args = parser.parse_args()
    return args

def get_args(path):
    # 合并命令行和yaml文件中的参数，命令行中的参数优先
    args = read_args() # 读取命令行指定的参数
    opt = vars(args) # 将命令行中的参数转化为字典
    args = read_yaml(path) # 读取yaml文件的中参数并转化为字典
    opt.update(args) # 合并字典
    args = opt # 更新字典

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