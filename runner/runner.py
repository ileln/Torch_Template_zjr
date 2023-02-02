import torch
import torch.nn as nn

# 运行器
class Runner():
    def __init__(self, args):
        super(Runner, self).__init__()

        # 参数
        self.start_epoch = 1
        self.best_accuracy = 1e15
        

