import torch
import torch.nn as nn
import numpy as np


class Cluster(nn.Module):
    
    def __init__(self, p_dropout, leaky_c=0.2, final_out_noden=22, input_feature=35, **dic):
        super(Cluster, self).__init__()
        
        