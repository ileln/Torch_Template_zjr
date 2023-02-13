import torch
import torch.nn as nn
import numpy as np

from .layers import GC_Block, PreGCN, BidirectionalLSTM

class Cluster(nn.Module):
    
    def __init__(self,p_dropout, leaky_c=0.2, final_out_noden=22, input_feature=35, hidden_feature=64, **dic):
        super(Cluster, self).__init__()
        
        self.p_dropout = p_dropout
        self.leaky_c= leaky_c
        self.final_out_noden = final_out_noden
        self.input_feature = input_feature
        self.hidden_feature = hidden_feature
        
        self.time_enhance = PreGCN(input_feature=self.input_feature, hidden_feature=64, node_n=self.final_out_noden*3, p_dropout=self.p_dropout, leaky_c=self.leaky_c)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_feature, out_channels=128, kernel_size=1, padding=0),
            nn.LeakyReLU(self.leaky_c),
            nn.MaxPool1d(kernel_size=5, padding=2),
            BidirectionalLSTM(input_size=128, hidden_size=128, output_size=128),
            BidirectionalLSTM(input_size=128, hidden_size=64, output_size=64)
        )
        self.decoder = nn.Sequential(
            nn.Upsample(size=5),
            nn.ConvTranspose1d(),
            
        )