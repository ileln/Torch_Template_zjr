import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, **dic):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(3, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 16)
        self.layer5 = nn.Linear(16 * 543, 250)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=2, end_dim=-1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.flatten(x)
        x = self.layer5(x)
        return x