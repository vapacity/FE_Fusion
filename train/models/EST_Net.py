import torch
import torch.nn as nn
from bin_to_voxel import *

class EST_Net(nn.Module):
    def __init__(self, use_adapter=False):
        super(EST_Net, self).__init__()
        # MLP部分
        self.est_mlp = nn.Sequential(
            nn.Linear(1, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 1)
        )
        self.use_adapter = use_adapter
        if use_adapter:
            self.adapter_conv = nn.Conv2d(in_channels=18, out_channels=2, kernel_size=1, bias=False) # 测试用
    
    def forward(self, event_volume):
        if self.use_adapter:
            return self.adapter_conv(EST_voxel_grid(event_volume, self.est_mlp))
        else:
            return EST_voxel_grid(event_volume, self.est_mlp)
