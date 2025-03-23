import models.TSFE_Net as TSFE_Net
import models.MF_Net as MF_Net
import models.DRW_Net as DRW_Net
import torch
import torch.nn as nn
from models.MF_Net import NetVLAD
import models.CBAM as CBAM




class DiftNet(nn.Module):
    def __init__(self, channel_sizes):
        super(DiftNet, self).__init__()
        self.VLAD_0 = NetVLAD(dim=256)
        self.VLAD_1 = NetVLAD(dim=256)
        self.VLAD_2 = NetVLAD(dim=256)
        self.drw_net = DRW_Net.DRW_Net()  # 假设 DRW_Net 在 DRW_Net 模块中定义
        self.conv1x1_0 = nn.Conv2d(1280, 256, kernel_size=1)
        self.conv1x1_1 = nn.Conv2d(1280, 256, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(640, 256, kernel_size=1)


    def forward(self, dift_feat_0, dift_feat_1, dift_feat_2):
        # DRW_Net 的前向传播
        M1 = self.VLAD_0(self.conv1x1_0(dift_feat_0))
        M2 = self.VLAD_1(self.conv1x1_1(dift_feat_1))
        M3 = self.VLAD_2(self.conv1x1_2(dift_feat_2))
        drw_output = self.drw_net(M1, M2, M3)
        # 返回主网络的输出
        return drw_output


