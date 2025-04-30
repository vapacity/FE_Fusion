# -- coding: utf-8 --**
# GCN model

import torch

import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import voxel_grid, max_pool, max_pool_x, GMMConv
import os
import sys
from models.MF_Net import NetVLAD
from torch_scatter import scatter_max

class ChannelFCBlock(torch.nn.Module):
    def __init__(self, in_channels=512, hidden_dim=1024, out_channels=256):
        super(ChannelFCBlock, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden_dim, kernel_size=1),  # 相当于 fc1
            torch.nn.BatchNorm2d(hidden_dim),
            torch.nn.Dropout(),
            torch.nn.ELU(inplace=True),
            torch.nn.Conv2d(hidden_dim, out_channels, kernel_size=1)  # 相当于 fc2
        )

    def forward(self, x):
        return self.block(x)


class GraphResidualBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GraphResidualBlock, self).__init__()
        self.left_conv1 = GMMConv(in_channel, out_channel, dim=3, kernel_size=5)
        self.left_bn1 = torch.nn.BatchNorm1d(out_channel)
        self.left_conv2 = GMMConv(out_channel, out_channel, dim=3, kernel_size=5)
        self.left_bn2 = torch.nn.BatchNorm1d(out_channel)

        self.shortcut_conv = GMMConv(in_channel, out_channel, dim=3, kernel_size=1)
        self.shortcut_bn = torch.nn.BatchNorm1d(out_channel)

    def forward(self, data):
        data.x = F.elu(self.left_bn2(
            self.left_conv2(F.elu(self.left_bn1(self.left_conv1(data.x, data.edge_index, data.edge_attr))),
                            data.edge_index, data.edge_attr)) + self.shortcut_bn(
            self.shortcut_conv(data.x, data.edge_index, data.edge_attr)))

        return data

def spatial_pool_NxN_scatter(data, N=16):
    pos = data.pos[:, :2]  # 只取XY
    batch = data.batch     # [N]
    x = data.x             # [N, C]
    B = int(batch.max()) + 1
    N_pts, C = x.size()

    # 归一化每个batch的pos到 [0, 1]
    batch_min = scatter_max(-pos, batch, dim=0)[0] * (-1)
    batch_max = scatter_max(pos, batch, dim=0)[0]

    batch_min = batch_min[batch]
    batch_max = batch_max[batch]

    norm_pos = (pos - batch_min) / (batch_max - batch_min + 1e-6)

    # 网格划分（N x N）
    grid_idx = (norm_pos * N).long().clamp(max=N - 1)
    cluster_id = grid_idx[:, 0] * N + grid_idx[:, 1]  # [0, N*N-1]

    final_cluster = batch * (N * N) + cluster_id

    # 聚合
    pooled, _ = scatter_max(x, final_cluster, dim=0, dim_size=B * N * N)  # [B*N*N, C]

    # reshape: [B, N*N, C] → [B, C, N, N]
    pooled = pooled.view(B, N * N, C).transpose(1, 2).view(B, C, N, N)

    # 替换掉 -inf（如果有空块）
    pooled = torch.nan_to_num(pooled, nan=0.0, neginf=0.0, posinf=0.0)

    return pooled

class GCN_Net(torch.nn.Module):
    def __init__(self):
        super(GCN_Net, self).__init__()
        self.extractor = GCN_Extractor()
        # self.channel_fc_block = ChannelFCBlock(out_channels=256)
        # self.VLAD = NetVLAD(dim=256)
        self.fc = torch.nn.Linear(8 * 512, 2048)

    def forward(self, data):
        x = self.extractor(data)
        x = x[0].view(-1, self.fc.weight.size(1))   # batch_size * 4096
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        
        # x = self.channel_fc_block(x)
        # x = self.VLAD(x)
        return x

class GCN_Extractor(torch.nn.Module):
    def __init__(self):
        super(GCN_Extractor, self).__init__()
        self.conv1 = GMMConv(1, 64, dim=3, kernel_size=5)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.block1 = GraphResidualBlock(64, 128)
        self.block2 = GraphResidualBlock(128, 256)
        self.block3 = GraphResidualBlock(256, 512)

        # self.fc1 = torch.nn.Linear(8 * 512, 1024)
        # self.bn = torch.nn.BatchNorm1d(1024)
        # self.drop_out = torch.nn.Dropout()
        # self.fc2 = torch.nn.Linear(1024, 256)

    def forward(self, data):
        conv1_output = self.conv1(data.x, data.edge_index, data.edge_attr)  # 相当于每个节点做 1*1 升维
        data.x = F.elu(self.bn1(conv1_output))
        cluster = voxel_grid(pos=data.pos, batch=data.batch, size=4)
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data = self.block1(data)
        cluster = voxel_grid(pos=data.pos, batch=data.batch, size=6)
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data = self.block2(data)
        cluster = voxel_grid(pos=data.pos, batch=data.batch, size=24)
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data = self.block3(data)
        cluster = voxel_grid(pos=data.pos, batch=data.batch, size=64)
        # x = spatial_pool_NxN_scatter(data, N=16)
        x = max_pool_x(cluster, data.x, data.batch, size=8)


        return x