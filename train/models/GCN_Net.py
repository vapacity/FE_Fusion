# -- coding: utf-8 --**
# GCN model

import torch

import torch.nn.functional as F
import torch_geometric.transforms as T
import torch_geometric
from torch_geometric.nn import voxel_grid, max_pool, max_pool_x, GMMConv
import os
import sys
from models.MF_Net import NetVLAD
from torch_scatter import scatter_max
from torch_geometric.nn import global_max_pool

def custom_voxel_grid(pos, voxel_size, batch=None):
    """
    自定义 voxel grid 方法，支持不等尺寸的 voxel（各轴可设定不同尺寸）。

    参数:
        pos (Tensor): 点坐标，形状为 (N, 3)
        voxel_size (Tensor or list): voxel 的 x, y, z 尺寸，例如 [36, 36, 20]
        batch (Tensor, optional): 每个点的 batch 编号，形状为 (N,)

    返回:
        cluster (Tensor): 每个点所属的 voxel 编号，形状为 (N,)
    """

    # 转换 voxel_size 为张量
    voxel_size = torch.tensor(voxel_size, dtype=pos.dtype, device=pos.device)
    
    # 计算每个点在哪个 voxel（浮点 -> floor -> 整数 voxel 索引）
    voxel_indices = torch.floor(pos / voxel_size).long()  # (N, 3)
    
    if batch is not None:
        # 添加 batch 维度，确保每个 batch 单独 voxel 编号
        voxel_indices = torch.cat([batch.view(-1, 1), voxel_indices], dim=1)  # (N, 4)
    
    # 使用 unique 返回每个 voxel 的唯一编号
    unique_voxels, cluster = torch.unique(voxel_indices, return_inverse=True, dim=0)
    
    return cluster, unique_voxels



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

class GCN_Net(torch.nn.Module):
    def __init__(self):
        super(GCN_Net, self).__init__()
        self.extractor = GCN_Extractor()

    def forward(self, data):
        x = self.extractor(data)
        return x

class GCN_Extractor_v2(torch.nn.Module):
    def __init__(self):
        super(GCN_Extractor_v2, self).__init__()
        self.conv1 = GMMConv(1, 64, dim=3, kernel_size=5)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.block1 = GraphResidualBlock(64, 128)
        self.block2 = GraphResidualBlock(128, 256)
        self.block3 = GraphResidualBlock(256, 512)
        self.input_dim = 512
        self.fc = torch.nn.Linear(self.input_dim, 256)
        self.VLAD = NetVLAD(dim=256)

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
        cluster = voxel_grid(pos=data.pos, batch=data.batch, size=32)      
        x = max_pool_x(cluster, data.x, data.batch, size=64)
        # 取0是因为返回值为(x, None)
        # x[0] size为 B x 64, 512
        batch_size = data.batch.max().item() + 1
        x = x[0].view(batch_size, 64, 512)  # B x 64 x 512
        x = self.fc(x)                      # B x 64 x 256
        x = x.permute(0, 2, 1).view(batch_size, 256, 8, 8)  # B x 256 x 8 x 8
        x = self.VLAD(x)
        return x

class GCN_Extractor(torch.nn.Module):
    def __init__(self):
        super(GCN_Extractor, self).__init__()
        self.conv1 = GMMConv(1, 64, dim=3, kernel_size=5)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.block1 = GraphResidualBlock(64, 128)
        self.block2 = GraphResidualBlock(128, 256)
        self.block3 = GraphResidualBlock(256, 512)
        self.input_dim = 16 * 512
        self.fc = torch.nn.Linear(self.input_dim, 4096)

    def forward(self, data):
        conv1_output = self.conv1(data.x, data.edge_index, data.edge_attr)  # 相当于每个节点做 1*1 升维
        data.x = F.elu(self.bn1(conv1_output))
        cluster = voxel_grid(pos=data.pos, batch=data.batch, size=4)    # 4*4*4
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data = self.block1(data)
        cluster = voxel_grid(pos=data.pos, batch=data.batch, size=6)    # 6*6*6
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data = self.block2(data)
        cluster = voxel_grid(pos=data.pos, batch=data.batch, size=24)   # 24*24*24
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data = self.block3(data)
        cluster = voxel_grid(pos=data.pos, batch=data.batch, size=64)   # 260 / 64 = 4, 360 
        # test_x = max_pool(cluster, data, transform=T.Cartesian(cat=False))
        # breakpoint()

        x = max_pool_x(cluster, data.x, data.batch, size=16) # 聚类成16个类 16*512
        # 取0是因为返回值为(x, None)
        x = x[0].view(-1, self.input_dim)
        x = self.fc(x)  # B 4096 (8192->4096)
        x = F.normalize(x, p=2, dim=1)
        return x