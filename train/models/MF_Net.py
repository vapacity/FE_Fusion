import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torchvision.models import resnet34
import numpy as np
from PIL import Image
import models.CBAM as CBAM
"""
Multi-Scale Fusion Network
Input: Result from TSFE-Net
Process:As image shows
Details:
"""
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out
    
# class NetVLAD(nn.Module):
#     """NetVLAD layer implementation"""

#     def __init__(self, num_clusters=64, dim=128, 
#                  normalize_input=True, vladv2=False):
#         """
#         Args:
#             num_clusters : int
#                 The number of clusters
#             dim : int
#                 Dimension of descriptors
#             alpha : float
#                 Parameter of initialization. Larger value is harder assignment.
#             normalize_input : bool
#                 If true, descriptor-wise L2 normalization is applied to input.
#             vladv2 : bool
#                 If true, use vladv2 otherwise use vladv1
#         """
#         super(NetVLAD, self).__init__()
#         self.num_clusters = num_clusters
#         self.dim = dim
#         self.alpha = 0
#         self.vladv2 = vladv2
#         self.normalize_input = normalize_input
#         self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=not vladv2)
#         self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

#     def init_params(self, clsts, traindescs):
#         #TODO replace numpy ops with pytorch ops
#         if self.vladv2 == False:
#             clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
#             dots = np.dot(clstsAssign, traindescs.T)
#             dots.sort(0)
#             dots = dots[::-1, :] # sort, descending

#             self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
#             self.centroids = nn.Parameter(torch.from_numpy(clsts))
#             self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))
#             self.conv.bias = None
#         else:
#             knn = NearestNeighbors(n_jobs=-1) #TODO faiss?
#             knn.fit(traindescs)
#             del traindescs
#             dsSq = np.square(knn.kneighbors(clsts, 2)[1])
#             del knn
#             self.alpha = (-np.log(0.01) / np.mean(dsSq[:,1] - dsSq[:,0])).item()
#             self.centroids = nn.Parameter(torch.from_numpy(clsts))
#             del clsts, dsSq

#             self.conv.weight = nn.Parameter(
#                 (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
#             )
#             self.conv.bias = nn.Parameter(
#                 - self.alpha * self.centroids.norm(dim=1)
#             )

#     def forward(self, x):
#         N, C = x.shape[:2]

#         if self.normalize_input:
#             x = F.normalize(x, p=2, dim=1)  # across descriptor dim

#         # soft-assignment
#         soft_assign = self.conv(x).view(N, self.num_clusters, -1)
#         soft_assign = F.softmax(soft_assign, dim=1)

#         x_flatten = x.view(N, C, -1)
        
#         # calculate residuals to each clusters
#         vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
#         for C in range(self.num_clusters): # slower than non-looped, but lower memory usage 
#             residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
#                     self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
#             residual *= soft_assign[:,C:C+1,:].unsqueeze(2)
#             vlad[:,C:C+1,:] = residual.sum(dim=-1)

#         vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
#         vlad = vlad.view(x.size(0), -1)  # flatten
#         vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

#         return vlad


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, alpha=100.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)
        
        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad

# core steps
channel_sizes = [128,256,512]
class MF_MainNet(nn.Module):
    def __init__(self, channel_sizes):
        super(MF_MainNet, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attn1 = CBAM.CBAM(channel_sizes[0])
        self.VLAD = NetVLAD(dim=channel_sizes[1])
        # Assuming each conv_x module doubles the number of channels and halves the feature map size
        self.conv4_x = self._make_layer(channel_sizes[0], channel_sizes[1], blocks=6, stride=2)
        self.attn2 = CBAM.CBAM(channel_sizes[1])
        self.conv5_x = self._make_layer(channel_sizes[1], channel_sizes[2], blocks=3, stride=2)
        self.conv1x1 = nn.Conv2d(channel_sizes[2],256,kernel_size=1)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.attn1(x)
        S1 = x
        #print("S1:",S1.shape)
        x = self.conv4_x(x)
        x = self.attn2(x)
        S2 = x
        #print("S2:",S2.shape)
        x = self.conv5_x(x)
        S3 = x
        #print("S3:",S3.shape)
        S3_256 = self.conv1x1(S3)
        M1 = self.VLAD(S3_256)
        return S1,S2,S3,M1
    
    
class MF_SubNet1(nn.Module):
    def __init__(self,channel_sizes):
        super(MF_SubNet1,self).__init__()
        self.conv1x1_s1 = nn.Conv2d(channel_sizes[0],128,kernel_size=1)
        self.conv1x1_s2 = nn.Conv2d(channel_sizes[1],128,kernel_size=1)
        self.bn=nn.BatchNorm2d(128)
        self.relu=nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2,mode='nearest')
        self.VLAD = NetVLAD(dim=256)
        self.attn = CBAM.CBAM(256)
        

    def forward(self, s1, s2):
        s1_processed = self.conv1x1_s1(s1)
        s1_processed = self.bn(s1_processed)
        s1_processed = self.relu(s1_processed)
        
        s2_processed = self.conv1x1_s2(s2)
        x=s2_processed
        s2_upsampled = self.upsample(s2_processed)

        s2_upsampled = self.bn(s2_upsampled)
        s2_upsampled = self.relu(s2_upsampled)
        
        # 将处理后的S1和S2拼接在一起
        fused_features = torch.cat([s1_processed, s2_upsampled], dim=1)
        features_processed=self.attn(fused_features)
        #print("M3:",features_processed.shape)
        M3 = self.VLAD(features_processed)
        return M3, s2_processed
    

class MF_SubNet2(nn.Module):
    def __init__(self,channel_sizes):
        super(MF_SubNet2,self).__init__()
        self.conv1x1_s3 = nn.Conv2d(channel_sizes[2],128,kernel_size=1)
        self.bn=nn.BatchNorm2d(128)
        self.relu=nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2,mode='nearest')
        self.VLAD = NetVLAD(dim = 256)
        self.attn1 = CBAM.CBAM(128)
        self.attn2 = CBAM.CBAM(256)

    def forward(self, s2, s3):
        s2_processed = self.bn(s2)
        s2_processed = self.relu(s2_processed)
        #print('S3:',s3.shape)
        s3_processed = self.conv1x1_s3(s3)
        s3_processed = self.bn(s3_processed)
        s3_processed = self.relu(s3_processed)
        s3_processed = self.attn1(s3_processed)

        s3_upsampled = self.upsample(s3_processed)

        s3_upsampled = self.bn(s3_upsampled)
        s3_upsampled = self.relu(s3_upsampled)
        
        # 将处理后的S1和S2拼接在一起
        fused_features = torch.cat([s2_processed, s3_upsampled], dim=1)
        
        processed_features=self.attn2(fused_features)
        #print("M2:",processed_features.shape)
        M2=self.VLAD(processed_features)
        return M2
'''
# test for Multi-Scale Fusion Network
height = 64
width = 64
batch_size=1

frame = torch.rand(batch_size, 1,height,width)

frame = frame.repeat(1,128,1,1)

model_mainNet = MF_MainNet(channel_sizes)
model_subNet1 = MF_SubNet1(channel_sizes)
model_subNet2 = MF_SubNet2(channel_sizes)

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_mainNet.to(device)
model_subNet1.to(device)
model_subNet2.to(device)
frame = frame.to(device)

# 将数据输入网络进行前向传播
S1,S2,S3,M1 = model_mainNet(frame)
print("Output shape:\nS1:", S1.shape," S2:",S2.shape," S3:",S3.shape)
M3,processed_S2 =model_subNet1(S1,S2)
M2 = model_subNet2(processed_S2,S3)
print("M1: ",M1.shape,"M2: ",M2.shape," M3:",M3.shape)
'''