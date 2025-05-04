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


# core steps
channel_sizes = [128,256,512]
class MF_MainNet(nn.Module):
    def __init__(self, channel_sizes):
        super(MF_MainNet, self).__init__()
        resnet = resnet34(pretrained=False)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attn1 = CBAM.CBAM(channel_sizes[0])
        # Assuming each conv_x module doubles the number of channels and halves the feature map size
        self.conv4_x = resnet.layer3
        self.attn2 = CBAM.CBAM(channel_sizes[1])
        self.conv5_x = resnet.layer4
        self.conv1x1 = nn.Conv2d(channel_sizes[2],256,kernel_size=1)

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
        M1_unvlad = self.conv1x1(S3)
        return S1,S2,S3,M1_unvlad
    
    
class MF_SubNet1(nn.Module):
    def __init__(self,channel_sizes):
        super(MF_SubNet1,self).__init__()
        self.conv1x1_s1 = nn.Conv2d(channel_sizes[0],128,kernel_size=1)
        self.conv1x1_s2 = nn.Conv2d(channel_sizes[1],128,kernel_size=1)
        self.bn=nn.BatchNorm2d(128)
        self.relu=nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2,mode='nearest')
        self.attn = CBAM.CBAM(256)
        

    def forward(self, s1, s2):
        s1_processed = self.conv1x1_s1(s1)
        s1_processed = self.bn(s1_processed)
        s1_processed = self.relu(s1_processed)
        
        s2_processed = self.conv1x1_s2(s2)
        s2_upsampled = self.upsample(s2_processed)

        s2_upsampled = self.bn(s2_upsampled)
        s2_upsampled = self.relu(s2_upsampled)
        
        # 将处理后的S1和S2拼接在一起
        fused_features = torch.cat([s1_processed, s2_upsampled], dim=1)
        M3_unvlad=self.attn(fused_features)
        #print("M3:",features_processed.shape)
        return M3_unvlad, s2_processed
    

class MF_SubNet2(nn.Module):
    def __init__(self,channel_sizes):
        super(MF_SubNet2,self).__init__()
        self.conv1x1_s3 = nn.Conv2d(channel_sizes[2],128,kernel_size=1)
        self.bn=nn.BatchNorm2d(128)
        self.relu=nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2,mode='nearest')
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
        
        M2_unvlad=self.attn2(fused_features)

        return M2_unvlad