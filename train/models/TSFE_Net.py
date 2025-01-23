import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
"""
TSFE-Net
Input: Frame and Event Volumes (from Brisbane dataset 25ms timestamp;25ms time interval window)
Process:
        1.Two streams
            Each:  Conv(7×7,64,/2)-Attn(5×5)-MaxPool2d(/2)-ResBlock0(3×3,64)-ResBlock1(3×3,64)-ResBlock2(3×3,64)-Attn(5×5)-BatchNorm-ReLU.
        2.Concatenate
Details:
        1.Conv: kernel=7x7 channel=64 stride=2
        2.ResBlock: same as ResNet34
        3.Attn: CBAM
        4.MaxPool2d: 2x2 stride=2
"""
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
        
    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.conv1 = BasicConv(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = BasicConv(in_channels, in_channels, kernel_size=3, padding=1, relu=False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        return F.relu(out)

class TSFE_Net(nn.Module):
    def __init__(self, mid_channels=64, use_frame=True, use_event=True):
        super(TSFE_Net, self).__init__()
        self.use_frame = use_frame  # 标志，指示是否使用 frame 数据
        self.use_event = use_event  # 标志，指示是否使用 event 数据

        # 定义 frame 和 event 的处理流
        self.conv1_frame = BasicConv(1, mid_channels, kernel_size=7, stride=2, padding=3)
        self.conv1_event = BasicConv(2, mid_channels, kernel_size=7, stride=2, padding=3)
        self.attn1 = CBAM(mid_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks = nn.Sequential(ResBlock(mid_channels), ResBlock(mid_channels), ResBlock(mid_channels))
        self.attn2 = CBAM(mid_channels)
        self.bn = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward_stream(self, x, is_event=False):
        if is_event:
            x = self.conv1_event(x)
        else:
            x = self.conv1_frame(x)
        x = self.attn1(x)
        x = self.maxpool(x)
        x = self.resblocks(x)
        x = self.attn2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def forward(self, frame, event):
        if self.use_frame and self.use_event:
            # 使用 frame 和 event 数据
            frame_features = self.forward_stream(frame, is_event=False)
            event_features = self.forward_stream(event, is_event=True)
            merged_features = torch.cat([frame_features, event_features], dim=1)
            return merged_features
        elif self.use_frame:
            # 仅使用 frame 数据，将特征复制一遍以加倍通道数
            frame_features = self.forward_stream(frame, is_event=False)
            doubled_features = torch.cat([frame_features, frame_features], dim=1)
            return doubled_features
        elif self.use_event:
            # 仅使用 event 数据，将特征复制一遍以加倍通道数
            event_features = self.forward_stream(event, is_event=True)
            doubled_features = torch.cat([event_features, event_features], dim=1)
            return doubled_features
        else:
            raise ValueError("必须至少启用 'use_frame' 或 'use_event' 其中之一。")


# # test for TESNET
# height = 256
# width =256
# batch_size=1

# event_volume = torch.rand(batch_size, 1,height,width)
# frame = torch.rand(batch_size,1,height,width)
# event_volume = event_volume.repeat(64,2,1,1)
# frame = frame.repeat(64,1,1,1)

# model = TSFE_Net(use_event=False)

# # 检查是否有可用的 GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# event_volume = event_volume.to(device)
# frame = frame.to(device)

# # 将数据输入网络进行前向传播
# output = model(frame, event_volume)

# # 输出结果的维度
# print("Output shape:", output.shape)

