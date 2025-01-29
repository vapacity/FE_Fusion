import torch
import torch.nn as nn
from torchvision.models import resnet34
from MF_Net import NetVLAD

# 标准Res34实现
class VPRFE_Net(nn.Module):
    def __init__(self, vlad_dim=512, C=9):
        super(VPRFE_Net, self).__init__()
        
        # 加载标准 ResNet34
        resnet = resnet34(pretrained=False)
        
        # 替换初始卷积层
        self.conv1 = nn.Conv2d(2 * C, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # BN+Relu是否不加
        # self.bn1 = resnet.bn1
        # self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 最大池化层

        # 替换残差块
        self.layer1 = resnet.layer1  # conv2_x: (batch_size, 64, 64, 64)
        self.layer2 = resnet.layer2  # conv3_x: (batch_size, 128, 32, 32)
        self.layer3 = resnet.layer3  # conv4_x: (batch_size, 256, 16, 16)
        self.layer4 = resnet.layer4  # conv5_x: (batch_size, 512, 8, 8)
        self.VLAD = NetVLAD(dim=vlad_dim)

    def forward(self, x):
        # 初始卷积层 + BN + ReLU + 最大池化
        x = self.conv1(x)
        x = self.maxpool(x)

        # ResNet 残差块
        x = self.layer1(x)  # conv2_x
        x = self.layer2(x)  # conv3_x
        x = self.layer3(x)  # conv4_x
        x = self.layer4(x)  # conv5_x
        x = NetVLAD(x)
        
        return x