import torch
import torch.nn as nn
from os.path import join, dirname, isfile
import tqdm
import numpy as np


class ValueLayer(nn.Module):
    def __init__(self, mlp_layers, activation=nn.ReLU(), num_channels=9):
        assert mlp_layers[-1] == 1, "Last layer of the mlp must have 1 input channel."
        assert mlp_layers[0] == 1, "First layer of the mlp must have 1 output channel"

        nn.Module.__init__(self)
        self.mlp = nn.ModuleList()
        self.activation = activation

        # create mlp
        in_channels = 1
        for out_channels in mlp_layers[1:]:
            self.mlp.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels

        # init with trilinear kernel
        path = "/root/autodl-tmp/FE_Fusion/train/models/trilinear_init.pth"
        if isfile(path):
            state_dict = torch.load(path)
            self.load_state_dict(state_dict)
        else:
            self.init_kernel(num_channels)

    def forward(self, x):
        # 加上batch维度和channel维度
        x = x[None,...,None]

        # apply mlp convolution
        for i in range(len(self.mlp[:-1])):
            x = self.activation(self.mlp[i](x))

        x = self.mlp[-1](x)
        x = x.squeeze()

        return x

    def init_kernel(self, num_channels):
        ts = torch.zeros((1, 2000))
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)

        torch.manual_seed(1)

        for _ in tqdm.tqdm(range(1000)):  # converges in a reasonable time
            optim.zero_grad()

            ts.uniform_(-1, 1)

            # gt
            gt_values = self.trilinear_kernel(ts, num_channels)

            # pred
            values = self.forward(ts)

            # optimize
            loss = (values - gt_values).pow(2).sum()

            loss.backward()
            optim.step()


    def trilinear_kernel(self, ts, num_channels):
        gt_values = torch.zeros_like(ts)

        gt_values[ts > 0] = (1 - (num_channels-1) * ts)[ts > 0]
        gt_values[ts < 0] = ((num_channels-1) * ts + 1)[ts < 0]

        gt_values[ts < -1.0 / (num_channels-1)] = 0
        gt_values[ts > 1.0 / (num_channels-1)] = 0

        return gt_values

class EST_Net(nn.Module):
    def __init__(self, 
                 voxel_dim = [9, 260, 346],  # C=9, H=260, W=346, 
                 mlp_layers=[1, 30, 30, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1),
                 use_adapter=False):
        super(EST_Net, self).__init__()
        # MLP部分
        self.use_adapter = use_adapter
        if use_adapter:
            self.adapter_conv = nn.Conv2d(in_channels=18, out_channels=2, kernel_size=1, bias=False) # 测试用
        self.value_layer = ValueLayer(mlp_layers,
                                      activation=activation,
                                      num_channels=voxel_dim[0])
        self.voxel_dim = voxel_dim
    
    def forward(self, events):
        # events is a tensor of shape (N, 5), where N is the number of events
        # 这里的期望输入是所有的tensor已经拼起来了
        # points is a list, since events can have any size
        B = int((1+events[-1,-1]).item())   # 最后的那个B应该是Batch_index里最大的缩影，就是batch_size
        num_voxels = int(2 * np.prod(self.voxel_dim) * B)
        vox = events[0].new_full([num_voxels,], fill_value=0)
        C, H, W = self.voxel_dim

        # get values for each channel
        x, y, t, p, b = events.t()  # 转置

        # normalizing timestamps
        for bi in range(B):
            t[events[:,-1] == bi] /= t[events[:,-1] == bi].max()

        p = (p+1)/2  # maps polarity to 0, 1

        idx_before_bins = x \
                          + W * y \
                          + 0 \
                          + W * H * C * p \
                          + W * H * C * 2 * b

        for i_bin in range(C):
            values = t * self.value_layer.forward(t-i_bin/(C-1))
            # draw in voxel grid
            idx = idx_before_bins + W * H * i_bin
            vox.put_(idx.long(), values, accumulate=True)

        vox = vox.view(-1, 2, C, H, W)
        vox = torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1)
        # 最终变成 B 2C H W

        if self.use_adapter:
            vox = self.adapter_conv(vox)

        vox = torch.nn.functional.interpolate(vox, size=(256, 256), mode='bilinear', align_corners=False)

        return vox