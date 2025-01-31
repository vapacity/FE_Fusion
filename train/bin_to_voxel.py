import torch

def EST_voxel_grid(raw_tensor, mlp_layer, H=260, W=346, C=9):
    """
    参数：
    - event_tensor: tensor [N, 4] (t, x, y, p) 在 CUDA 上
    - mlp_layer: nn.Module, 计算体素值的 MLP，必须在相同的 CUDA 设备上
    - H: int, 事件图像高度
    - W: int, 事件图像宽度
    - C: int, 时间通道数（默认 9）

    返回：
    - torch.Tensor: 体素网格, 形状 [2C, H, W]
    """

    # 确保所有计算都在 CUDA 上
    device = raw_tensor.device
    mlp_layer = mlp_layer.to(device)

    mask = raw_tensor.any(dim=1)
    event_tensor = raw_tensor[mask]
    # 初始化体素网格
    num_voxels = int(2 * H * W * C)
    vox = torch.zeros((num_voxels),device=event_tensor.device)

    if event_tensor.numel() == 0:
        print("event_tensor is empty!")
    else:
        # event_tensor torch.Size([18377, 4])

        t_min = event_tensor[:, 0].min()  # 其实是0
        t_max = event_tensor[:, 0].max()        
        # 计算坐标
        t = ((event_tensor[:, 0] - t_min) / (t_max - t_min)).float()
        x = event_tensor[:, 1].long().clamp(0, W - 1)
        y = event_tensor[:, 2].long().clamp(0, H - 1)
        p = event_tensor[:, 3].long()  # polarity

        # breakpoint()  # 看下p
        idx_before_bins = x \
                        + W * y \
                        + 0 \
                        + W * H * C * p \


        for i_bin in range(C):
            values = t * (mlp_layer((t-i_bin/(C-1)).unsqueeze(1))).squeeze(1)   # 大小也是N
            idx = idx_before_bins + W * H * i_bin   # 其实就是插入 p c_i h w 处
            vox.put_(idx.long(), values, accumulate=True)

    vox = vox.view(2, C, H, W)
    # 调整形状并插值到 256×256
    voxel_grid = vox.reshape(-1, H, W)  # [2C, H, W]

    voxel_grid = torch.nn.functional.interpolate(voxel_grid.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)

    return voxel_grid  # shape: [2C, 256, 256]，在 CUDA 上