import torch

def EST_voxel_grid(event_tensor, mlp_layer, H=260, W=346, C=9):
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
    device = event_tensor.device
    mlp_layer = mlp_layer.to(device)

    # 计算最小时间和最大时间 (避免 Python for 循环)
    t_min = event_tensor[:, 0].min()
    t_max = event_tensor[:, 0].max()

    # 计算时间 bin 大小
    time_bin_size = (t_max - t_min) / C

    # 初始化体素网格
    voxel_grid = torch.zeros((2, C, H, W), dtype=torch.float32, device=device)

    # 计算时间 bin 索引
    temporal_bins = ((event_tensor[:, 0] - t_min) / time_bin_size).long().clamp(0, C - 1)  # 限制在 [0, C-1]

    # 计算整数坐标 (确保范围合法)
    x = event_tensor[:, 1].long().clamp(0, W - 1)
    y = event_tensor[:, 2].long().clamp(0, H - 1)
    p = event_tensor[:, 3].long()  # polarity

    # 计算 MLP 生成的体素值
    t_tensor = event_tensor[:, 0].view(-1, 1)  # [N, 1]
    voxel_values = mlp_layer(t_tensor).squeeze(-1)  # [N]

    # 根据 polarity 更新不同通道
    voxel_grid.index_add_(0, p, torch.zeros_like(voxel_grid))  # 预防 index_add_ 操作时报错
    voxel_grid.index_add_(1, temporal_bins, torch.zeros_like(voxel_grid))  # 预防 index_add_ 操作时报错

    voxel_grid[p, temporal_bins, y, x] += voxel_values  # 直接更新体素网格

    # 调整形状并插值到 256×256
    voxel_grid = voxel_grid.reshape(-1, H, W)  # [2C, H, W]
    voxel_grid = torch.nn.functional.interpolate(voxel_grid.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)

    return voxel_grid  # shape: [2C, 256, 256]，仍在 CUDA 上