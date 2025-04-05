import torch
import numpy as np

def normalize_event_volume(tensor):
    """
    对每个通道分别归一化，使每个通道最大值为255
    输入:
        tensor: torch.Tensor, shape [C, H, W]
    返回:
        tensor: 归一化后的tensor, dtype仍为 float32
    """
    for c in range(tensor.shape[0]):
        channel = tensor[c]
        current_max = channel.max()
        if current_max > 0:
            scale_factor = 1 / current_max
            tensor[c] = channel * scale_factor
        else:
            tensor[c] = torch.zeros_like(channel)  # 防止除以0后是NaN
    return tensor

def get_data_from_path(event_path):
    data = []   # [secs, nsecs, x, y, p]
    event_data = np.load(event_path)
    if not event_data.any():
        print("no event data exist")
        return torch.zeros((1, 4), dtype=torch.float)

    # 解析第一行，获取基准时间
    first_components = event_data[0]
    first_secs, first_nsecs = int(first_components[0]), int(first_components[1])


    for event_line in event_data:
        if len(event_line) == 5:  # 确保行包含 5 个元素 (secs, nsecs, x, y, p)
            secs, nsecs, x, y, p = map(int, event_line)
            # 计算相对时间戳（纳秒）

            t = (secs - first_secs) * int(1e6) + (nsecs - first_nsecs) / int(1e3)   # 防止上溢
            data.append([t, x, y, p])
            # print(f"event:{[t, x, y, p]}")


    # 转换为 Tensor
    del event_data
    data = torch.tensor(data, dtype=torch.float)

    return data