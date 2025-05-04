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

    first_secs, first_nsecs = event_data[0, 0], event_data[0, 1]
    rel_time = (event_data[:, 0] - first_secs) * 1e6 + (event_data[:, 1] - first_nsecs) / 1e3
    data = np.stack((rel_time, event_data[:, 2], event_data[:, 3], event_data[:, 4]), axis=1).astype(np.float32)    # t, x, y, p
    data = torch.from_numpy(data).float()

    return data