import torch
import numpy as np

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
        components = event_data[0]
        if len(components) == 5:  # 确保行包含 5 个元素 (secs, nsecs, x, y, p)
            secs, nsecs, x, y, p = map(int, components)
            # 计算相对时间戳（纳秒）
            t = (secs - first_secs) * int(1e9) + (nsecs - first_nsecs)
            data.append([t, x, y, p])

    # 转换为 Tensor
    del event_data
    data = torch.tensor(data, dtype=torch.float)

    return data