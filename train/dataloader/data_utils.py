import torch

def get_data_from_path(event_path, max_len):
    data = []
    with open(event_path, 'r') as f:
        lines = f.readlines()

        if not lines:
            return torch.zeros((max_len, 4), dtype=torch.long)
            raise ValueError(f"File {event_path} is empty.")

        # 解析第一行，获取基准时间
        first_components = lines[0].strip().split()
        first_secs, first_nsecs = int(first_components[0]), int(first_components[1])

        for line in lines:
            components = line.strip().split()
            if len(components) == 5:  # 确保行包含 5 个元素 (secs, nsecs, x, y, p)
                secs, nsecs, x, y, p = map(int, components)
                # 计算相对时间戳（纳秒）
                t = (secs - first_secs) * int(1e9) + (nsecs - first_nsecs)
                data.append([t, x, y, p])

    # 转换为 Tensor
    data = torch.tensor(data, dtype=torch.long)
    # breakpoint()

    # 计算需要填充的行数
    pad_len = max_len - data.shape[0]
    if pad_len > 0:
        padding = torch.zeros((pad_len, 4), dtype=torch.long)
        data = torch.cat([data, padding], dim=0)

    return data