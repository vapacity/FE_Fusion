import os
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
def normalize_event_volume(event_volume):
    # # 找到数组中的最大值
    # max_value = np.max(event_volume)
    
    # # 避免除以零的情况
    # if max_value == 0:
    #     return event_volume  # 如果最大值是0，则直接返回原数组
    
    # # 归一化处理
    # normalized_volume = event_volume / max_value
    # return normalized_volume
    return event_volume

class DatabaseDataset(Dataset):
    def __init__(self, database_dirs, transform=None):
        """
        Args:
            database_dirs (list): 数据库样本的文件夹路径列表。
            transform (callable, optional): 图像的预处理变换。
        """
        self.database_dirs = database_dirs
        self.transform = transform

        # 获取所有数据库文件中的时间戳
        self.database_timestamps = self._get_database_timestamps()

    def _get_database_timestamps(self):
        """
        获取所有数据库中的时间戳。
        """
        timestamps = set()
        for dir in self.database_dirs:
            frame_dir = os.path.join(dir, "frame")
            event_dir = os.path.join(dir, "event")

            # 获取帧文件夹中的时间戳
            for frame_file in os.listdir(frame_dir):
                if frame_file.endswith(".png"):
                    timestamps.add(frame_file.replace(".png", ""))

            # 获取事件体文件夹中的时间戳
            for event_file in os.listdir(event_dir):
                if event_file.endswith(".npy"):
                    timestamps.add(event_file.replace(".npy", ""))

        return sorted(list(timestamps))

    def _load_frame(self, dir, timestamp):
        """
        加载帧图像
        Args:
            dir (str): 文件所在的目录（database_dirs中的某一个）。
            timestamp (str): 帧图像的时间戳。
        Returns:
            Image: 加载的图像。
        """
        frame_path = os.path.join(dir, "frame", f"{timestamp}.png")
        frame = Image.open(frame_path).convert('L')  # 转换为灰度图
        if self.transform:
            frame = self.transform(frame)
        return frame

    def _load_event_volume(self, dir, timestamp):
        """
        加载事件体数据
        Args:
            dir (str): 文件所在的目录（database_dirs中的某一个）。
            timestamp (str): 事件体数据的时间戳。
        Returns:
            Tensor: 加载的事件体数据。
        """
        event_path = os.path.join(dir, "event", f"{timestamp}.npy")
        event_volume = np.load(event_path)
        event_volume = normalize_event_volume(event_volume)
        event_volume = torch.tensor(event_volume).float()
        event_volume = torch.nn.functional.interpolate(event_volume.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
        return event_volume

    def __len__(self):
        return len(self.database_timestamps)

    def __getitem__(self, idx):
        timestamp = self.database_timestamps[idx]

        # 在database_dirs中查找对应时间戳的样本
        for dir in self.database_dirs:
            frame_path = os.path.join(dir, "frame", f"{timestamp}.png")
            event_path = os.path.join(dir, "event", f"{timestamp}.npy")

            if os.path.exists(frame_path) and os.path.exists(event_path):
                frame = self._load_frame(dir, timestamp)
                event_volume = self._load_event_volume(dir, timestamp)
                return frame, event_volume

        raise FileNotFoundError(f"No data found for timestamp {timestamp}")