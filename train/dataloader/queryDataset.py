import os
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from dataUtils import get_data_from_path

def normalize_event_volume(event_volume):
    # max_value = np.max(event_volume)

    # if max_value == 0:
    #     return event_volume
    
    # # 归一化处理
    # normalized_volume = event_volume / max_value
    # return normalized_volume
    return event_volume

class QueryDataset(Dataset):
    def __init__(self, txt_file, query_dir, database_dirs, transform=None, event_vpr=False):
        """
        Args:
            txt_file (str): 包含三元组信息的txt文件路径。
            query_dir (str): 查询样本（query）的文件夹路径。
            database_dirs (list): 数据库样本的文件夹路径列表。
            transform (callable, optional): 图像的预处理变换。
        """
        self.query_dir = query_dir
        self.database_dirs = database_dirs
        self.transform = transform
        self.event_vpr = event_vpr
        
        # 解析txt文件，获取query, positive, negatives的时间戳
        self.triplets = self._parse_triplets(txt_file)
    
    def _parse_triplets(self, txt_file):
        """
        解析txt文件中的三元组信息
        """
        triplets = []
        with open(txt_file, 'r') as file:
            for line in file:
                # 按空格或逗号分割每一行
                timestamps = line.strip().split(',')
                query = timestamps[0].strip()
                positive = timestamps[1].strip()
                negatives = [t.strip() for t in timestamps[2:]]
                triplets.append((query, positive, negatives))
        return triplets

    def _find_file_in_database(self, timestamp, file_type):
        """
        在database_dirs中查找时间戳对应的文件
        Args:
            timestamp (str): 文件的时间戳。
            file_type (str): 文件类型 ("frame" 或 "event")。
        Returns:
            str: 找到的文件路径，如果未找到则返回None。
        """
        for dir in self.database_dirs:
            if file_type == "frame":
                file_path = os.path.join(dir, "frame", f"{timestamp}.png")
            elif file_type == "event":
                file_path = os.path.join(dir, "event", f"{timestamp}.npy")
            elif file_type == "bin":
                file_path = os.path.join(dir, "bin", f"bin_{timestamp}.npy")
            
            if os.path.exists(file_path):
                return file_path  # 返回第一个找到的路径
        return None  # 如果没有找到，返回None

    def _load_frame(self, dir, timestamp):
        """
        加载帧图像
        Args:
            dir (str): 文件所在的目录（query_dir 或 database_dirs中的某一个）。
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
            dir (str): 文件所在的目录（query_dir 或 database_dirs中的某一个）。
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
    
    def _load_event_bin(self, dir, timestamp):
        """
        加载事件体素网格的bin（用于Event-VPR）
        Args:
            dir (str): 文件所在的目录（database_dirs中的某一个）。
            timestamp (str): 事件体数据的时间戳。
            mlp_layer: 外部传入的mlp层，用于做EST处理
        Returns:
            Tensor: 加载的事件体数据。
        """
        event_bin_txt_path = os.path.join(dir, "bin", f"bin_{timestamp}.npy")
        data =  get_data_from_path(event_bin_txt_path)
        # data: [n, 4]
        return data

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        query_timestamp, pos_timestamp, neg_timestamps = self.triplets[idx]

        # 加载query样本（query_dir）
        query_frame = self._load_frame(self.query_dir, query_timestamp)
        if not self.event_vpr:
            query_event_volume = self._load_event_volume(self.query_dir, query_timestamp)
        else:
            query_event_volume = self._load_event_bin(self.query_dir, query_timestamp)

        # 加载正样本，首先在database_dirs中查找正样本
        pos_frame_path = self._find_file_in_database(pos_timestamp, "frame")
        if not self.event_vpr:
            pos_event_path = self._find_file_in_database(pos_timestamp, "event")
        else:
            pos_event_path = self._find_file_in_database(pos_timestamp, "bin")

        if pos_frame_path is None :
            raise FileNotFoundError(f"Positive frame path not found for timestamp {pos_timestamp}")
        
        if pos_event_path is None:
            raise FileNotFoundError(f"Positive event path not found for timestamp {pos_timestamp}")

        pos_frame = Image.open(pos_frame_path).convert('L')
        if self.transform:
            pos_frame = self.transform(pos_frame)

        if not self.event_vpr:
            pos_event_volume = np.load(pos_event_path)
            pos_event_volume = normalize_event_volume(pos_event_volume)
            pos_event_volume = torch.tensor(pos_event_volume).float()
            pos_event_volume = torch.nn.functional.interpolate(pos_event_volume.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
        else:
            data = get_data_from_path(pos_event_path)
            pos_event_volume = data


        # 加载负样本，在database_dirs中查找每一个负样本
        neg_frames = []
        neg_event_volumes = []

        for neg_timestamp in neg_timestamps:
            neg_frame_path = self._find_file_in_database(neg_timestamp, "frame")
            if not self.event_vpr:
                neg_event_path = self._find_file_in_database(neg_timestamp, "event")
            else:
                neg_event_path = self._find_file_in_database(neg_timestamp, "bin")

            if neg_frame_path is None or neg_event_path is None:
                raise FileNotFoundError(f"Negative sample not found for timestamp {neg_timestamp}")

            neg_frame = Image.open(neg_frame_path).convert('L')
            if self.transform:
                neg_frame = self.transform(neg_frame)
            neg_frames.append(neg_frame)
            #print("in load Data neg frame:",neg_frame.size())
    
            if not self.event_vpr:
                neg_event_volume = np.load(neg_event_path)
                neg_event_volume = normalize_event_volume(neg_event_volume)
                neg_event_volume = torch.tensor(neg_event_volume).float()
                neg_event_volume = torch.nn.functional.interpolate(neg_event_volume.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
                neg_event_volumes.append(neg_event_volume)
                #print("in load Data neg event:",neg_event_volume.size())
            else:
                data = get_data_from_path(neg_event_path)
                neg_event_volumes.append(data)  # neg_event_volumes [n,4]
                

        # 将列表转为张量
        # 就返回列表吧
        # neg_frames = torch.stack(neg_frames)  # 形状为 [num, 1, 256, 256]
        # neg_event_volumes = torch.stack(neg_event_volumes)  # 形状为 [num, C, 256, 256] 或 [num, max_N, 4]
        #print("in load Data:",len(neg_frames),len(neg_event_volumes))
        return query_frame, query_event_volume, pos_frame, pos_event_volume, neg_frames, neg_event_volumes
