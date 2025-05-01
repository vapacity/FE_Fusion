import os
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from dataUtils import get_data_from_path, normalize_event_volume
import copy
class QueryDataset(Dataset):
    def __init__(self, txt_file, query_dir, database_dirs, transform=None, graph_transform=None, event_vpr=False, graph_as_frame=False):
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
        self.graph_transform = graph_transform
        self.event_vpr = event_vpr
        self.graph_as_frame = graph_as_frame
        
        # 解析txt文件，获取query, positives, negatives的时间戳
        self.triplets = self._parse_triplets(txt_file)
    
    def _parse_triplets(self, txt_file):
        """
        解析txt文件中的三元组信息
        """
        triplets = []
        with open(txt_file, 'r') as file:
            for line in file:
                # 按分号分割每一行
                parts = line.strip().split(';')
                query = parts[0].strip()
                positives = [p.strip() for p in parts[1].strip().split(',')]
                negatives = [n.strip() for n in parts[2].strip().split(',')]
                # 随机选择10个正样本和100个负样本
                if len(positives) > 10:
                    positives = np.random.choice(positives, 10, replace=False)
                if len(negatives) > 100:
                    negatives = np.random.choice(negatives, 100, replace=False)
                triplets.append((query, positives, negatives))
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
                file_path = os.path.join(dir, "bin", f"{timestamp}.npy")
            elif file_type == "processed":
                file_path = os.path.join(dir, "processed", f"{timestamp}.pt")
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
        event_volume = torch.tensor(event_volume).float()
        event_volume = normalize_event_volume(event_volume)
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
        event_bin_txt_path = os.path.join(dir, "bin", f"{timestamp}.npy")
        data =  get_data_from_path(event_bin_txt_path)
        # data: [n, 4]
        return data
    
    def _load_graph(self, dir, timestamp):
        """
        加载图数据
        """
        graph_path = os.path.join(dir, "processed", f"{timestamp}.pt")
        data = torch.load(graph_path)
        if self.graph_transform:
            data = self.graph_transform(data)
        return data

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        query_timestamp, pos_timestamps, neg_timestamps = self.triplets[idx]

        # 加载query样本（query_dir）
        query_frame = self._load_frame(self.query_dir, query_timestamp)
        if not self.event_vpr:
            query_event_volume = self._load_event_volume(self.query_dir, query_timestamp)
        else:
            query_event_volume = self._load_event_bin(self.query_dir, query_timestamp)
        if self.graph_as_frame:
            query_graph = self._load_graph(self.query_dir, query_timestamp)

        # 加载正样本，首先在database_dirs中查找正样本
        pos_frames_dict = {}
        pos_graphs_dict = {}
        pos_event_volumes_dict = {}

        for pos_timestamp in pos_timestamps:
            if self.graph_as_frame:
                pos_graph_path = self._find_file_in_database(pos_timestamp, "processed")
                if pos_graph_path is None:
                    raise FileNotFoundError(f"Positive graph path not found for timestamp {pos_timestamp}")
                pos_graph = torch.load(pos_graph_path)
                if self.graph_transform:
                    pos_graph = self.graph_transform(pos_graph)
                pos_graphs_dict[pos_timestamp] = pos_graph
            else:
                pos_frame_path = self._find_file_in_database(pos_timestamp, "frame")
                if pos_frame_path is None:
                    raise FileNotFoundError(f"Positive frame path not found for timestamp {pos_timestamp}")
                
                pos_frame = Image.open(pos_frame_path).convert('L')
                if self.transform:
                    pos_frame = self.transform(pos_frame)
                pos_frames_dict[pos_timestamp] = pos_frame

            if not self.event_vpr:
                pos_event_path = self._find_file_in_database(pos_timestamp, "event")
            else:
                pos_event_path = self._find_file_in_database(pos_timestamp, "bin")
            if pos_event_path is None:
                raise FileNotFoundError(f"Positive event path not found for timestamp {pos_timestamp}")

            

            if not self.event_vpr:
                pos_event_volume = np.load(pos_event_path)
                pos_event_volume = torch.tensor(pos_event_volume).float()
                pos_event_volume = normalize_event_volume(pos_event_volume)
                pos_event_volume = torch.nn.functional.interpolate(pos_event_volume.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
                pos_event_volumes_dict[pos_timestamp] = pos_event_volume
            else:
                data = get_data_from_path(pos_event_path)
                pos_event_volumes_dict[pos_timestamp] = data

        # 加载负样本，在database_dirs中查找每一个负样本
        neg_frames_dict = {}
        neg_graphs_dict = {}
        neg_event_volumes_dict = {}

        for neg_timestamp in neg_timestamps:
            neg_graph_path = self._find_file_in_database(neg_timestamp, "processed")
            neg_frame_path = self._find_file_in_database(neg_timestamp, "frame")
            if self.graph_as_frame:
                if neg_graph_path is None:
                    raise FileNotFoundError(f"Negative sample not found for timestamp {neg_timestamp}")
                neg_graph = torch.load(neg_graph_path)
                if self.graph_transform:
                    neg_graph = self.graph_transform(neg_graph)
                neg_graphs_dict[neg_timestamp] = neg_graph
            else:
                if neg_frame_path is None:
                    raise FileNotFoundError(f"Negative sample not found for timestamp {neg_timestamp}")
                neg_frame = Image.open(neg_frame_path).convert('L')
                if self.transform:
                    neg_frame = self.transform(neg_frame)
                neg_frames_dict[neg_timestamp] = neg_frame

            if not self.event_vpr:
                neg_event_path = self._find_file_in_database(neg_timestamp, "event")
            else:
                neg_event_path = self._find_file_in_database(neg_timestamp, "bin")
  
    
            if not self.event_vpr:
                neg_event_volume = np.load(neg_event_path)
                neg_event_volume = torch.tensor(neg_event_volume).float()
                neg_event_volume = normalize_event_volume(neg_event_volume)
                neg_event_volume = torch.nn.functional.interpolate(neg_event_volume.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
                neg_event_volumes_dict[neg_timestamp] = neg_event_volume
            else:
                data = get_data_from_path(neg_event_path)
                neg_event_volumes_dict[neg_timestamp] = data  # neg_event_volumes [n,4]
        
        if self.graph_as_frame:
            selected_pos_timestamps = list(pos_graphs_dict.keys())  # 字符串
            selected_neg_timestamps = list(neg_graphs_dict.keys())
            pos_graphs = [pos_graphs_dict[timestamp] for timestamp in selected_pos_timestamps]
            pos_event_volumes = [pos_event_volumes_dict[timestamp] for timestamp in selected_pos_timestamps]
            neg_graphs = [neg_graphs_dict[timestamp] for timestamp in selected_neg_timestamps]
            neg_event_volumes = [neg_event_volumes_dict[timestamp] for timestamp in selected_neg_timestamps]
        else:
            selected_pos_timestamps = list(pos_frames_dict.keys())  # 字符串
            selected_neg_timestamps = list(neg_frames_dict.keys())
            pos_frames = [pos_frames_dict[timestamp] for timestamp in selected_pos_timestamps]
            pos_event_volumes = [pos_event_volumes_dict[timestamp] for timestamp in selected_pos_timestamps]
            neg_frames = [neg_frames_dict[timestamp] for timestamp in selected_neg_timestamps]
            neg_event_volumes = [neg_event_volumes_dict[timestamp] for timestamp in selected_neg_timestamps]


        # 如果 < 10, pad 0
        if (self.graph_as_frame and len(pos_graphs) < 10) or (not self.graph_as_frame and len(pos_frames) < 10):
            if self.graph_as_frame:
                padding_size = 10 - len(pos_graphs)
                pos_graphs_padded = pos_graphs + [copy.deepcopy(pos_graphs[0]) for _ in range(padding_size)]
                pos_graphs = pos_graphs_padded
            else:
                padding_size = 10 - len(pos_frames)
                pos_frames_padded = pos_frames + [torch.zeros_like(pos_frames[0]) for _ in range(padding_size)]
                pos_frames = pos_frames_padded
            pos_event_volumes_padded = pos_event_volumes + [torch.zeros_like(pos_event_volumes[0]) for _ in range(padding_size)]
            selected_pos_timestamps_padded = selected_pos_timestamps + [f"p{pad_index}" for pad_index in range(padding_size)]
            pos_event_volumes, selected_pos_timestamps = pos_event_volumes_padded, selected_pos_timestamps_padded
        if (self.graph_as_frame and len(neg_graphs) < 100) or (not self.graph_as_frame and len(neg_frames) < 100):
            if self.graph_as_frame:
                padding_size = 100 - len(neg_graphs)
                neg_graphs_padded = neg_graphs + [copy.deepcopy(neg_graphs[0]) for _ in range(padding_size)]
                neg_graphs = neg_graphs_padded
            else:
                padding_size = 100 - len(neg_frames)
                neg_frames_padded = neg_frames + [torch.zeros_like(neg_frames[0]) for _ in range(padding_size)]
                neg_frames = neg_frames_padded
            neg_event_volumes_padded = neg_event_volumes + [torch.zeros_like(neg_event_volumes[0]) for _ in range(padding_size)]
            selected_neg_timestamps_padded = selected_neg_timestamps + [f"p{pad_index}" for pad_index in range(padding_size)]
            neg_event_volumes, selected_neg_timestamps = neg_event_volumes_padded, selected_neg_timestamps_padded

        if self.graph_as_frame:
            return query_graph, query_event_volume, pos_graphs, pos_event_volumes, neg_graphs, neg_event_volumes, selected_pos_timestamps, selected_neg_timestamps
        else:
            return query_frame, query_event_volume, pos_frames, pos_event_volumes, neg_frames, neg_event_volumes, selected_pos_timestamps, selected_neg_timestamps
