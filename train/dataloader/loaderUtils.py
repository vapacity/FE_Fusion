import torch
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from torch.utils.data.dataloader import default_collate


def collate_database_vpr(data):
    # data[0] Batchsize of frames data[1] B N 4
    frames = []
    events = []
    for i, (frame, event) in enumerate(data):
        frames.append(frame)
        batch_index = torch.full((event.shape[0], 1), i, device=event.device)  # 创建 batch 维度
        events.append(torch.cat([event, batch_index], dim=1))  # 拼接 batch 维, N,4 和 N,1 拼接成 N,5
    # events = [
    # [N_1, 5],  # 第 1 个样本的事件
    # [N_2, 5],  # 第 2 个样本的事件
    # ...
    # [N_B, 5]]   # 第 B 个样本的事件
    frames = default_collate(frames)
    events = torch.cat(events, dim=0)
    return frames, events


def collate_query_vpr(data, batch_size):
    query_frame_lst = []
    query_event_volume_lst = []
    pos_frame_lst = []
    pos_event_volume_lst = []
    neg_frames_lst_multi = [[] for _ in range(batch_size)]     # 要变成 N B shape
    neg_event_volumes_lst_multi = [[] for _ in range(batch_size)]
    for i, (query_frame, query_event_volume, pos_frame, pos_event_volume, neg_frames, neg_event_volumes) in enumerate(data):
        query_frame_lst.append(query_frame)
        batch_index_query = torch.full((query_event_volume.shape[0], 1), i, device=query_event_volume.device)  # 创建 batch 维度
        query_event_volume_lst.append(torch.cat([query_event_volume, batch_index_query], dim=1))  # 拼接 batch 维, N,4 和 N,1 拼接成 N,5
        pos_frame_lst.append(pos_frame)
        batch_index_pos = torch.full((pos_event_volume.shape[0], 1), i, device=pos_event_volume.device)  # 创建 batch 维度
        pos_event_volume_lst.append(torch.cat([pos_event_volume, batch_index_pos], dim=1))  # 拼接 batch 维, N,4 和 N,1 拼接成 N,5

        for j, neg_frame in enumerate(neg_frames):
            neg_frames_lst_multi[j].append(neg_frame)   # + num 1 256 256
            neg_event_volume = neg_event_volumes[j]
            batch_index_neg = torch.full((neg_event_volume.shape[0], 1), i, device=neg_event_volume.device)  # 创建 batch 维度
            neg_event_volumes_lst_multi[j].append(torch.cat([neg_event_volume, batch_index_neg], dim=1))  # 拼接 batch 维, N,4 和 N,1 拼接成 N,5


    query_frame_lst = default_collate(query_frame_lst)
    query_event_volume_lst = torch.cat(query_event_volume_lst, dim=0)
    pos_frame_lst = default_collate(pos_frame_lst)
    pos_event_volume_lst = torch.cat(pos_event_volume_lst, dim=0)
    neg_frames_lst = [default_collate(neg_frames_per_instance) for neg_frames_per_instance in neg_frames_lst_multi]    # 递归处理得到tensor
    neg_event_volumes_lst = [default_collate(neg_event_volumes_per_instance) for neg_event_volumes_per_instance in neg_event_volumes_lst_multi]
    return query_frame_lst, query_event_volume_lst, pos_frame_lst, pos_event_volume_lst, neg_frames_lst, neg_event_volumes_lst