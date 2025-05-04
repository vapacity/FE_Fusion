import torch
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from torch.utils.data.dataloader import default_collate

# def collate_database_vpr_test(data):
#     # data[0] Batchsize of frames data[1] B N 4
#     frames = []
#     events = []
#     timestamps = []
#     for i, (frame, event, timestamp) in enumerate(data):
#         frames.append(frame)
#         batch_index = torch.full((event.shape[0], 1), i, device=event.device)  # 创建 batch 维度
#         events.append(torch.cat([event, batch_index], dim=1))  # 拼接 batch 维, N,4 和 N,1 拼接成 N,5
#         timestamps.append(timestamp)
#     # events = [
#     # [N_1, 5],  # 第 1 个样本的事件
#     # [N_2, 5],  # 第 2 个样本的事件
#     # ...
#     # [N_B, 5]]   # 第 B 个样本的事件
#     frames = default_collate(frames)
#     events = torch.cat(events, dim=0)
#     timestamps = default_collate(timestamps)
    
#     return frames, events, timestamps

# def collate_database_vpr(data):
#     # data[0] Batchsize of frames data[1] B N 4
#     frames = []
#     events = []
#     for i, (frame, event) in enumerate(data):
#         frames.append(frame)
#         batch_index = torch.full((event.shape[0], 1), i, device=event.device)  # 创建 batch 维度
#         events.append(torch.cat([event, batch_index], dim=1))  # 拼接 batch 维, N,4 和 N,1 拼接成 N,5
#     # events = [
#     # [N_1, 5],  # 第 1 个样本的事件
#     # [N_2, 5],  # 第 2 个样本的事件
#     # ...
#     # [N_B, 5]]   # 第 B 个样本的事件
#     frames = default_collate(frames)
#     events = torch.cat(events, dim=0)
#     return frames, events


# def collate_query_vpr(data, num_negatives):
#     query_frame_lst = []
#     query_event_volume_lst = []
#     pos_frame_lst = []
#     pos_event_volume_lst = []
#     neg_frames_lst_multi = [[] for _ in range(num_negatives)]     # 要变成 N B shape
#     neg_event_volumes_lst_multi = [[] for _ in range(num_negatives)]
#     for i, (query_frame, query_event_volume, pos_frame, pos_event_volume, neg_frames, neg_event_volumes) in enumerate(data):
#         query_frame_lst.append(query_frame)
#         batch_index_query = torch.full((query_event_volume.shape[0], 1), i, device=query_event_volume.device)  # 创建 batch 维度
#         query_event_volume_lst.append(torch.cat([query_event_volume, batch_index_query], dim=1))  # 拼接 batch 维, N,4 和 N,1 拼接成 N,5
#         pos_frame_lst.append(pos_frame)
#         batch_index_pos = torch.full((pos_event_volume.shape[0], 1), i, device=pos_event_volume.device)  # 创建 batch 维度
#         pos_event_volume_lst.append(torch.cat([pos_event_volume, batch_index_pos], dim=1))  # 拼接 batch 维, N,4 和 N,1 拼接成 N,5

#         for j, neg_frame in enumerate(neg_frames):
#             neg_frames_lst_multi[j].append(neg_frame)   # + num 1 256 256
#             neg_event_volume = neg_event_volumes[j]
#             batch_index_neg = torch.full((neg_event_volume.shape[0], 1), i, device=neg_event_volume.device)  # 创建 batch 维度
#             neg_event_volumes_lst_multi[j].append(torch.cat([neg_event_volume, batch_index_neg], dim=1))  # 拼接 batch 维, N,4 和 N,1 拼接成 N,5


#     query_frame_lst = default_collate(query_frame_lst)
#     query_event_volume_lst = torch.cat(query_event_volume_lst, dim=0)
#     pos_frame_lst = default_collate(pos_frame_lst)
#     pos_event_volume_lst = torch.cat(pos_event_volume_lst, dim=0)
#     neg_frames_lst = [default_collate(neg_frames_per_instance) for neg_frames_per_instance in neg_frames_lst_multi]    # 递归处理得到tensor
#     neg_event_volumes_lst = [torch.cat(neg_event_volumes_per_instance, dim=0) for neg_event_volumes_per_instance in neg_event_volumes_lst_multi]
#     return query_frame_lst, query_event_volume_lst, pos_frame_lst, pos_event_volume_lst, neg_frames_lst, neg_event_volumes_lst


import torch
import numpy as np
import os
import sys
from tqdm import tqdm
from torch_geometric.data import Data, Batch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from torch.utils.data.dataloader import default_collate

def collate_vpr_data(data):
    # data[0] Batchsize of frames data[1] B N 4
    events = []
    for i, event in enumerate(data):
        batch_index = torch.full((event.shape[0], 1), i, device=event.device)  # 创建 batch 维度
        events.append(torch.cat([event, batch_index], dim=1))  # 拼接 batch 维, N,4 和 N,1 拼接成 N,5
    # events = [
    # [N_1, 5],  # 第 1 个样本的事件
    # [N_2, 5],  # 第 2 个样本的事件
    # ...
    # [N_B, 5]]   # 第 B 个样本的事件
    events = torch.cat(events, dim=0) # 拼成一个大的
    return events

def collate_database(data, graph_as_frame, event_vpr_as_frame):
    frames = []
    events = []
    timestamps = []
    for i, (frame, event, timestamp) in enumerate(data):
        frames.append(frame)
        events.append(event)
        timestamps.append(timestamp)

    if graph_as_frame:
        frames = Batch.from_data_list(frames)
    elif event_vpr_as_frame:
        frames = collate_vpr_data(frames)
    else:
        frames = default_collate(frames)
    events = default_collate(events)
    timestamps = default_collate(timestamps)
    return frames, events, timestamps

def collate_query(data, graph_as_frame, event_vpr_as_frame):
    query_frame_lst = []
    query_event_volume_lst = []
    pos_frames_lst = []
    pos_event_volumes_lst = []
    neg_frames_lst = []
    neg_event_volumes_lst = []
    selected_pos_timestamps_lst = []
    selected_neg_timestamps_lst = []
    for i, (query_frame, query_event_volume, pos_frames, pos_event_volumes, neg_frames, neg_event_volumes, selected_pos_timestamps, selected_neg_timestamps) in enumerate(data):
        query_frame_lst.append(query_frame)
        query_event_volume_lst.append(query_event_volume)
        pos_frames_lst.append(pos_frames)
        pos_event_volumes_lst.append(pos_event_volumes)
        neg_frames_lst.append(neg_frames)
        neg_event_volumes_lst.append(neg_event_volumes)
        selected_pos_timestamps_lst.append(selected_pos_timestamps)
        selected_neg_timestamps_lst.append(selected_neg_timestamps)

    if graph_as_frame:
        query_frame_lst = Batch.from_data_list(query_frame_lst)
        pos_frames_lst = pos_frames_lst
        neg_frames_lst = neg_frames_lst
    elif event_vpr_as_frame:
        query_frame_lst = collate_vpr_data(query_frame_lst)   # 都是列表
        pos_frames_lst = pos_frames_lst
        neg_frames_lst = neg_frames_lst
    else:
        query_frame_lst = default_collate(query_frame_lst)
        pos_frames_lst = default_collate(pos_frames_lst)
        neg_frames_lst = default_collate(neg_frames_lst)
    
    query_event_volume_lst = default_collate(query_event_volume_lst)
    pos_event_volumes_lst = default_collate(pos_event_volumes_lst)
    neg_event_volumes_lst = default_collate(neg_event_volumes_lst)
    selected_pos_timestamps_lst = default_collate(selected_pos_timestamps_lst)
    selected_neg_timestamps_lst = default_collate(selected_neg_timestamps_lst)
    return query_frame_lst, query_event_volume_lst, pos_frames_lst, pos_event_volumes_lst, neg_frames_lst, neg_event_volumes_lst, selected_pos_timestamps_lst, selected_neg_timestamps_lst


def update_test_features(model, database_loader):
    database_features = []
    timestamps_list = []
    with torch.no_grad():
        model.eval()
        for db_batch in tqdm(database_loader, desc="Processing database batches"):
            db_frames, db_event_volumes, timestamp = db_batch
            db_frames, db_event_volumes = db_frames.cuda(), db_event_volumes.cuda()

            # 提取特征
            db_features = model(db_frames, db_event_volumes)

            # 将特征保存到列表中
            database_features.append(db_features.detach().cpu())
            timestamps_list.extend(timestamp)

    # 使用 torch.cat 将不同批次的数据拼接起来，而不是 stack
    database_features = torch.cat(database_features, dim=0)
    return database_features, timestamps_list

