import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from functools import partial
import wandb

from dataloader.queryDataset import QueryDataset
from dataloader.databaseDataset import DatabaseDataset
from dataloader.loaderUtils import *
import argparse
from datetime import datetime
from loss import MultiNegativeTripletLoss
from models.DIFT_Net import DiftNet
from test_Recall_utils import recall_at_n_with_distance
from torch.nn import CosineSimilarity
import random
import numpy as np
from models import FE_Main_Net


def generate_paths(exp_item, processed_data_path, experiment_item):
    query_path = processed_data_path+experiment_item[exp_item]['query']
    database_paths = [f"{processed_data_path}{db}" for db in experiment_item[exp_item]['database']]
    triplet_path = processed_data_path+'triplets/'+exp_item+"_multi/triplet_result.txt"
    return query_path, database_paths, triplet_path

def generate_test_paths(exp_item, processed_data_path, experiment_item):
    query_path = processed_data_path+experiment_item[exp_item]['query']
    database_paths = [f"{processed_data_path}{db}" for db in experiment_item[exp_item]['database']]
    query_gps_path = processed_data_path+experiment_item[exp_item]['query']+'/interpolated_gps.txt'
    database_gps_paths = [processed_data_path+db+'/interpolated_gps.txt' for db in experiment_item[exp_item]['database']]
    return query_path, database_paths, query_gps_path, database_gps_paths

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Train FE-Net with different configurations.")
    parser.add_argument('--use_frame', action='store_true', help="Use only frames (default: False)")
    parser.add_argument('--use_event', action='store_true', help="Use only events (default: False)")
    parser.add_argument('--event_vpr', action='store_true', help="Reproduce Event VPR")
    parser.add_argument('--graph_as_frame', action='store_true', help="Use graph as frame")
    parser.add_argument('--experiment_name', type=str, default="experiment_3", help="Experiment Name")
    parser.add_argument('--num_epochs', type=int, default=200, help="num_epochs")
    parser.add_argument('--batch_size', type=int, default=4, help="batch_size")
    parser.add_argument('--test_batch_factor', type=int, default=4, help="test_batch_factor")
    parser.add_argument('--load_model', type=str, default=None, help="load_model")
    parser.add_argument('--disable_wandb', action='store_true', help="Disable wandb logging")
    parser.add_argument('--num_workers', type=int, default=16, help="num_workers")

    return parser.parse_args()


# 创建数据集和 DataLoader
if __name__ == "__main__":
    args = parse_args()
    seed = 42
    set_seed(seed)
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if args.use_frame and args.use_event:
        current_time = current_time + "_FEFusion"
    elif args.use_frame:
        current_time = current_time + "_Frame"
    elif args.use_event:
        current_time = current_time + "_Event"


    # Initialize wandb
    if not args.disable_wandb:
        wandb.init(
            project=f"fe-fusion-{args.experiment_name}",
            name=f"fe-fusion-graph-as-frame-{current_time}",
        config={
            "use_frame": args.use_frame,
            "use_event": args.use_event,
            "event_vpr": args.event_vpr,
            "graph_as_frame": args.graph_as_frame,
            "experiment_name": args.experiment_name,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size
        }
    )

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    processed_data_path = '/root/autodl-tmp/processed_data/'
    save_path = '/root/autodl-tmp/FE_Fusion/runs/'

    experiment_item = {
        'experiment_1': {
            'query': 'sr',
            'database': ['dt','mn']
        },
        'experiment_2': {
            'query': 'mn',
            'database': ['ss2','dt']
        },
        'experiment_3': {
            'query': 'sr',
            'database': ['ss2','dt']
        },
        'experiment_4': {
            'query': 'sr',
            'database': ['ss2','mn']
        }
    }

    test_experiment_item = {
        'experiment_1': {
            'query': 'ss2',
            'database': ['ss1']
        },
        'experiment_2': {
            'query': 'sr',
            'database': ['ss1']
        },
        'experiment_3': {
            'query': 'mn',
            'database': ['ss1']
        },
        'experiment_4': {
            'query': 'dt',
            'database': ['ss1']
        }
    }

    # 获取路径
    query_dir, database_dirs, triplet_file = generate_paths(args.experiment_name, processed_data_path, experiment_item)
    test_query_dir, test_database_dirs, test_query_gps_path, test_database_gps_paths = generate_test_paths(args.experiment_name, processed_data_path, test_experiment_item)
    test_train_query_dir, test_train_database_dirs, test_train_query_gps_path, test_train_database_gps_paths = generate_test_paths(args.experiment_name, processed_data_path, experiment_item)

    save_dir = save_path + f"result_{current_time}/saved_model"
    BATCH_SIZE = args.batch_size
    num_epochs = args.num_epochs
    channel_sizes = [128, 256, 512]
    global_num_negatives = 12

    dataset = QueryDataset(triplet_file, query_dir, database_dirs, transform, event_vpr=args.event_vpr, graph_as_frame=args.graph_as_frame)
    databaseDataset = DatabaseDataset(database_dirs=database_dirs,transform=transform, event_vpr=args.event_vpr, graph_as_frame=args.graph_as_frame)    # train 时用不到gps信息，用gps的话可能会少一些数据
    test_query_dataset = DatabaseDataset(database_dirs=[test_query_dir],transform=transform, event_vpr=args.event_vpr, graph_as_frame=args.graph_as_frame, use_timestamps_from_gps=True)
    test_database_dataset = DatabaseDataset(database_dirs=test_database_dirs,transform=transform, event_vpr=args.event_vpr, graph_as_frame=args.graph_as_frame, use_timestamps_from_gps=True)
    test_train_query_dataset =  DatabaseDataset(database_dirs=[query_dir],transform=transform, event_vpr=args.event_vpr, graph_as_frame=args.graph_as_frame, use_timestamps_from_gps=True)
    test_train_database_dataset = DatabaseDataset(database_dirs=database_dirs,transform=transform, event_vpr=args.event_vpr, graph_as_frame=args.graph_as_frame, use_timestamps_from_gps=True)

    if not args.event_vpr:
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_workers, collate_fn=partial(collate_query_normal, graph_as_frame=args.graph_as_frame))
        database_loader = DataLoader(databaseDataset, batch_size=BATCH_SIZE*args.test_batch_factor, shuffle=False, num_workers=args.num_workers, collate_fn=partial(collate_database_normal, graph_as_frame=args.graph_as_frame))
        test_query_loader = DataLoader(test_query_dataset, batch_size=BATCH_SIZE*args.test_batch_factor, shuffle=False, num_workers=args.num_workers, collate_fn=partial(collate_database_normal, graph_as_frame=args.graph_as_frame))
        test_database_loader = DataLoader(test_database_dataset, batch_size=BATCH_SIZE*args.test_batch_factor, shuffle=False, num_workers=args.num_workers, collate_fn=partial(collate_database_normal, graph_as_frame=args.graph_as_frame))
        test_train_query_loader = DataLoader(test_train_query_dataset, batch_size=BATCH_SIZE*args.test_batch_factor, shuffle=False, num_workers=args.num_workers, collate_fn=partial(collate_database_normal, graph_as_frame=args.graph_as_frame))
        test_train_database_loader = DataLoader(test_train_database_dataset, batch_size=BATCH_SIZE*args.test_batch_factor, shuffle=False, num_workers=args.num_workers, collate_fn=partial(collate_database_normal, graph_as_frame=args.graph_as_frame))
    else:
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_workers, collate_fn=partial(collate_query_vpr, graph_as_frame=args.graph_as_frame))
        database_loader = DataLoader(databaseDataset, batch_size=BATCH_SIZE*args.test_batch_factor, shuffle=False, num_workers=args.num_workers, collate_fn=partial(collate_database_vpr, graph_as_frame=args.graph_as_frame))
        test_query_loader = DataLoader(test_query_dataset, batch_size=BATCH_SIZE*args.test_batch_factor, shuffle=False, num_workers=args.num_workers, collate_fn=partial(collate_database_vpr, graph_as_frame=args.graph_as_frame))
        test_database_loader = DataLoader(test_database_dataset, batch_size=BATCH_SIZE*args.test_batch_factor, shuffle=False, num_workers=args.num_workers, collate_fn=partial(collate_database_vpr, graph_as_frame=args.graph_as_frame))
        test_train_query_loader = DataLoader(test_train_query_dataset, batch_size=BATCH_SIZE*args.test_batch_factor, shuffle=False, num_workers=args.num_workers, collate_fn=partial(collate_database_vpr, graph_as_frame=args.graph_as_frame))
        test_train_database_loader = DataLoader(test_train_database_dataset, batch_size=BATCH_SIZE*args.test_batch_factor, shuffle=False, num_workers=args.num_workers, collate_fn=partial(collate_database_vpr, graph_as_frame=args.graph_as_frame))

    if args.graph_as_frame:
        model = FE_Main_Net.MainNet(channel_sizes, use_frame=True, use_event=False, event_vpr=False, graph_as_frame=True).cuda()  # Only frames
    elif args.use_frame:
        model = FE_Main_Net.MainNet(channel_sizes, use_event=False, event_vpr=False).cuda()  # Only frames
    elif args.use_event:
        model = FE_Main_Net.MainNet(channel_sizes, use_frame=False, event_vpr=args.event_vpr).cuda()  # Only events
    else:
        model = FE_Main_Net.MainNet(channel_sizes).cuda()  # Both frames and events
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))

    ######################################### train parameters #########################################
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    margin = 0.1
    criterion = MultiNegativeTripletLoss(margin=margin).cuda()  # 使用自定义的多负样本三元组损失函数

    ######################################### test prepare #########################################
    test_database_gps_data = {}
    test_query_gps_data = {}
    test_train_database_gps_data = {}
    test_train_query_gps_data = {}
    for gps_file in test_database_gps_paths:
        with open(gps_file, 'r') as f:
            for line in f:
                lat, lon, timestamp = line.strip().split()
                test_database_gps_data[timestamp] = (float(lat), float(lon), float(timestamp))
    with open(test_query_gps_path, 'r') as f:
        for line in f:
            lat, lon, timestamp = line.strip().split()
            test_query_gps_data[timestamp] = (float(lat), float(lon), float(timestamp))

    for gps_file in test_train_database_gps_paths:
        with open(gps_file, 'r') as f:
            for line in f:
                lat, lon, timestamp = line.strip().split()
                test_train_database_gps_data[timestamp] = (float(lat), float(lon), float(timestamp))
    with open(test_train_query_gps_path, 'r') as f:
        for line in f:
            lat, lon, timestamp = line.strip().split()
            test_train_query_gps_data[timestamp] = (float(lat), float(lon), float(timestamp))  

    ######################################### main loop #########################################
    for epoch in range(num_epochs):
        model_path = os.path.join(save_dir, f'model_{"eventVPR" if args.event_vpr else "FEFusion"}_epoch_{epoch+1}.pth')



        ######################################### test #########################################
        model.eval()
        database_features, timestamps_list = update_test_features(model, database_loader)
        database_features_dict = {timestamp: feature for timestamp, feature in zip(timestamps_list, database_features)}
        if epoch >= 1:
            test_database_features, test_timestamps_list = update_test_features(model, test_database_loader)
            test_train_database_features, test_train_timestamps_list = update_test_features(model, test_train_database_loader)

            # Calculate recalls
            recall_1 = recall_at_n_with_distance(test_query_loader, test_database_features, test_query_gps_data, test_database_gps_data, model, test_timestamps_list, N=1, distance_threshold=75)
            recall_5 = recall_at_n_with_distance(test_query_loader, test_database_features, test_query_gps_data, test_database_gps_data, model, test_timestamps_list, N=5, distance_threshold=75)
            train_recall_1 = recall_at_n_with_distance(test_train_query_loader, test_train_database_features, test_train_query_gps_data, test_train_database_gps_data, model, test_train_timestamps_list, N=1, distance_threshold=75)
            train_recall_5 = recall_at_n_with_distance(test_train_query_loader, test_train_database_features, test_train_query_gps_data, test_train_database_gps_data, model, test_train_timestamps_list, N=5, distance_threshold=75)

            # Log test metrics to wandb
            if not args.disable_wandb:
                wandb.log({
                    "test_recall@1": recall_1,
                    "test_recall@5": recall_5,
                    "train_recall@1": train_recall_1,
                    "train_recall@5": train_recall_5
                })

        ######################################### train #########################################
        epoch_loss = 0
        model.train()  # 设置模型为训练模式
        
        with tqdm(total=len(dataloader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for train_batch_idx, batch in enumerate(dataloader):
                # 清零优化器梯度
                optimizer.zero_grad()

                # 将数据移动到 GPU
                query_frame_single, query_event_volume_single, pos_frames_multi, pos_event_volumes_multi, neg_frames_multi ,neg_event_volumes_multi, pos_timestamps, neg_timestamps = batch
                # pos_frames_and_event_volumes_dict 和 neg_frames_and_event_volumes_dict 都是 list of dicts
                query_frame_single, query_event_volume_single = query_frame_single.cuda(), query_event_volume_single.cuda()
                # 前向传播计算 query 和 pos 的特征表示
                anchor_output = model(query_frame_single, query_event_volume_single)  # 锚点特征


                ############################################# 选取正样本和负样本 #############################################
                query_feature = anchor_output.detach().cpu()
                all_similarity_scores = torch.norm(query_feature.unsqueeze(1) - database_features.unsqueeze(0), p=2, dim=2)    # (b, 1, i)与(1, l, i)

                pos_frame_batch = []
                pos_event_volume_batch = []
                neg_frames_batch = [[] for _ in range(global_num_negatives)]
                neg_event_volumes_batch = [[] for _ in range(global_num_negatives)]
                min_num_negatives = global_num_negatives

                # batch_size = query_event_volume_single.size(0)
                for batch_idx in range(query_event_volume_single.size(0)):
                    all_similarity_scores_item = all_similarity_scores[batch_idx]   # length
                    all_similarity_scores_item_dict = {timestamp: score for timestamp, score in zip(timestamps_list, all_similarity_scores_item)}   # 存储对于某个query, 所有database中timestamp对应的score
                    pos_timestamps_item = [pos_timestamp_line[batch_idx] for pos_timestamp_line in pos_timestamps]
                    neg_timestamps_item = [neg_timestamp_line[batch_idx] for neg_timestamp_line in neg_timestamps]

                    pos_timestamps_item = [pos_timestamps_item[i] for i in range(len(pos_timestamps_item)) if not pos_timestamps_item[i].startswith("p")]
                    neg_timestamps_item = [neg_timestamps_item[i] for i in range(len(neg_timestamps_item)) if not neg_timestamps_item[i].startswith("p")]

                    if args.graph_as_frame:
                        pos_frames_and_event_volumes_dict = {pos_timestamps_item[i]: (pos_frames_multi[batch_idx][i], pos_event_volumes_multi[i][batch_idx]) for i in range(len(pos_timestamps_item))}
                        neg_frames_and_event_volumes_dict = {neg_timestamps_item[i]: (neg_frames_multi[batch_idx][i], neg_event_volumes_multi[i][batch_idx]) for i in range(len(neg_timestamps_item))}
                    else:
                        pos_frames_and_event_volumes_dict = {pos_timestamps_item[i]: (pos_frames_multi[i][batch_idx], pos_event_volumes_multi[i][batch_idx]) for i in range(len(pos_timestamps_item))}
                        neg_frames_and_event_volumes_dict = {neg_timestamps_item[i]: (neg_frames_multi[i][batch_idx], neg_event_volumes_multi[i][batch_idx]) for i in range(len(neg_timestamps_item))}
                    
                    pos_item_scores = torch.stack([all_similarity_scores_item_dict[timestamp] for timestamp in pos_timestamps_item])    # [10]
                    neg_item_scores = torch.stack([all_similarity_scores_item_dict[timestamp] for timestamp in neg_timestamps_item])    # [100]

                    # 从 pos_timestamps_item 中找到最接近的index
                    _, pos_item_scores_indices = torch.topk(pos_item_scores, 1, dim=0, largest=False, sorted=True)
                    best_pos_timestamp = pos_timestamps_item[pos_item_scores_indices[0]]
                    # 30%概率，positive 是随机找的，这个似乎可以提点效果
                    if np.random.random() < 0.3:
                        best_pos_timestamp = np.random.choice(pos_timestamps_item)
                    pos_score = all_similarity_scores_item_dict[best_pos_timestamp]
                    
                    min_num_negatives = min(min_num_negatives, len(neg_item_scores))
                    hard_negative_scores, hard_negative_scores_indices = torch.topk(neg_item_scores, min_num_negatives, dim=0, largest=False, sorted=True)
                    hard_negative_timestamps = [neg_timestamps_item[i] for i in hard_negative_scores_indices]
                    # 在hard_negative_timestamps之外，额外在score小于positive + margin的negative里随机选replace_num个，替换掉topk里面的最后replace_num个
                    replace_num = 4
                    valid_neg_indices = []
                    for i, neg_timestamp in enumerate(neg_timestamps_item):
                        if neg_timestamp not in hard_negative_timestamps:
                            neg_score = all_similarity_scores_item_dict[neg_timestamp]
                            if neg_score < pos_score + margin:
                                valid_neg_indices.append(i)
                    if len(valid_neg_indices) > replace_num:
                        random_indices_new = np.random.choice(valid_neg_indices, replace_num, replace=False)
                    else:
                        random_indices_new = valid_neg_indices
                    if len(random_indices_new) > 0:
                        random_timestamps_new = [neg_timestamps_item[i] for i in random_indices_new]
                        # Replace last replace_num hard negatives with random ones
                        hard_negative_timestamps = hard_negative_timestamps[:-len(random_timestamps_new)] + random_timestamps_new


                    pos_frame, pos_event_volume = pos_frames_and_event_volumes_dict[best_pos_timestamp]
                    pos_frame_batch.append(pos_frame)
                    pos_event_volume_batch.append(pos_event_volume)

                    for i, hard_negative_timestamp in enumerate(hard_negative_timestamps):
                        neg_frames_batch[i].append(neg_frames_and_event_volumes_dict[hard_negative_timestamp][0])
                        neg_event_volumes_batch[i].append(neg_frames_and_event_volumes_dict[hard_negative_timestamp][1])

                if args.graph_as_frame:
                    pos_frame = Batch.from_data_list(pos_frame_batch)
                else:
                    pos_frame = torch.stack(pos_frame_batch)    # B C H W   # TODO: graph 要改成Batch.from_data_list(batch_data_list)
                pos_event_volume = torch.stack(pos_event_volume_batch)
                if args.graph_as_frame:
                    neg_frames = [Batch.from_data_list(neg_frames_batch[i]) for i in range(min_num_negatives)]
                else:
                    neg_frames = [torch.stack(neg_frames_batch[i]) for i in range(min_num_negatives)]
                neg_event_volumes = [torch.stack(neg_event_volumes_batch[i]) for i in range(min_num_negatives)]

                ############################################# 选取正样本和负样本结束 #############################################
                pos_frame, pos_event_volume = pos_frame.cuda(), pos_event_volume.cuda()
                pos_output = model(pos_frame, pos_event_volume)  # 正样本特征


                #串行版本计算
                all_negative_outputs = []
                for i in range(min_num_negatives):
                    # 提取第 i 个负样本
                    neg_frame = neg_frames[i]  # 形状为 [batch_size, 1, 256, 256]
                    neg_event_volume = neg_event_volumes[i]  # 不use vpr 形状为 [batch_size, 2, 256, 256] use vpr 形状为 [batch_size, max_len, 4]
                    neg_frame, neg_event_volume = neg_frame.cuda(), neg_event_volume.cuda()
                    neg_output = model(neg_frame, neg_event_volume)  # 形状为 [batch_size, feature_dim]

                    # 将该负样本的特征添加到列表
                    all_negative_outputs.append(neg_output)     # torch.Size([8, 16384])
                # 将所有负样本特征拼接成 [batch_size, num_negatives, feature_dim]
                negative_outputs = torch.stack(all_negative_outputs, dim=1)  # [batch_size, global_num_negatives, feature_dim]

                # 并行计算负样本特征
                # neg_frames = torch.cat(neg_frames, dim=0).cuda()  # [batch_size * global_num_negatives, 1, 256, 256]
                # neg_event_volumes = torch.cat(neg_event_volumes, dim=0).cuda()  # [batch_size * global_num_negatives, 2, 256, 256]
                # neg_outputs = model(neg_frames, neg_event_volumes)  # [batch_size * global_num_negatives, feature_dim]
                # negative_outputs = neg_outputs.view(query_frame_single.size(0), global_num_negatives, -1)

                # 计算三元组损失
                batch_loss = criterion(anchor_output, pos_output, negative_outputs) # anchor_output有梯度，query_feature没有梯度
                epoch_loss += batch_loss.item()
                batch_loss.backward()
                optimizer.step()

                # Log batch loss to wandb
                if not args.disable_wandb:
                    wandb.log({"batch_loss": batch_loss.item()})
                pbar.update(1)

        # Log epoch metrics
        average_loss = epoch_loss / len(dataloader)
        if not args.disable_wandb:
            wandb.log({"train_loss": average_loss})

        if epoch == 0:
            os.makedirs(save_dir, exist_ok=True)
        # Save model
        torch.save(model.state_dict(), model_path)

