import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from functools import partial


from dataloader.queryDataset import QueryDataset
from dataloader.databaseDataset import DatabaseDataset
from dataloader.loaderUtils import *
import argparse
from datetime import datetime
from loss import MultiNegativeTripletLoss
from net import Net
from models.DIFT_Net import DiftNet
from test_Recall_utils import recall_at_n_with_distance
from torch.nn import CosineSimilarity
    
def parse_args():
    parser = argparse.ArgumentParser(description="Train FE-Net with different configurations.")
    parser.add_argument('--use_frame', action='store_true', help="Use only frames (default: False)")
    parser.add_argument('--use_event', action='store_true', help="Use only events (default: False)")
    parser.add_argument('--event_vpr', action='store_true', help="Reproduce Event VPR")
    parser.add_argument('--use_dift', action='store_true', help="Use DIFT")
    parser.add_argument('--experiment_name', type=str, default="experiment_1", help="Experiment Name")
    parser.add_argument('--num_epochs', type=int, default=200, help="num_epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="batch_size")

    return parser.parse_args()


# 创建数据集和 DataLoader
if __name__ == "__main__":
    args = parse_args()
    # Default to using both frames and events if no specific argument is provided
    if args.use_frame and args.use_event:
        raise ValueError("Cannot use both 'use_frame' and 'use_event' at the same time.")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    processed_data_path = '/root/autodl-tmp/processed_data/'
    save_path = '/root/autodl-tmp/FE_Fusion/runs/'

    def generate_paths(exp_item):
        query_path = processed_data_path+experiment_item[exp_item]['query']
        database_paths = [f"{processed_data_path}{db}" for db in experiment_item[exp_item]['database']]
        triplet_path = processed_data_path+'triplets/'+exp_item+"_multi/triplet_result.txt"
        return query_path, database_paths, triplet_path

    def generate_test_paths(exp_item):
        query_path = processed_data_path+experiment_item[exp_item]['query']
        database_paths = [f"{processed_data_path}{db}" for db in experiment_item[exp_item]['database']]
        # 对 test来说 只取query即可，因此和train相同也无妨
        query_gps_path = processed_data_path+experiment_item[exp_item]['query']+'/interpolated_gps.txt'
        database_gps_paths = [processed_data_path+db+'/interpolated_gps.txt' for db in experiment_item[exp_item]['database']]
        return query_path, database_paths, query_gps_path, database_gps_paths

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
    query_dir, database_dirs, triplet_file = generate_paths(args.experiment_name)
    test_query_dir, test_database_dirs, test_query_gps_path, test_database_gps_paths = generate_test_paths(args.experiment_name)

    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    save_dir = save_path + f"result_{current_time}/saved_model"
    loss_file = save_path + f"result_{current_time}/loss.txt"
    os.makedirs(os.path.dirname(loss_file), exist_ok=True)
    os.makedirs(save_dir,exist_ok=True)

    BATCH_SIZE = args.batch_size
    num_epochs = args.num_epochs
    channel_sizes = [128, 256, 512]
    global_num_negatives = 12

    dataset = QueryDataset(triplet_file, query_dir, database_dirs, transform, event_vpr=args.event_vpr, use_dift=args.use_dift)
    databaseDataset = DatabaseDataset(database_dirs=database_dirs,transform=transform, event_vpr=args.event_vpr, use_dift=args.use_dift)    # train 时用不到gps信息，用gps的话可能会少一些数据
    test_query_dataset = DatabaseDataset(database_dirs=[test_query_dir],transform=transform, event_vpr=args.event_vpr, use_dift=args.use_dift, use_timestamps_from_gps=True)
    test_database_dataset = DatabaseDataset(database_dirs=test_database_dirs,transform=transform, event_vpr=args.event_vpr, use_dift=args.use_dift, use_timestamps_from_gps=True)

    if not args.event_vpr:
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=24)
        database_loader = DataLoader(databaseDataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)
        test_query_loader = DataLoader(test_query_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)
        test_database_loader = DataLoader(test_database_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)
    else:
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=24, collate_fn=partial(collate_query_vpr, num_negatives=global_num_negatives))
        database_loader = DataLoader(databaseDataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, collate_fn=collate_database_vpr)
        test_query_loader = DataLoader(test_query_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, collate_fn=collate_database_vpr_test)
        test_database_loader = DataLoader(test_database_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, collate_fn=collate_database_vpr_test)


    if args.use_dift:
        model = DiftNet(channel_sizes).cuda()
    else:
        model = Net(args.use_event, args.use_frame, args.event_vpr, channel_sizes=channel_sizes).cuda()

    ######################################### train parameters #########################################
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    margin = 0.1
    criterion = MultiNegativeTripletLoss(margin=margin).cuda()  # 使用自定义的多负样本三元组损失函数
    loss_history = []
    accuracy_history = []

    ######################################### test prepare #########################################
    test_database_gps_data = {}
    test_query_gps_data = {}
    for gps_file in test_database_gps_paths:
        with open(gps_file, 'r') as f:
            for line in f:
                lat, lon, timestamp = line.strip().split()
                test_database_gps_data[timestamp] = (float(lat), float(lon), float(timestamp))
    with open(test_query_gps_path, 'r') as f:
        for line in f:
            lat, lon, timestamp = line.strip().split()
            test_query_gps_data[timestamp] = (float(lat), float(lon), float(timestamp))

    ######################################### main loop #########################################
    if args.use_dift:
        for epoch in range(num_epochs):
            pass
    else:
        for epoch in range(num_epochs):
            model_path = os.path.join(save_dir, f'model_{"eventVPR" if args.event_vpr else "FEFusion"}_epoch_{epoch+1}.pth')
            ######################################### train #########################################
            epoch_loss = 0
            model.eval()
            database_features, timestamps_list = update_test_features(model, database_loader)
            database_features_dict = {timestamp: feature for timestamp, feature in zip(timestamps_list, database_features)}
            model.train()  # 设置模型为训练模式
            
            with tqdm(total=len(dataloader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
                for batch in dataloader:
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
                    # cos = CosineSimilarity(dim=2, eps=1e-8)
                    # all_similarity_scores = cos(query_feature.unsqueeze(1), database_features.unsqueeze(0))    #[B, length]
                    # all_similarity_scores = 1 - torch.einsum('bi,li->bl', [query_feature, database_features])    # 如果改回L2
                    all_similarity_scores = torch.norm(query_feature.unsqueeze(1) - database_features.unsqueeze(0), p=2, dim=2)    # (b, 1, i)与(1, l, i)

                    pos_frame_batch = []
                    pos_event_volume_batch = []
                    neg_frames_batch = [[] for _ in range(global_num_negatives)]
                    neg_event_volumes_batch = [[] for _ in range(global_num_negatives)]
                    min_num_negatives = global_num_negatives

                    for batch_idx in range(query_frame_single.size(0)):
                        all_similarity_scores_item = all_similarity_scores[batch_idx]   # length
                        all_similarity_scores_item_dict = {timestamp: score for timestamp, score in zip(timestamps_list, all_similarity_scores_item)}   # 存储对于某个query, 所有database中timestamp对应的score
                        pos_timestamps_item = [pos_timestamp_line[batch_idx] for pos_timestamp_line in pos_timestamps]
                        neg_timestamps_item = [neg_timestamp_line[batch_idx] for neg_timestamp_line in neg_timestamps]

                        pos_timestamps_item = [pos_timestamps_item[i] for i in range(len(pos_timestamps_item)) if not pos_timestamps_item[i].startswith("p")]
                        neg_timestamps_item = [neg_timestamps_item[i] for i in range(len(neg_timestamps_item)) if not neg_timestamps_item[i].startswith("p")]

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

                    pos_frame = torch.stack(pos_frame_batch)    # B C H W
                    pos_event_volume = torch.stack(pos_event_volume_batch)
                    neg_frames = [torch.stack(neg_frames_batch[i]) for i in range(min_num_negatives)]
                    neg_event_volumes = [torch.stack(neg_event_volumes_batch[i]) for i in range(min_num_negatives)]

                    ############################################# 选取正样本和负样本结束 #############################################
                    pos_frame, pos_event_volume = pos_frame.cuda(), pos_event_volume.cuda()
                    pos_output = model(pos_frame, pos_event_volume)  # 正样本特征

                    # 初始化用于存储所有负样本特征的列表
                    all_negative_outputs = []

                    # 逐个计算每个负样本的特征
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

                    # 计算三元组损失
                    with torch.autograd.detect_anomaly():
                        batch_loss = criterion(anchor_output, pos_output, negative_outputs) # anchor_output有梯度，query_feature没有梯度
                        epoch_loss += batch_loss.item()
                        batch_loss.backward()
                    optimizer.step()

                    # 更新进度条上的损失信息
                    pbar.set_postfix(loss=f"{batch_loss.item():.4f}")
                    pbar.update(1)

            # 打印当前 epoch 的平均损失
            average_loss = epoch_loss / len(dataloader) 
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss}')

            ######################################### log and save #########################################
            # 写入损失
            with open(loss_file, 'a') as f:
                f.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss}\n')
            torch.save(model.state_dict(), model_path)


            ######################################### test #########################################
            model.eval()
            output_file = save_dir.replace("saved_model", "test_recall_results.txt")
            with open(output_file, 'a') as f:
                database_features, timestamps_list = update_test_features(model, test_database_loader)
                # 计算 Recall@1 和 Recall@5
                print("calculate recall")
                recall_1 = recall_at_n_with_distance(test_query_loader, database_features, test_query_gps_data, test_database_gps_data, model, timestamps_list, N=1, distance_threshold=75, use_dift=args.use_dift)
                recall_5 = recall_at_n_with_distance(test_query_loader, database_features, test_query_gps_data, test_database_gps_data, model, timestamps_list, N=5, distance_threshold=75, use_dift=args.use_dift)
                
                # 保存结果
                f.write(f"Epoch {epoch}, Recall@1: {recall_1:.4f}, Recall@5: {recall_5:.4f}\n")
                f.flush()
                print(f"Epoch {epoch} 结果已保存: Recall@1: {recall_1:.4f}, Recall@5: {recall_5:.4f}")