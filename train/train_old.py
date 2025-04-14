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
        triplet_path = processed_data_path+'triplets/'+exp_item+"/triplet_result.txt"
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

    dataset = QueryDataset(triplet_file, query_dir, database_dirs, transform, event_vpr=args.event_vpr, use_dift=args.use_dift)
    databaseDataset = DatabaseDataset(database_dirs=database_dirs,transform=transform, event_vpr=args.event_vpr, use_dift=args.use_dift)
    test_query_dataset = DatabaseDataset(database_dirs=[test_query_dir],transform=transform, event_vpr=args.event_vpr, use_dift=args.use_dift, use_timestamps_from_gps=True)
    test_database_dataset = DatabaseDataset(database_dirs=test_database_dirs,transform=transform, event_vpr=args.event_vpr, use_dift=args.use_dift, use_timestamps_from_gps=True)

    if not args.event_vpr:
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=24)
        database_loader = DataLoader(databaseDataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)
        test_query_loader = DataLoader(test_query_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)
        test_database_loader = DataLoader(test_database_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)
    else:
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=24, collate_fn=partial(collate_query_vpr, num_negatives=10))
        database_loader = DataLoader(databaseDataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, collate_fn=collate_database_vpr)
        test_query_loader = DataLoader(test_query_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, collate_fn=collate_database_vpr_test)
        test_database_loader = DataLoader(test_database_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, collate_fn=collate_database_vpr_test)


    if args.use_dift:
        model = DiftNet(channel_sizes).cuda()
    else:
        model = Net(args.use_event, args.use_frame, args.event_vpr, channel_sizes=channel_sizes).cuda()

    ######################################### train parameters #########################################
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = MultiNegativeTripletLoss(margin=0.1).cuda()  # 使用自定义的多负样本三元组损失函数
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
            model.train()  # 设置模型为训练模式
            
            with tqdm(total=len(dataloader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
                for batch in dataloader:
                    # 将数据移动到 GPU
                    query_frame_single, query_event_volume_single, pos_frame_single, pos_event_volume_single, neg_frames_multi ,neg_event_volumes_multi = batch # 带multi的第0维多一个维度
                    query_frame_single, query_event_volume_single, pos_frame_single, pos_event_volume_single = query_frame_single.cuda(), query_event_volume_single.cuda(), pos_frame_single.cuda(), pos_event_volume_single.cuda()

                    # print("neg frames:",neg_frames.size())
                    # print("neg event:",neg_event_volumes.size())
                    # 清零优化器梯度
                    optimizer.zero_grad()

                    # 前向传播计算 query 和 pos 的特征表示
                    anchor_output = model(query_frame_single, query_event_volume_single)  # 锚点特征
                    pos_output = model(pos_frame_single, pos_event_volume_single)  # 正样本特征
                    num_negatives = len(neg_frames_multi)  # 获取负样本数量，假设为 10

                    # 初始化用于存储所有负样本特征的列表
                    all_negative_outputs = []

                    # 逐个计算每个负样本的特征
                    for i in range(num_negatives):
                        # 提取第 i 个负样本
                        neg_frame = neg_frames_multi[i]  # 形状为 [batch_size, 1, 256, 256]
                        neg_event_volume = neg_event_volumes_multi[i]  # 不use vpr 形状为 [batch_size, 2, 256, 256] use vpr 形状为 [batch_size, max_len, 4]
                        neg_frame, neg_event_volume = neg_frame.cuda(), neg_event_volume.cuda()

                        # 计算第 i 个负样本的特征
                        neg_output = model(neg_frame, neg_event_volume)  # 形状为 [batch_size, feature_dim]

                        # 将该负样本的特征添加到列表
                        all_negative_outputs.append(neg_output)     # torch.Size([8, 16384])

                    # 将所有负样本特征拼接成 [batch_size, num_negatives, feature_dim]
                    negative_outputs = torch.stack(all_negative_outputs, dim=1)  # [batch_size, 10, feature_dim]

                    # 计算三元组损失
                    with torch.autograd.detect_anomaly():
                        batch_loss = criterion(anchor_output, pos_output, negative_outputs)
                        epoch_loss += batch_loss.item()
                        # breakpoint()
                        # 反向传播并优化
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