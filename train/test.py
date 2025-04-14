import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from functools import partial

from models import FE_Net, EST_Net
from models.DIFT_Net import DiftNet
from dataloader.queryDataset import QueryDataset
from dataloader.databaseDataset import DatabaseDataset
from dataloader.loaderUtils import collate_database_vpr_test
import argparse
from datetime import datetime
from test_Recall_utils import recall_at_n_with_distance
from net import Net
import yaml
import time

def wait_for_model(model_path, timeout=1000):
    start_time = time.time()  # 记录开始时间
    while time.time() - start_time < timeout:
        if os.path.exists(model_path):
            time.sleep(100)     # 防止正在保存
            return True  # 如果模型路径存在，返回 True
        time.sleep(10)  # 每10秒检查一次模型路径
    return False  # 超过20分钟还没找到，返回 False

def update_test_features_dift(model, database_loader):
    database_features = []
    timestamps_list = []
    with torch.no_grad():
        for db_batch in tqdm(database_loader, desc="Processing database batches"):
            dift_feat_0, dift_feat_1, dift_feat_2, timestamp = db_batch
            dift_feat_0, dift_feat_1, dift_feat_2 = dift_feat_0.cuda(), dift_feat_1.cuda(), dift_feat_2.cuda()

            # 提取特征
            db_features = model(dift_feat_0, dift_feat_1, dift_feat_2)

            # 将特征保存到列表中
            database_features.append(db_features.cpu())
            timestamps_list.extend(timestamp)
            
    return torch.cat(database_features, dim=0), timestamps_list

def update_test_features(model, database_loader):
    database_features = []
    timestamps_list = []
    with torch.no_grad():
        for db_batch in tqdm(database_loader, desc="Processing database batches"):
            db_frames, db_event_volumes, timestamp = db_batch
            db_frames, db_event_volumes = db_frames.cuda(), db_event_volumes.cuda()

            # 提取特征
            db_features = model(db_frames, db_event_volumes)

            # 将特征保存到列表中
            database_features.append(db_features.cpu())
            timestamps_list.extend(timestamp)

    # 使用 torch.cat 将不同批次的数据拼接起来，而不是 stack
    database_features = torch.cat(database_features, dim=0)
    return database_features, timestamps_list

    
def generate_test_paths(exp_item):
    query_path = processed_data_path+experiment_item[exp_item]['query']
    database_paths = [f"{processed_data_path}{db}" for db in experiment_item[exp_item]['database']]
    # 对 test来说 只取query即可，因此和train相同也无妨
    triplet_path = processed_data_path+'triplets/'+exp_item+"/triplet_result_test.txt"
    query_gps_path = processed_data_path+experiment_item[exp_item]['query']+'/interpolated_gps.txt'
    database_gps_paths = [processed_data_path+db+'/interpolated_gps.txt' for db in experiment_item[exp_item]['database']]
    return query_path, database_paths, triplet_path, query_gps_path, database_gps_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Train FE-Net with different configurations.")
    parser.add_argument('--use_frame', action='store_true', help="Use only frames (default: False)")
    parser.add_argument('--use_event', action='store_true', help="Use only events (default: False)")
    parser.add_argument('--event_vpr', action='store_true', help="Reproduce Event VPR")
    parser.add_argument('--use_dift', action='store_true', help="Use DIFT")
    parser.add_argument('--model_dir', type=str, help="Model Dir")
    parser.add_argument('--start_epoch', type=int, default=1, help="start epoch, default 50")
    parser.add_argument('--end_epoch', type=int, default=150, help="end epoch, default 150")
    parser.add_argument('--experiment_name', type=str, default="experiment_1", help="Experiment Name")


    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    use_frame = args.use_frame
    use_event = args.use_event
    event_vpr = args.event_vpr
    use_dift = args.use_dift
    experiment_name = args.experiment_name

    
    if use_frame and use_event:
        raise ValueError("Cannot use both 'use_frame' and 'use_event' at the same time.")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    processed_data_path = '/root/autodl-tmp/processed_data/'

    # experiment_item = {
    #     'experiment_1': {
    #         'query': 'sr',
    #         'database': ['dt','mn']
    #     },
    #     'experiment_2': {
    #         'query': 'mn',
    #         'database': ['ss2','dt']
    #     },
    #     'experiment_3': {
    #         'query': 'sr',
    #         'database': ['ss2','dt']
    #     },
    #     'experiment_4': {
    #         'query': 'sr',
    #         'database': ['ss2','mn']
    #     }
    # }

    experiment_item = {
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

    
    test_query_dir, test_database_dirs, test_triplet_file, test_query_gps_path, test_database_gpt_paths = generate_test_paths(experiment_name)
    BATCH_SIZE = 8

    test_query_dataset = DatabaseDataset(database_dirs=[test_query_dir],transform=transform, event_vpr=event_vpr, use_dift=use_dift, use_timestamps_from_gps=True)
    test_database_dataset = DatabaseDataset(database_dirs=test_database_dirs,transform=transform, event_vpr=event_vpr, use_dift=use_dift, use_timestamps_from_gps=True)

    if not event_vpr:
        test_query_loader = DataLoader(test_query_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)
        test_database_loader = DataLoader(test_database_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)
    else:
        test_query_loader = DataLoader(test_query_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, collate_fn=collate_database_vpr_test)
        test_database_loader = DataLoader(test_database_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, collate_fn=collate_database_vpr_test)

    test_database_gps_data = {}
    test_query_gps_data = {}

    for gps_file in test_database_gpt_paths:
        with open(gps_file, 'r') as f:
            for line in f:
                lat, lon, timestamp = line.strip().split()
                test_database_gps_data[timestamp] = (float(lat), float(lon), float(timestamp))
    with open(test_query_gps_path, 'r') as f:
        for line in f:
            lat, lon, timestamp = line.strip().split()
            test_query_gps_data[timestamp] = (float(lat), float(lon), float(timestamp))


    channel_sizes = [128, 256, 512]
    if use_dift:
        model = DiftNet(channel_sizes).cuda()
    else:
        model = Net(use_event, use_frame, event_vpr, channel_sizes=channel_sizes).cuda()

    model_dir = args.model_dir
    output_file = model_dir.replace("saved_model", "test_recall_results.txt")
    with open(output_file, 'a') as f:
        for epoch in range(args.start_epoch, args.end_epoch, 1):
            model_path = os.path.join(model_dir, f'model_{"eventVPR" if event_vpr else "FEFusion"}_epoch_{epoch}.pth')
            if not os.path.exists(model_path):
                print(f"模型 {model_path} 不存在，等待...")
                if not wait_for_model(model_path, 2000):
                    print(f"模型 {model_path} 在指定时间内没有产生，跳过")
                    continue
            
            # 加载模型
            model.load_state_dict(torch.load(model_path))
            model.eval()

            print("update features")
            if use_dift:
                database_features, timestamps_list = update_test_features_dift(model, test_database_loader)
            else:
                database_features, timestamps_list = update_test_features(model, test_database_loader)
            # 计算 Recall@1 和 Recall@5
            
            print("calculate recall")
            recall_1 = recall_at_n_with_distance(test_query_loader, database_features, test_query_gps_data, test_database_gps_data, model, timestamps_list, N=1, distance_threshold=75, use_dift=use_dift)
            recall_5 = recall_at_n_with_distance(test_query_loader, database_features, test_query_gps_data, test_database_gps_data, model, timestamps_list, N=5, distance_threshold=75, use_dift=use_dift)
            
            # 保存结果
            f.write(f"Epoch {epoch}, Recall@1: {recall_1:.4f}, Recall@5: {recall_5:.4f}\n")
            f.flush()
            print(f"Epoch {epoch} 结果已保存: Recall@1: {recall_1:.4f}, Recall@5: {recall_5:.4f}")




