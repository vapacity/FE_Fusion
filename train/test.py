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
import time
import torch_geometric.transforms as T

def wait_for_model(model_path, timeout=1000):
    start_time = time.time()  # 记录开始时间
    while time.time() - start_time < timeout:
        if os.path.exists(model_path):
            time.sleep(100)     # 防止正在保存
            return True  # 如果模型路径存在，返回 True
        time.sleep(10)  # 每10秒检查一次模型路径
    return False  # 超过20分钟还没找到，返回 False

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

    parser.add_argument('--model_dir', type=str, default=None, help="model_dir")
    parser.add_argument('--start_epoch', type=int, default=1, help="start_epoch")
    parser.add_argument('--end_epoch', type=int, default=200, help="end_epoch")

    return parser.parse_args()


# 创建数据集和 DataLoader
if __name__ == "__main__":
    args = parse_args()


    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    graph_transform_test =  T.Compose([T.Cartesian(cat=False), T.RandomScale([0.99999, 1])])

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


    BATCH_SIZE = args.batch_size
    num_epochs = args.num_epochs
    channel_sizes = [128, 256, 512]
    global_num_negatives = 12

    test_query_dataset = DatabaseDataset(database_dirs=[test_query_dir],transform=transform, graph_transform=graph_transform_test, event_vpr=args.event_vpr, graph_as_frame=args.graph_as_frame, use_timestamps_from_gps=True)
    test_database_dataset = DatabaseDataset(database_dirs=test_database_dirs,transform=transform, graph_transform=graph_transform_test, event_vpr=args.event_vpr, graph_as_frame=args.graph_as_frame, use_timestamps_from_gps=True)
    test_train_query_dataset =  DatabaseDataset(database_dirs=[query_dir],transform=transform, graph_transform=graph_transform_test, event_vpr=args.event_vpr, graph_as_frame=args.graph_as_frame, use_timestamps_from_gps=True)
    test_train_database_dataset = DatabaseDataset(database_dirs=database_dirs,transform=transform, graph_transform=graph_transform_test, event_vpr=args.event_vpr, graph_as_frame=args.graph_as_frame, use_timestamps_from_gps=True)

    if not args.event_vpr:
        test_query_loader = DataLoader(test_query_dataset, batch_size=BATCH_SIZE*args.test_batch_factor, shuffle=False, num_workers=args.num_workers, collate_fn=partial(collate_database_normal, graph_as_frame=args.graph_as_frame))
        test_database_loader = DataLoader(test_database_dataset, batch_size=BATCH_SIZE*args.test_batch_factor, shuffle=False, num_workers=args.num_workers, collate_fn=partial(collate_database_normal, graph_as_frame=args.graph_as_frame))
        test_train_query_loader = DataLoader(test_train_query_dataset, batch_size=BATCH_SIZE*args.test_batch_factor, shuffle=False, num_workers=args.num_workers, collate_fn=partial(collate_database_normal, graph_as_frame=args.graph_as_frame))
        test_train_database_loader = DataLoader(test_train_database_dataset, batch_size=BATCH_SIZE*args.test_batch_factor, shuffle=False, num_workers=args.num_workers, collate_fn=partial(collate_database_normal, graph_as_frame=args.graph_as_frame))
    else:
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
    for epoch in range(args.start_epoch, args.end_epoch, 1):
        model_dir = os.path.join(args.model_dir, "saved_model")
        model_path = os.path.join(model_dir, f'model_{"eventVPR" if args.event_vpr else "FEFusion"}_epoch_{epoch}.pth')
        # if not wait_for_model(model_path):
        #     continue
        model.load_state_dict(torch.load(model_path))
        metrics_dir = os.path.join(args.model_dir, "metrics")
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)
        ######################################### test #########################################
        model.eval()

        test_database_features, test_timestamps_list = update_test_features(model, test_database_loader)
        test_train_database_features, test_train_timestamps_list = update_test_features(model, test_train_database_loader)

        # Calculate recalls
        recall_1 = recall_at_n_with_distance(test_query_loader, test_database_features, test_query_gps_data, test_database_gps_data, model, test_timestamps_list, N=1, distance_threshold=75)
        recall_5 = recall_at_n_with_distance(test_query_loader, test_database_features, test_query_gps_data, test_database_gps_data, model, test_timestamps_list, N=5, distance_threshold=75)
        train_recall_1 = recall_at_n_with_distance(test_train_query_loader, test_train_database_features, test_train_query_gps_data, test_train_database_gps_data, model, test_train_timestamps_list, N=1, distance_threshold=75)
        train_recall_5 = recall_at_n_with_distance(test_train_query_loader, test_train_database_features, test_train_query_gps_data, test_train_database_gps_data, model, test_train_timestamps_list, N=5, distance_threshold=75)

        with open(os.path.join(metrics_dir, "result.txt"), "a") as f:
            f.write(f"recall_1: {recall_1}, recall_5: {recall_5}, train_recall_1: {train_recall_1}, train_recall_5: {train_recall_5}\n")
