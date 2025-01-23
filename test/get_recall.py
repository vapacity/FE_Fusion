import os
import torch
from torch.utils.data import DataLoader
from train.dataloader.databaseLoader import DatabaseDataset
from torch.nn import CosineSimilarity
from geopy.distance import geodesic
from tqdm import tqdm
import torchvision.transforms as transforms
import train.models.FE_Net as FE_Net
from torch.utils.data import ConcatDataset

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 生成特征的函数
def generate_features(data_loader, model):
    features = []
    gps_data_list = []
    with torch.no_grad():
        for frames, event_volumes, gps_info in tqdm(data_loader, desc="Generating features"):
            frames, event_volumes = frames.cuda(), event_volumes.cuda()
            feature_batch = model(frames, event_volumes)
            features.append(feature_batch)
            gps_data_list.extend(gps_info)
    features_tensor = torch.cat(features, dim=0)
    return features_tensor

# 定义 Recall 计算函数
def recall_at_n_with_distance(query_loader, database_features, query_gps_data, database_gps_data, model, N=5, distance_threshold=75):
    total_queries = 0
    correct_count = 0
    global_idx = 0
    database_features = torch.tensor(database_features).cuda()

    with torch.no_grad():
        for frames, event_volumes, _ in query_loader:
            frames, event_volumes = frames.cuda(), event_volumes.cuda()
            query_features = model(frames, event_volumes)
            cos = CosineSimilarity(dim=2, eps=1e-8)
            similarity_scores = cos(query_features.unsqueeze(1), database_features.unsqueeze(0))

            top_n_scores, top_n_indices = torch.topk(similarity_scores, N, dim=1, largest=True, sorted=True)
            
           
            for i in range(query_features.size(0)):
                query_lat, query_lon,query_time = query_gps_data[global_idx]
                for idx in top_n_indices[i]:
                    db_lat, db_lon, db_time = database_gps_data[idx.item()]
                    distance = geodesic((query_lat, query_lon), (db_lat, db_lon)).meters
                    if distance < distance_threshold:
                        # print("query timestamp:",query_time)
                        # print("matches timestamp:",db_time)
                        correct_count += 1
                        break
                global_idx += 1
            total_queries += query_features.size(0)

    recall = correct_count / total_queries
    return recall

# 模型评估函数
def evaluate_models_in_directory(model_dir, interval, database_loader, database_gps_data, query_loader, query_gps_data, output_file):
    with open(output_file, 'w') as f:
        for epoch in range(40, 50, interval):
            model_path = os.path.join(model_dir, f"model_epoch_{epoch}.pth")
            if not os.path.exists(model_path):
                print(f"模型 {model_path} 不存在，跳过")
                continue
            
            # 加载模型
            model = FE_Net.MainNet(channel_sizes=[128, 256, 512]).cuda()
            model.load_state_dict(torch.load(model_path))
            model.eval()
            database_features = generate_features(database_loader, model)
            # 计算 Recall@1 和 Recall@5
            recall_1 = recall_at_n_with_distance(query_loader, database_features, query_gps_data, database_gps_data, model, N=1, distance_threshold=75)
            recall_5 = recall_at_n_with_distance(query_loader, database_features, query_gps_data, database_gps_data, model, N=5, distance_threshold=75)
            
            # 保存结果
            f.write(f"Epoch {epoch}, Recall@1: {recall_1:.4f}, Recall@5: {recall_5:.4f}\n")
            print(f"Epoch {epoch} 结果已保存: Recall@1: {recall_1:.4f}, Recall@5: {recall_5:.4f}")

# 设置模型目录路径和加载间隔
model_dir = "/root/FE_Fusion/train/result_2024-11-14/saved_model"
interval = 2  # 加载间隔
output_file ="/root/FE_Fusion/train/result_2024-11-14"+"/recall_results.txt"

# 读取 GPS 数据
# database_dirs = ['/root/autodl-tmp/processed_data/dvs_vpr_2020-04-24-15-12-03','/root/autodl-tmp/processed_data/dvs_vpr_2020-04-28-09-14-11']
# gps_files = ['/root/autodl-tmp/processed_data/dvs_vpr_2020-04-24-15-12-03/filtered_interpolated_gps.txt','/root/autodl-tmp/processed_data/dvs_vpr_2020-04-28-09-14-11/filtered_interpolated_gps.txt']
database_dirs=['/root/autodl-tmp/processed_data/dvs_vpr_2020-04-21-17-03-03']
gps_files = ['/root/autodl-tmp/processed_data/dvs_vpr_2020-04-21-17-03-03/filtered_interpolated_gps.txt']
database_gps_data = []
for gps_file in gps_files:
    with open(gps_file, 'r') as f:
        for line in f:
            lat, lon, timestamp = line.strip().split()
            database_gps_data.append((float(lat), float(lon),float(timestamp)))

# query_dir = "/root/autodl-tmp/processed_data/dvs_vpr_2020-04-29-06-20-23"
# query_gps_file = "/root/autodl-tmp/processed_data/dvs_vpr_2020-04-29-06-20-23/filtered_interpolated_gps.txt"
query_dir = "/root/autodl-tmp/processed_data/dvs_vpr_2020-04-22-17-24-21"
query_gps_file = "/root/autodl-tmp/processed_data/dvs_vpr_2020-04-22-17-24-21/filtered_interpolated_gps.txt"
query_gps_data = []
with open(query_gps_file, 'r') as f:
    for line in f:
        lat, lon, timestamp = line.strip().split()
        query_gps_data.append((float(lat), float(lon),float(timestamp)))

# 加载数据集
datasets = [DatabaseDataset(dir, gps, transform) for dir, gps in zip(database_dirs, gps_files)]
combined_dataset = ConcatDataset(datasets)
database_loader = DataLoader(combined_dataset, batch_size=8, shuffle=False, num_workers=1)


query_dataset = DatabaseDataset(query_dir, query_gps_file, transform)
query_loader = DataLoader(query_dataset, batch_size=8, shuffle=False, num_workers=2)

# 评估模型
evaluate_models_in_directory(model_dir, interval, database_loader, database_gps_data, query_loader, query_gps_data, output_file)
