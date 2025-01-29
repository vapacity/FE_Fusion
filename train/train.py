import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import FE_Net, EST_Net
from dataloader.queryLoader import QueryDataset
from dataloader.databaseLoader import DatabaseDataset
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train FE-Net with different configurations.")
    parser.add_argument('--use_frame', action='store_true', help="Use only frames (default: False)")
    parser.add_argument('--use_event', action='store_true', help="Use only events (default: False)")
    parser.add_argument('--both', action='store_true', help="Use both frames and events (default: True)")
    parser.add_argument('--event_vpr', action='store_true', help="Use both frames and events (default: False)")
    return parser.parse_args()


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

processed_data_path = '/root/autodl-tmp/processed_data/'
save_path = '/root/Refactor_FE/result/'
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
# 获取路径
def generate_paths(exp_item):
    query_path = processed_data_path+experiment_item[exp_item]['query']
    database_paths = [f"{processed_data_path}{db}" for db in experiment_item[exp_item]['database']]
    triplet_path = processed_data_path+'triplets/'+exp_item+"/triplet_result.txt"
    return query_path,database_paths,triplet_path

def update_database_features(model, database_loader):
    model.eval()
    database_features = []
    database_frames = []
    database_event_volumes = []

    with torch.no_grad():
        for db_batch in tqdm(database_loader, desc="Processing database batches"):
            db_frames, db_event_volumes = db_batch
            db_frames, db_event_volumes = db_frames.cuda(), db_event_volumes.cuda()

            # 提取特征
            db_features = model(db_frames, db_event_volumes)

            # 将特征保存到列表中
            database_features.append(db_features.cpu())
            database_frames.append(db_frames.cpu())
            database_event_volumes.append(db_event_volumes.cpu())

    # 使用 torch.cat 将不同批次的数据拼接起来，而不是 stack
    database_features = torch.cat(database_features, dim=0)
    database_frames = torch.cat(database_frames, dim=0)
    database_event_volumes = torch.cat(database_event_volumes, dim=0)
    model.train()
    return database_features, database_frames, database_event_volumes

class MultiNegativeTripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(MultiNegativeTripletLoss, self).__init__()
        self.triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negatives):
        # anchor 和 positive 的形状为 [batch_size, feature_dim]
        # negatives 的形状为 [batch_size, num_negatives, feature_dim]
        
        # 存储每个负样本的损失
        all_triplet_losses = []

        # 遍历每个负样本
        for i in range(negatives.size(1)):
            # 提取第 i 个负样本，形状为 [batch_size, feature_dim]
            negative = negatives[:, i, :]
            
            # 计算三元组损失
            triplet_loss = self.triplet_loss_fn(anchor, positive, negative)
            all_triplet_losses.append(triplet_loss)
        
        # 将所有负样本的损失堆叠并求平均
        all_triplet_losses = torch.stack(all_triplet_losses)  # [num_negatives]
        return all_triplet_losses.mean()

# 创建数据集和 DataLoader

query_dir,database_dirs,triplet_file = generate_paths('experiment_1')
save_dir = save_path + "result_2025_1_23/saved_model"
loss_file = save_path +"result_2025_1_23/loss.txt"

dataset = QueryDataset(triplet_file, query_dir, database_dirs, transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
databaseDataset = DatabaseDataset(database_dirs=database_dirs,transform=transform)
database_loader = DataLoader(databaseDataset, batch_size=8, shuffle=False)


start_epoch = 1  # 默认从第 1 个 epoch 开始
# model_path = os.path.join(save_dir, "model_epoch_0.pth")  # 例如加载到第 50 个 epoch 的模型


channel_sizes = [128, 256, 512]
args = parse_args()

# Default to using both frames and events if no specific argument is provided
if args.use_frame and args.use_event:
    raise ValueError("Cannot use both 'use_frame' and 'use_event' at the same time.")

class Net(nn.Module):
    def __init__(self, use_event, use_frame, use_vpr, use_adapter=False):
        super(Net, self).__init__()
        if use_frame:
            self.main_model = FE_Net.MainNet(channel_sizes, use_event=False)  # Only frames
        elif use_event:
            self.main_model = FE_Net.MainNet(channel_sizes, use_frame=False)  # Only events
        else:
            self.main_model = FE_Net.MainNet(channel_sizes)  # Both frames and events
        if use_vpr:
            self.est_model = EST_Net(use_adapter)
        self.use_vpr = use_vpr
        self.use_adapter = use_adapter

    def forward(self, frame_batch, event_volume_batch):
        if self.use_vpr:
            event_volume_batch = torch.stack([self.est_model(event_volume) for event_volume in event_volume_batch]) # event_volume: tensor [len, 4]
            if self.use_adapter:
                return self.main_model(frame_batch, event_volume_batch)
        else:
            return self.main_model(frame_batch, event_volume_batch)
            
model = Net(args.use_event, args.use_frame, args.event_vpr, use_adapter=True).cuda()

# 如果存在模型文件，加载模型权重，并设置起始 epoch
# if os.path.exists(model_path):
#     print(f"加载模型权重: {model_path}")
#     model.load_state_dict(torch.load(model_path))
#     start_epoch = int(model_path.split('_')[-1].split('.')[0])  # 从模型文件名中提取 epoch 数

# 初始化优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = MultiNegativeTripletLoss(margin=0.1).cuda()  # 使用自定义的多负样本三元组损失函数


loss_history = []
accuracy_history = []



# 训练循环
# 训练循环
num_epochs = 100  # 设定训练的 epoch 数量
for epoch in range(num_epochs):
    # 更新数据库特征 (可以选择在每个 epoch 开始或结束时更新)
    database_features, database_frames, database_event_volumes = update_database_features(model, database_loader)
    #print("database dimensions:",database_features.size(),database_frames.size(),database_event_volumes.size())

    epoch_loss = 0
    model.train()  # 设置模型为训练模式
    
    with tqdm(total=len(dataloader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
        for batch in dataloader:
            # 将数据移动到 GPU
            query_frame_single, query_event_volume_single, pos_frame_single, pos_event_volume_single, neg_frames_multi ,neg_event_volumes_multi = batch
#            query_frame, query_event_volume, pos_frame, pos_event_volume, neg_frames, neg_event_volumes
            query_frame_single, query_event_volume_single = query_frame_single.cuda(), query_event_volume_single.cuda()
            pos_frame_single, pos_event_volume_single = pos_frame_single.cuda(), pos_event_volume_single.cuda()
            # print("query_frames:",query_frames.size())
            neg_frames_multi = neg_frames_multi.cuda() #[batch_size, num_neg, 1, 256, 256]
            neg_event_volumes_multi =neg_event_volumes_multi.cuda() # not use vpr: [8, num_neg, 2, 256, 256] use vpr: [batch_size, num_neg, n, 4]
            
            # print("neg frames:",neg_frames.size())
            # print("neg event:",neg_event_volumes.size())
            # 清零优化器梯度
            optimizer.zero_grad()

            # 前向传播计算 query 和 pos 的特征表示
            anchor_output = model(query_frame_single, query_event_volume_single)  # 锚点特征
            pos_output = model(pos_frame_single, pos_event_volume_single)  # 正样本特征

            database_features = database_features.cuda()

            distances = torch.cdist(anchor_output, database_features)  # [batch_size, database_size]

            bottom_n_scores, bottom_n_indices = torch.topk(distances, 1, dim=1, largest=True, sorted=True)
      
            # 遍历每个批次中的样本，逐个提取最远的负样本
            for batch_idx in range(bottom_n_indices.size(0)):  # 遍历 batch 中的每个样本
                # 选取该样本的最远的负样本
                selected_neg_index = bottom_n_indices[batch_idx, 0]  # 选择距离最远的负样本
                # 替换掉当前样本的负样本
                neg_frames_multi[batch_idx][0] = database_frames[selected_neg_index].cuda()  # 替换负样本帧
                neg_event_volumes_multi[batch_idx][0] = database_event_volumes[selected_neg_index].cuda()  # 替换负样本事件体
                
            batch_size = query_frame_single.size(0)  # 获取批次大小
            num_negatives = neg_frames_multi.size(1)  # 获取负样本数量，假设为 10

            # 初始化用于存储所有负样本特征的列表
            all_negative_outputs = []

            # 逐个计算每个负样本的特征
            for i in range(num_negatives):
                # 提取第 i 个负样本
                neg_frame = neg_frames_multi[:, i, :, :, :]  # 形状为 [batch_size, 1, 256, 256]
                if not args.event_vpr:
                    neg_event_volume = neg_event_volumes_multi[:, i, :, :, :]  # 不use vpr 形状为 [batch_size, 2, 256, 256] use vpr 形状为 [batch_size, len, 4]
                else:
                    neg_event_volume = neg_event_volumes_multi[:, i, :, :]

                # 计算第 i 个负样本的特征
                neg_output = model(neg_frame, neg_event_volume)  # 形状为 [batch_size, feature_dim]

                # 将该负样本的特征添加到列表
                all_negative_outputs.append(neg_output)

            # 将所有负样本特征拼接成 [batch_size, num_negatives, feature_dim]
            negative_outputs = torch.stack(all_negative_outputs, dim=1)  # [batch_size, 10, feature_dim]

            # 计算三元组损失
            batch_loss = criterion(anchor_output, pos_output, negative_outputs)

            # 计算损失 (假设 criterion 是三元组损失函数)
            batch_loss = criterion(anchor_output, pos_output, negative_outputs)
            epoch_loss += batch_loss.item()

            # 反向传播并优化
            batch_loss.backward()
            optimizer.step()

            # 更新进度条上的损失信息
            pbar.set_postfix(loss=batch_loss.item())
            pbar.update(1)


    
    # 打印当前 epoch 的平均损失
    average_loss = epoch_loss / len(dataloader) * 10
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss}')

    # 确保目录存在，如果没有就创建它
    os.makedirs(os.path.dirname(loss_file), exist_ok=True)

    # 写入损失
    with open(loss_file, 'a') as f:
        f.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss}\n')

    # 保存模型
    model_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
    os.makedirs(save_dir,exist_ok=True)
    
    torch.save(model.state_dict(), model_path)