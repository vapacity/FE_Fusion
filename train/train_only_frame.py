import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import FE_Net
from loadData import TripletDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from DatabaseDataset import DatabaseDataset
from torch.nn import CosineSimilarity

# 首先定义一个简单的变换函数，将图像缩放到256x256
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

query_dir = "/root/autodl-tmp/processed_data/dvs_vpr_2020-04-29-06-20-23"
database_dirs =  ["/root/autodl-tmp/processed_data/dvs_vpr_2020-04-24-15-12-03","/root/autodl-tmp/processed_data/dvs_vpr_2020-04-28-09-14-11"]
txt_file = "/root/autodl-tmp/processed_data/dvs_vpr_2020-04-29-06-20-23/triplet_result.txt"

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



import torch
import torch.nn as nn

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
dataset = TripletDataset(txt_file, query_dir, database_dirs, transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
databaseDataset = DatabaseDataset(database_dirs=database_dirs,transform=transform)
database_loader = DataLoader(databaseDataset, batch_size=8, shuffle=False)
# 定义模型（假设你有一个叫 FE_Net.MainNet 的网络）
channel_sizes = [128, 256, 512]
model = FE_Net.MainNet(channel_sizes).cuda()  # 将模型移动到 GPU

# 初始化优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = MultiNegativeTripletLoss(margin=0.1).cuda()  # 使用自定义的多负样本三元组损失函数

# 初始化权重
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

model.apply(weights_init)

loss_history = []
accuracy_history = []

loss_file = "/root/FE_Fusion/train/result_2024-11-10/loss.txt"
save_dir = "/root/FE_Fusion/train/result_2024-11-10/saved_model"
# 训练循环
# 训练循环
num_epochs = 50  # 设定训练的 epoch 数量
for epoch in range(num_epochs):
    # 更新数据库特征 (可以选择在每个 epoch 开始或结束时更新)
    database_features,database_frames, database_event_volumes = update_database_features(model, database_loader)
    #print("database dimensions:",database_features.size(),database_frames.size(),database_event_volumes.size())

    epoch_loss = 0
    model.train()  # 设置模型为训练模式
    
    with tqdm(total=len(dataloader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
        for batch in dataloader:
            # 将数据移动到 GPU

            query_frames, query_event_volumes, pos_frames, pos_event_volumes, neg_frames ,neg_event_volumes= batch
            query_frames, query_event_volumes = query_frames.cuda(), query_event_volumes.cuda()
            pos_frames, pos_event_volumes = pos_frames.cuda(), pos_event_volumes.cuda()
            # print("query_frames:",query_frames.size())
            neg_frames = neg_frames.cuda() #[8, 10, 1, 256, 256]
            neg_event_volumes =neg_event_volumes.cuda() # [8, 10, 2, 256, 256]
            
            # print("neg frames:",neg_frames.size())
            # print("neg event:",neg_event_volumes.size())
            # 清零优化器梯度
            optimizer.zero_grad()

            # 前向传播计算 query 和 pos 的特征表示
            anchor_output = model(query_frames, query_event_volumes)  # 锚点特征
            pos_output = model(pos_frames, pos_event_volumes)  # 正样本特征

            database_features = database_features.cuda()

            distances = torch.cdist(anchor_output, database_features)  # [batch_size, database_size]

            bottom_n_scores, bottom_n_indices = torch.topk(distances, 1, dim=1, largest=True, sorted=True)
      
            # 遍历每个批次中的样本，逐个提取最远的负样本
            for batch_idx in range(bottom_n_indices.size(0)):  # 遍历 batch 中的每个样本
                # 选取该样本的最远的负样本
                selected_neg_index = bottom_n_indices[batch_idx, 0]  # 选择距离最远的负样本
                # 替换掉当前样本的负样本
                neg_frames[batch_idx][0] = database_frames[selected_neg_index].cuda()  # 替换负样本帧
                neg_event_volumes[batch_idx][0] = database_event_volumes[selected_neg_index].cuda()  # 替换负样本事件体
                
            # 确保 neg_frames 和 neg_event_volumes 是 list 或者 tensor
            # neg_frames_tensor = torch.stack(neg_frames)  # 使用 stack
            # neg_event_volumes_tensor = torch.stack(neg_event_volumes)  # 使用 stack

            # # 打印最终拼接后负样本的形状
            # print(f"Final neg_frames_tensor shape: {neg_frames_tensor.shape}")
            # print(f"Final neg_event_volumes_tensor shape: {neg_event_volumes_tensor.shape}")

            # 计算负样本特征
            # negative_outputs = model(neg_frames, neg_event_volumes)  # 一次性计算负样本特征
            batch_size = query_frames.size(0)  # 获取批次大小
            num_negatives = neg_frames.size(1)  # 获取负样本数量，假设为 10

            # 初始化用于存储所有负样本特征的列表
            all_negative_outputs = []

            # 逐个计算每个负样本的特征
            for i in range(num_negatives):
                # 提取第 i 个负样本
                neg_frame = neg_frames[:, i, :, :, :]  # 形状为 [batch_size, 1, 256, 256]
                neg_event_volume = neg_event_volumes[:, i, :, :, :]  # 形状为 [batch_size, 2, 256, 256]

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

    # 打开文件，追加写入模式 ('a') 用于将每次 epoch 的损失写入文件
    with open(loss_file, 'a') as f:  # 使用 'a' 模式表示追加写入
        f.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss}\n')  # 记录损失，并换行

    # 保存模型
    model_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
    torch.save(model.state_dict(), model_path)