import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import FE_Net
from loadData import TripletDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# 定义多负样本的 Triplet Loss
class MultiNegativeTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(MultiNegativeTripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negatives):
        # 计算 anchor 和 positive 之间的欧氏距离
        positive_distance = torch.nn.functional.pairwise_distance(anchor, positive)
        losses = []
        # 对每个负样本计算 anchor 和 negative 之间的距离，并且使用三元组损失计算
        for negative in negatives:
            negative_distance = torch.nn.functional.pairwise_distance(anchor, negative)
            loss = torch.nn.functional.relu(positive_distance - negative_distance + self.margin)
            losses.append(loss)
        
        # 对所有负样本的损失取平均
        losses = torch.stack(losses, dim=1)  # [batch_size, num_negatives]
        return losses.mean(dim=1).mean()  # 对每个样本的所有负样本取平均，然后对整个批次取平均


query_dir = "/root/autodl-tmp/processed_data/dvs_vpr_2020-04-24-15-12-03"
database_dirs =  ["/root/autodl-tmp/processed_data/dvs_vpr_2020-04-28-09-14-11","/root/autodl-tmp/processed_data/dvs_vpr_2020-04-29-06-20-23"]
txt_file = "/root/autodl-tmp/processed_data/dvs_vpr_2020-04-24-15-12-03/query_triplet_test.txt"

# 定义数据变换
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 创建数据集和 DataLoader
dataset = TripletDataset(txt_file, query_dir, database_dirs, transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

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

# def validate(model, dataloader):
#     model.eval()  # 设置模型为验证模式
#     correct_count = 0
#     total_count = 0
#     with torch.no_grad():  # 在验证时不计算梯度
#         for batch in dataloader:
#             query_frames, query_event_volumes, pos_frames, pos_event_volumes, neg_frames, neg_event_volumes = batch

#             query_frames, query_event_volumes = query_frames.cuda(), query_event_volumes.cuda()
#             pos_frames, pos_event_volumes = pos_frames.cuda(), pos_event_volumes.cuda()

#             # 前向传播
#             query_output = model(query_frames, query_event_volumes)
#             pos_output = model(pos_frames, pos_event_volumes)

#             # 计算准确率：如果锚点和正样本之间的距离小于负样本，则认为正确
#             positive_distance = torch.nn.functional.pairwise_distance(query_output, pos_output)
            
#             correct = positive_distance < torch.tensor(0.1).cuda()  # 你可以调整这个阈值
#             correct_count += correct.sum().item()
#             total_count += len(query_frames)

#     accuracy = correct_count / total_count
#     return accuracy
loss_file = "/root/FE_Fusion/train/loss.txt"
save_dir = "/root/FE_Fusion/train/saved_models"
# 训练循环
num_epochs = 50  # 设定训练的 epoch 数量
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    epoch_loss = 0
    #  query_frames, query_event_volumes, pos_frames, pos_event_volumes, neg_frames, neg_event_volumes = batch
    # 使用 tqdm 显示进度条
    with tqdm(total=len(dataloader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
        for batch in dataloader:
            # 将数据移动到 GPU
            query_frames, query_event_volumes, pos_frames, pos_event_volumes, neg_frames, neg_event_volumes = batch

            query_frames, query_event_volumes = query_frames.cuda(), query_event_volumes.cuda()
            pos_frames, pos_event_volumes = pos_frames.cuda(), pos_event_volumes.cuda()
            neg_frames = [neg.cuda() for neg in neg_frames]
            neg_event_volumes = [neg.cuda() for neg in neg_event_volumes]

            # 清零优化器梯度
            optimizer.zero_grad()

            # 前向传播
            anchor_output = model(query_frames, query_event_volumes)  # 计算锚点样本的特征表示
            pos_output = model(pos_frames, pos_event_volumes)  # 计算正样本的特征表示

            # 对每个负样本计算损失
            negative_outputs = [model(neg_frame, neg_event_volume) for neg_frame, neg_event_volume in zip(neg_frames, neg_event_volumes)]
            batch_loss = criterion(anchor_output, pos_output, negative_outputs)

            # 累加当前批次的损失
            epoch_loss += batch_loss.item()

            # 反向传播并优化
            batch_loss.backward()
            optimizer.step()

            # 更新进度条上的损失信息
            pbar.set_postfix(loss=batch_loss.item())
            pbar.update(1)
    
    # 打印当前 epoch 的平均损失
    average_loss = epoch_loss / len(dataloader) * 10
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')

    # 打开文件，追加写入模式 ('a') 用于将每次 epoch 的损失写入文件
    with open(loss_file, 'a') as f:  # 使用 'a' 模式表示追加写入
        f.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}\n')  # 记录损失，并换行

    # 保存模型
    model_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
    torch.save(model.state_dict(), model_path)