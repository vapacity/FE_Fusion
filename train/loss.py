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
        length = negatives.size(1)

        # 遍历每个负样本
        for i in range(length):
            # 提取第 i 个负样本，形状为 [batch_size, feature_dim]
            negative = negatives[:, i, :]
            
            # 计算三元组损失
            triplet_loss = self.triplet_loss_fn(anchor, positive, negative)
            all_triplet_losses.append(triplet_loss)
        
        # 将所有负样本的损失堆叠并求平均
        all_triplet_losses = torch.stack(all_triplet_losses)  # [num_negatives]
        return all_triplet_losses.mean() * length