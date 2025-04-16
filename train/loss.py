import torch
import torch.nn as nn


class MultiNegativeTripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(MultiNegativeTripletLoss, self).__init__()
        self.triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2)     # 默认是会开根号的，是L2 norm
        self.margin = margin

    def forward(self, anchor, positive, negatives):
        # anchor 和 positive 的形状为 [batch_size, feature_dim]
        # negatives 的形状为 [batch_size, num_negatives, feature_dim]
        
        # 存储每个负样本的损失
        all_triplet_losses = []
        length = negatives.size(1)
        # pos_cos_distance = 1 - torch.einsum('bi,bi->b', [anchor, positive])

        scale_factor = 10
        for i in range(length):
            # 提取第 i 个负样本，形状为 [batch_size, feature_dim]
            negative = negatives[:, i, :]

            # neg_cos_distance = 1 - torch.einsum('bi,bi->b', [anchor, negative])
            # triplet_loss = torch.clamp(pos_cos_distance - neg_cos_distance + self.margin, min=0)

            # L2损失版本
            triplet_loss = self.triplet_loss_fn(anchor, positive, negative)
            all_triplet_losses.append(triplet_loss)
        
        # 将所有负样本的损失堆叠并求平均
        all_triplet_losses = torch.stack(all_triplet_losses)  # [num_negatives, batch_size]
        return all_triplet_losses.mean() * scale_factor   # 扩大一点