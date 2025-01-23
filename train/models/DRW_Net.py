import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
# DRWNet
class DRW_Net(nn.Module):
    def __init__(self,hidden_size=12):
        super(DRW_Net, self).__init__()
        # Fully connected layers for average and max weights
        self.fc1_avg = nn.Linear(3, hidden_size)  # 3 -> hidden_size
        self.fc2_avg = nn.Linear(hidden_size, 3)  # hidden_size -> 3
        self.fc1_max = nn.Linear(3, hidden_size)  # 3 -> hidden_size
        self.fc2_max = nn.Linear(hidden_size, 3)  # hidden_size -> 3
        self.relu = nn.ReLU()

    def forward(self, D1, D2, D3):
        device = D1.device
        D2 = D2.to(device)
        D3 = D3.to(device)
        # Calculate the channel-wise average and maximum of the sub-descriptors
        G_avg = torch.tensor([
            D1.mean().item(),
            D2.mean().item(),
            D3.mean().item()
        ],device=device).unsqueeze(0)  # Shape: (1, 3)
        
        G_max = torch.tensor([
            D1.max().item(),
            D2.max().item(),
            D3.max().item()
        ],device=device).unsqueeze(0)  # Shape: (1, 3)
        
        # Learn weights from the average and maximum
        w_avg = self.fc2_avg(self.relu(self.fc1_avg(G_avg)))
        w_max = self.fc2_max(self.relu(self.fc1_max(G_max)))
        
        # Combine and normalize weights using softmax
        w = F.softmax(w_avg + w_max, dim=-1)
        
        # Split the weights for each descriptor
        w1, w2, w3 = w[:, 0], w[:, 1], w[:, 2]
        
        # Ensure w has the correct shape for broadcasting
        w1 = w1.view(-1, 1)
        w2 = w2.view(-1, 1)
        w3 = w3.view(-1, 1)
        '''
        print("w1:",w1)
        print("w2:",w2)
        print("w3:",w3)
        print("sum:",w1+w2+w3)
        '''
        # Weight the original descriptors
        D1_weighted = D1 * w1
        D2_weighted = D2 * w2
        D3_weighted = D3 * w3
        '''
        print("D1:",D1)
        print("D2:",D2)
        print("D3:",D3)
        '''
        # Sum the weighted descriptors
        D_final = D1_weighted + D2_weighted + D3_weighted
        
        return D_final
'''
M1 = torch.zeros(16,16384)
M2 = torch.zeros(16,16384)
M3 = torch.zeros(16,16384)
input_size = M2.shape[1]
model = DRW_Net(input_size)

output = model(M1,M2,M3)
#print(output.shape)  # Should print torch.Size([1, 16384])
'''

# Triplet loss
triplet_loss = nn.TripletMarginLoss(margin = 0.1,p=2,eps=1e-7)
anchor = torch.randn(100,128,requires_grad = True)
positive = torch.randn(100, 128, requires_grad=True)
negative = torch.randn(100, 128, requires_grad=True)
