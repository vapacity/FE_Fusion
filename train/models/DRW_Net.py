import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
# DRWNet
class DRW_Net(nn.Module):
    def __init__(self, num_descriptors, hidden_size=12):
        super(DRW_Net, self).__init__()
        # Fully connected layers for average and max weights
        self.fc1_avg = nn.Linear(num_descriptors, hidden_size)  # num_descriptors -> hidden_size
        self.fc2_avg = nn.Linear(hidden_size, num_descriptors)  # hidden_size -> num_descriptors
        self.fc1_max = nn.Linear(num_descriptors, hidden_size)  # num_descriptors -> hidden_size
        self.fc2_max = nn.Linear(hidden_size, num_descriptors)  # hidden_size -> num_descriptors
        self.relu = nn.ReLU()

    def forward(self, descriptors):
        """
        Forward pass that supports any number of descriptors
        Args:
            descriptors: Variable number of descriptor tensors, each with shape [batch_size, num_descriptors, dim]
        Returns:
            D_final: Weighted combination of all descriptors with shape [batch_size, dim]
        """
        if descriptors.shape[1] == 0:
            raise ValueError("At least one descriptor must be provided")
        
        # Calculate the channel-wise average and maximum of the sub-descriptors
        
        # channel-wise average and maximum
        G_avg = descriptors.mean(dim=2) # shape: [batch_size, num_descriptors]
        G_max = descriptors.max(dim=2).values     # shape: [batch_size, num_descriptors]
        
        # Learn weights from the average and maximum
        w_avg = self.fc2_avg(self.relu(self.fc1_avg(G_avg)))    # shape: [batch_size, num_descriptors]
        w_max = self.fc2_max(self.relu(self.fc1_max(G_max)))    # shape: [batch_size, num_descriptors]
        
        # Combine and normalize weights using softmax
        w = F.softmax(w_avg + w_max, dim=-1)  # shape: [batch_size, num_descriptors]
        
        # Weight the original descriptors
        weighted_descriptors = descriptors * w.unsqueeze(2)  # shape: [batch_size, num_descriptors, dim]
        
        # Sum along the descriptor dimension to get the final weighted representation
        D_final = weighted_descriptors.sum(dim=1)  # shape: [batch_size, dim]
        D_final = F.normalize(D_final, p=2, dim=1)

        
        return D_final
