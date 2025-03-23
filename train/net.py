import torch
import torch.nn as nn
import torch.optim as optim
from models import FE_Net, EST_Net

class Net(nn.Module):
    def __init__(self, use_event, use_frame, use_vpr, channel_sizes, use_adapter=False, use_dift=False):
        super(Net, self).__init__()
        if use_frame:
            self.main_model = FE_Net.MainNet(channel_sizes, use_event=False)  # Only frames
        elif use_event:
            self.main_model = FE_Net.MainNet(channel_sizes, use_frame=False)  # Only events
        else:
            self.main_model = FE_Net.MainNet(channel_sizes)  # Both frames and events
        if use_vpr:
            self.est_model = EST_Net.EST_Net(use_adapter=use_adapter)
        self.use_vpr = use_vpr
        self.use_adapter = use_adapter

    def forward(self, frame_batch, event_volume_batch):
        if self.use_vpr:
            # 这里应该直接对 B Len 4 的tensor进行处理
            event_volume_batch = self.est_model(event_volume_batch) # event_volume: tensor [len, 4]
            if self.use_adapter:
                return self.main_model(frame_batch, event_volume_batch)
            else:
                raise ValueError("Not implemented")
        else:
            return self.main_model(frame_batch, event_volume_batch)