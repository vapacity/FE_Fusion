import torch
import torch.nn as nn
import torch.optim as optim
from models import FE_Net, EST_Net

class Net(nn.Module):
    def __init__(self, use_event, use_frame, event_vpr, channel_sizes, use_dift=False):
        super(Net, self).__init__()
        if use_frame:
            self.main_model = FE_Net.MainNet(channel_sizes, use_event=False, event_vpr=False)  # Only frames
        elif use_event:
            self.main_model = FE_Net.MainNet(channel_sizes, use_frame=False, event_vpr=event_vpr)  # Only events
        else:
            self.main_model = FE_Net.MainNet(channel_sizes)  # Both frames and events
        if event_vpr:
            self.est_model = EST_Net.EST_Net()
        self.use_vpr = event_vpr

    def forward(self, frame_batch, event_volume_batch):
        if self.use_vpr:
            # 这里应该直接对 B Len 4 的tensor进行处理
            return self.main_model(frame_batch, self.est_model(event_volume_batch))
        return self.main_model(frame_batch, event_volume_batch)