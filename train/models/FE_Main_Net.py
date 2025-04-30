import models.TSFE_Net as TSFE_Net
import models.MF_Net as MF_Net
import models.DRW_Net as DRW_Net
import models.GCN_Net as GCN_Net
import torch
import torch.nn as nn


class MainNet(nn.Module):
    def __init__(self, channel_sizes, use_frame=True, use_event=True, event_vpr=False, graph_as_frame=False):
        super(MainNet, self).__init__()
        self.tsfe_net = TSFE_Net.TSFE_Net(use_event=use_event, use_frame=use_frame, event_vpr=event_vpr)  # 假设 TSFE_Net 在 TSFE_Net 模块中定义
        self.mf_main_net = MF_Net.MF_MainNet(channel_sizes)  # 假设 MF_MainNet 在 MF_Net 模块中定义
        self.mf_sub_net1 = MF_Net.MF_SubNet1(channel_sizes)  # 假设 MF_SubNet1 在 MF_Net 模块中定义
        self.mf_sub_net2 = MF_Net.MF_SubNet2(channel_sizes)  # 假设 MF_SubNet2 在 MF_Net 模块中定义
        self.drw_net = DRW_Net.DRW_Net(num_descriptors=3)  # 假设 DRW_Net 在 DRW_Net 模块中定义
        self.gcn_net = GCN_Net.GCN_Net()
        self.graph_as_frame = graph_as_frame

    def forward(self, frames, events):
        if self.graph_as_frame:
            M1 = self.gcn_net(frames)
            drw_output = M1
        else:
            # TSFE_Net 的前向传播
            tsfe_output = self.tsfe_net(frames, events)
            
            # MF_Net 的前向传播
            S1,S2,S3,M1 = self.mf_main_net(tsfe_output)
            
            #print('testpoint1: S1',S1.shape,'S2',S2.shape,'S3',S3.shape)
            M3, processed_S2 =self.mf_sub_net1(S1,S2)
            M2 = self.mf_sub_net2(processed_S2,S3)
            #print('testpoint2: M1',M1.shape,'M2',M2.shape,'M3',M3.shape)
            # DRW_Net 的前向传播
            descriptors = torch.stack([M1,M2,M3], dim=1)
            drw_output = self.drw_net(descriptors)
        
        # 返回主网络的输出
        return drw_output


