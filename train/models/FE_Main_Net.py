import models.TSFE_Net as TSFE_Net
import models.MF_Net as MF_Net
import models.DRW_Net as DRW_Net
import models.GCN_Net as GCN_Net
import models.EST_Net as EST_Net
import torch
import torch.nn as nn
import models.NetVLAD as NetVLAD


class MainNet(nn.Module):
    def __init__(self, channel_sizes, use_frame=True, use_event=True, event_vpr_as_frame=False, graph_as_frame=False, fusion_vpr_mid=False):
        super(MainNet, self).__init__()
        self.tsfe_net = TSFE_Net.TSFE_Net(use_event=use_event, use_frame=use_frame, event_vpr_as_frame=event_vpr_as_frame, output_both=fusion_vpr_mid)  # 假设 TSFE_Net 在 TSFE_Net 模块中定义
        self.mf_main_net = MF_Net.MF_MainNet(channel_sizes)  # 假设 MF_MainNet 在 MF_Net 模块中定义
        self.mf_sub_net1 = MF_Net.MF_SubNet1(channel_sizes)  # 假设 MF_SubNet1 在 MF_Net 模块中定义
        self.mf_sub_net2 = MF_Net.MF_SubNet2(channel_sizes)  # 假设 MF_SubNet2 在 MF_Net 模块中定义
        self.event_vpr_as_frame = event_vpr_as_frame
        self.graph_as_frame = graph_as_frame
        self.fusion_vpr_mid = fusion_vpr_mid
        self.use_frame = use_frame
        self.use_event = use_event

        if event_vpr_as_frame:
            self.est_net = EST_Net.EST_Net()
        if fusion_vpr_mid:
            self.fusion_vpr_mid = fusion_vpr_mid
            self.mf_sub_net1_copy = MF_Net.MF_SubNet1(channel_sizes)  # 假设 MF_SubNet1 在 MF_Net 模块中定义
            self.mf_sub_net2_copy = MF_Net.MF_SubNet2(channel_sizes)  # 假设 MF_SubNet2 在 MF_Net 模块中定义
            self.cross_spatial_attention_1 = CrossSpatialAttention_v2()
            self.cross_spatial_attention_2 = CrossSpatialAttentionSingle_v2()
            self.cross_spatial_attention_3 = CrossSpatialAttentionSingle_v2()

        if graph_as_frame:
            self.gcn_net = GCN_Net.GCN_Net()
            
        self.VLAD_1 = NetVLAD.NetVLAD(dim=256)
        self.VLAD_2 = NetVLAD.NetVLAD(dim=256)
        self.VLAD_3 = NetVLAD.NetVLAD(dim=256)
        self.drw_net = DRW_Net.DRW_Net(num_descriptors=3)

    def forward(self, frames, events):
        if self.graph_as_frame:
            assert self.use_frame == False, "use_frame cannot be both True in the settings of graph_as_frame"
            if not self.use_event:
                gcn_output = self.gcn_net(frames)
                drw_output = gcn_output
            else:
                 # TSFE_Net 的前向传播
                tsfe_output = self.tsfe_net(frames, events)
                
                # MF_Net 的前向传播
                S1,S2,S3,M1_unvlad = self.mf_main_net(tsfe_output)
                
                #print('testpoint1: S1',S1.shape,'S2',S2.shape,'S3',S3.shape)
                M3_unvlad, processed_S2 =self.mf_sub_net1(S1,S2)
                M2_unvlad = self.mf_sub_net2(processed_S2,S3)
                #print('testpoint2: M1',M1.shape,'M2',M2.shape,'M3',M3.shape)
                # DRW_Net 的前向传播
                M1 = self.VLAD_1(M1_unvlad)
                M2 = self.VLAD_2(M2_unvlad)
                M3 = self.VLAD_3(M3_unvlad)
                descriptors = torch.stack([M1,M2,M3], dim=1)
                drw_output = self.drw_net(descriptors)
                drw_output = torch.cat([drw_output, gcn_output], dim=1)

        elif self.event_vpr_as_frame:
            frames = self.est_net(frames)
            if not self.fusion_vpr_mid:
                # TSFE_Net 的前向传播
                tsfe_output = self.tsfe_net(frames, events)
                
                # MF_Net 的前向传播
                S1,S2,S3,M1_unvlad = self.mf_main_net(tsfe_output)
                
                #print('testpoint1: S1',S1.shape,'S2',S2.shape,'S3',S3.shape)
                M3_unvlad, processed_S2 =self.mf_sub_net1(S1,S2)
                M2_unvlad = self.mf_sub_net2(processed_S2,S3)
                #print('testpoint2: M1',M1.shape,'M2',M2.shape,'M3',M3.shape)
                # DRW_Net 的前向传播
                M1 = self.VLAD_1(M1_unvlad)
                M2 = self.VLAD_2(M2_unvlad)
                M3 = self.VLAD_3(M3_unvlad)
                descriptors = torch.stack([M1,M2,M3], dim=1)
                drw_output = self.drw_net(descriptors)
                
            else:
                tsfe_output_1, tsfe_output_2 = self.tsfe_net(frames, events)
                S1_1,S2_1,S3_1,M1_1_unvlad = self.mf_main_net(tsfe_output_1)
                S1_2,S2_2,S3_2,M1_2_unvlad = self.mf_main_net(tsfe_output_2)
                M1_unvlad = self.cross_spatial_attention_1(M1_1_unvlad,M1_2_unvlad)

                M3_unvlad_1, processed_S2_1 =self.mf_sub_net1(S1_1,S2_1)
                M2_unvlad_1 = self.mf_sub_net2(processed_S2_1,S3_1)

                M3_unvlad_2, processed_S2_2 =self.mf_sub_net1_copy(S1_2,S2_2)
                M2_unvlad_2 = self.mf_sub_net2_copy(processed_S2_2,S3_2)

                M2_unvlad = self.cross_spatial_attention_2(M2_unvlad_1,M2_unvlad_2)
                M3_unvlad = self.cross_spatial_attention_3(M3_unvlad_1,M3_unvlad_2)

                M1 = self.VLAD_1(M1_unvlad)
                M2 = self.VLAD_2(M2_unvlad)
                M3 = self.VLAD_3(M3_unvlad)
                descriptors = torch.stack([M1,M2,M3], dim=1)
                drw_output = self.drw_net(descriptors)

        else:
            # TSFE_Net 的前向传播
            tsfe_output = self.tsfe_net(events, events)
            
            # MF_Net 的前向传播
            S1,S2,S3,M1_unvlad = self.mf_main_net(tsfe_output)
            
            #print('testpoint1: S1',S1.shape,'S2',S2.shape,'S3',S3.shape)
            M3_unvlad, processed_S2 =self.mf_sub_net1(S1,S2)
            M2_unvlad = self.mf_sub_net2(processed_S2,S3)
            #print('testpoint2: M1',M1.shape,'M2',M2.shape,'M3',M3.shape)
            # DRW_Net 的前向传播
            M1 = self.VLAD_1(M1_unvlad)
            M2 = self.VLAD_2(M2_unvlad)
            M3 = self.VLAD_3(M3_unvlad)
            descriptors = torch.stack([M1,M2,M3], dim=1)
            drw_output = self.drw_net(descriptors)
        # 返回主网络的输出
        return drw_output



class CrossSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(4, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, s1, s2):
        """
        Args:
            s1 (torch.Tensor): Input feature map 1, shape (B, C, H, W)
            s2 (torch.Tensor): Input feature map 2, shape (B, C, H, W)

        Returns:
            torch.Tensor: Attended feature map s2, shape (B, C, H, W)
            torch.Tensor: Generated spatial attention map, shape (B, 1, H, W)
        """
        avg_out_s1 = torch.mean(s1, dim=1, keepdim=True)
        max_out_s1, _ = torch.max(s1, dim=1, keepdim=True)
        avg_out_s2 = torch.mean(s2, dim=1, keepdim=True)
        max_out_s2, _ = torch.max(s2, dim=1, keepdim=True)

        x = torch.cat([avg_out_s1, max_out_s1, avg_out_s2, max_out_s2], dim=1)
        attention_map = self.sigmoid(self.conv(x))

        attended_s2 = attention_map * s2
        return attended_s2

class CrossSpatialAttentionSingle(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, s1, s2):
        """
        Args:
            s1 (torch.Tensor): Input feature map 1, shape (B, C, H, W)
            s2 (torch.Tensor): Input feature map 2, shape (B, C, H, W)

        Returns:
            torch.Tensor: Attended feature map s2, shape (B, C, H, W)
            torch.Tensor: Generated spatial attention map, shape (B, 1, H, W)
        """
        avg_out_s1 = torch.mean(s1, dim=1, keepdim=True)
        max_out_s1, _ = torch.max(s1, dim=1, keepdim=True)

        x = torch.cat([avg_out_s1, max_out_s1], dim=1)
        attention_map = self.sigmoid(self.conv(x))

        attended_s2 = attention_map * s2
        return attended_s2

class CrossSpatialAttention_v2(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(512, 256, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, s1, s2):
        """
        Args:
            s1 (torch.Tensor): Input feature map 1, shape (B, C, H, W)
            s2 (torch.Tensor): Input feature map 2, shape (B, C, H, W)

        Returns:
            torch.Tensor: Attended feature map s2, shape (B, C, H, W)
            torch.Tensor: Generated spatial attention map, shape (B, 1, H, W)
        """

        x = torch.cat([s1, s2], dim=1)
        attended = self.conv(x)
        return attended
    
class CrossSpatialAttentionSingle_v2(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.conv_2 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.conv_final = nn.Conv2d(512, 256, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, s1, s2):
        """
        Args:
            s1 (torch.Tensor): Input feature map 1, shape (B, C, H, W)
            s2 (torch.Tensor): Input feature map 2, shape (B, C, H, W)

        Returns:
            torch.Tensor: Attended feature map s2, shape (B, C, H, W)
            torch.Tensor: Generated spatial attention map, shape (B, 1, H, W)
        """
        avg_out_s1 = torch.mean(s1, dim=1, keepdim=True)
        max_out_s1, _ = torch.max(s1, dim=1, keepdim=True)
        avg_out_s2 = torch.mean(s2, dim=1, keepdim=True)
        max_out_s2, _ = torch.max(s2, dim=1, keepdim=True)

        x = torch.cat([avg_out_s1, max_out_s1], dim=1)
        attention_map_1 = self.sigmoid(self.conv(x))

        x = torch.cat([avg_out_s2, max_out_s2], dim=1)
        attention_map_2 = self.sigmoid(self.conv_2(x))

        # spatial cross attention
        attended_s1 = attention_map_2 * s1
        attended_s2 = attention_map_1 * s2

        final_attented = self.conv_final(torch.cat([attended_s1, attended_s2], dim=1))

        return final_attented