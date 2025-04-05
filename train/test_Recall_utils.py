import os
import torch
from torch.nn import CosineSimilarity
from geopy.distance import geodesic


# 定义 Recall 计算函数
# 这里的query_loader和train的不同
def recall_at_n_with_distance(query_loader, database_features, query_gps_data, database_gps_data, model, timestamps_list, N=5, distance_threshold=75, use_dift=False):
    total_queries = 0
    correct_count = 0
    query_global_idx = 0
    database_features = database_features.clone().detach().to("cuda")

    with torch.no_grad():
        if use_dift:
           pass
        else: 
            for frames, event_volumes, timestamp_batch in query_loader:
                frames, event_volumes = frames.cuda(), event_volumes.cuda()
                query_features = model(frames, event_volumes)
                cos = CosineSimilarity(dim=2, eps=1e-8)
                similarity_scores = cos(query_features.unsqueeze(1), database_features.unsqueeze(0))    #[B, length]
                top_n_scores, top_n_indices = torch.topk(similarity_scores, N, dim=1, largest=True, sorted=True)
                
            
                for i in range(query_features.size(0)):     # 0到B-1，也就是batch内index
                    query_lat, query_lon, query_time = query_gps_data[timestamp_batch[i]]
                    for idx in top_n_indices[i]:
                        db_lat, db_lon, db_time = database_gps_data[timestamps_list[idx.item()]]
                        distance = geodesic((query_lat, query_lon), (db_lat, db_lon)).meters
                        if distance < distance_threshold:
                            correct_count += 1
                            break
                    query_global_idx += 1
                total_queries += query_features.size(0)

    recall = correct_count / total_queries
    return recall
