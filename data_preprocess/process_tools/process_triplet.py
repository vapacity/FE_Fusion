import os
import h5py
#from process_event import process_event
#from process_frame import process_frame
#from read_nmea import process_gps
from tqdm import tqdm
import numpy as np
from geopy.distance import geodesic

file_name=[
#'dvs_vpr_2020-04-21-17-03-03',
#'dvs_vpr_2020-04-22-17-24-21',
'dvs_vpr_2020-04-24-15-12-03',
#'dvs_vpr_2020-04-27-18-13-29',
'dvs_vpr_2020-04-28-09-14-11',
'dvs_vpr_2020-04-29-06-20-23']


def read_hdf5_gps_data(h5_file_path):
    gps_data = []

    with h5py.File(h5_file_path, 'r') as f:
        for group_name in f.keys():
            group = f[group_name]
            if 'interpolated_lat' in group and 'interpolated_lon' in group:
                interpolated_lat = group['interpolated_lat'][()]
                interpolated_lon = group['interpolated_lon'][()]
                gps_data.append((group_name, interpolated_lat, interpolated_lon))
        
    return gps_data

# def find_triplet_samples(gps_data, pos_threshold=10, neg_threshold=75):
#     triplet_samples = []

#     for anchor_idx, anchor in enumerate(tqdm(gps_data, desc="Finding triplet samples")):
#         anchor_name, anchor_lat, anchor_lon = anchor
#         pos_samples = []
#         neg_samples = []
        
#         for sample_idx, sample in enumerate(gps_data):
#             sample_name, sample_lat, sample_lon = sample
#             if anchor_name == sample_name:
#                 continue

#             distance = geodesic((anchor_lat, anchor_lon), (sample_lat, sample_lon)).meters

#             if distance < pos_threshold:
#                 pos_samples.append(sample_idx)
#             elif distance > neg_threshold:
#                 neg_samples.append(sample_idx)
        
#         if pos_samples and len(neg_samples) >= 10:
#             positive_sample = np.random.choice(pos_samples)
#             negative_samples = np.random.choice(neg_samples, 10, replace=False)
#             triplet_samples.append((anchor_idx, positive_sample, negative_samples))
    
#     return triplet_samples

# # 示例调用
# triplet_samples = find_triplet_samples(gps_data)
# print(triplet_samples)
# print(f"找到 {len(triplet_samples)} 个三元组样本")

# def save_triplet_samples_to_hdf5(h5_file_path, gps_data,triplet_samples):
#     with h5py.File(h5_file_path, 'a') as f:
#         for anchor_idx, positive_sample, negative_samples in triplet_samples:
#             group_name = gps_data[anchor_idx][0]
#             group = f[group_name]
#             if 'positive' in group:
#                 del group['positive']
#             if 'negative' in group:
#                 del group['negative']
#             group.create_dataset('positive', data=positive_sample)
#             group.create_dataset('negative', data=negative_samples)


# 输入两个路径query和database，返回三元组
# 三元组在这里直接存储实际数据，需要统一格式。把query的一套frame（这个存储路径），event_volumn,gps_data以及对应negative和positive的对应数据都得提取出来，能通过一个文件直接访问所有内容。
def read_gps_file(gps_file_path):
    """读取GPS文件并返回包含(lat, lon, timestamp)的列表"""
    gps_data = []
    with open(gps_file_path, 'r') as file:
        for line in file:
            lat, lon, timestamp = line.strip().split()
            gps_data.append((float(lat), float(lon), float(timestamp)))
    return gps_data

def find_triplet_samples(query_gps_path, database_gps_paths, save_path,pos_threshold=10, neg_threshold=75):
    # 读取query和database的GPS数据
    query_gps_data = read_gps_file(query_gps_path)
    database_gps_data = []
    
    # 读取所有的database文件路径
    for gps_path in database_gps_paths:
        database_gps_data.extend(read_gps_file(gps_path))
    
    triplet_samples = []

    # 遍历query_gps中的每个点，查找对应的三元组
    for anchor_idx, anchor in enumerate(tqdm(query_gps_data, desc="Finding triplet samples")):
        anchor_lat, anchor_lon, anchor_time = anchor
        pos_samples = []
        neg_samples = []

        # 遍历database_gps中的每个点
        for sample_idx, sample in enumerate(database_gps_data):
            sample_lat, sample_lon, sample_time = sample

            # 计算anchor和sample之间的距离
            distance = geodesic((anchor_lat, anchor_lon), (sample_lat, sample_lon)).meters

            # 根据距离筛选正样本和负样本
            if distance < pos_threshold:
                pos_samples.append(sample_time)
            elif distance > neg_threshold:
                neg_samples.append(sample_time)

        # 当存在正样本且负样本数目不小于10时，随机选择一个正样本和10个负样本
        if pos_samples and len(neg_samples) >= 10:
            positive_sample_time = np.random.choice(pos_samples)
            negative_sample_times = np.random.choice(neg_samples, 10, replace=False)
            triplet_samples.append((anchor_time, positive_sample_time, negative_sample_times))
    
        # 检查文件是否存在
    if os.path.exists(save_path):
        # 以追加模式打开文件，避免覆盖现有内容
        with open(save_path, 'a') as file:
            for triplet in triplet_samples:
                # 将 triplet 格式化为字符串，每个元素用逗号分隔
                line = f"{triplet[0]}, {triplet[1]}, {', '.join(map(str, triplet[2]))}\n"
                file.write(line)
    else:
        # 如果文件不存在，创建文件并写入内容
        with open(save_path, 'w') as file:
            for triplet in triplet_samples:
                line = f"{triplet[0]}, {triplet[1]}, {', '.join(map(str, triplet[2]))}\n"
                file.write(line)

        return triplet_samples
    