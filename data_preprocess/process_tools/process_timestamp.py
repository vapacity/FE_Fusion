import os
import h5py
#from process_event import process_event
#from process_frame import process_frame
#from read_nmea import process_gps
from tqdm import tqdm
import numpy as np
from geopy.distance import geodesic

file_name=[
'dvs_vpr_2020-04-21-17-03-03',
'dvs_vpr_2020-04-22-17-24-21',
#'dvs_vpr_2020-04-24-15-12-03',
'dvs_vpr_2020-04-27-18-13-29',
#'dvs_vpr_2020-04-28-09-14-11',
#'dvs_vpr_2020-04-29-06-20-23'
]

# function: get_index
# 从frame的文件夹中获得所有frame的文件名（去掉后缀后为时间戳），并添加进写入一个文件中
def write_timestamp_from_dir(frame_dir, processed_dir, output_file):
    # 获取 frame_path 目录下的所有文件
    file_names = os.listdir(frame_dir)
    file_names_graph = os.listdir(processed_dir)
    
    # 过滤出 PNG 文件并去掉后缀，得到时间戳
    timestamps = [os.path.splitext(file_name)[0] for file_name in file_names if file_name.endswith('.png')]
    timestamps_graph = [os.path.splitext(file_name)[0] for file_name in file_names_graph if file_name.endswith('.pt')]
    # 交集
    timestamps = [timestamp for timestamp in timestamps if timestamp in timestamps_graph]
    # 将时间戳排序
    timestamps.sort()
    # 将时间戳写入输出文件
    with open(output_file, 'w') as f:
        for timestamp in timestamps:
            f.write(f"{timestamp}\n")

