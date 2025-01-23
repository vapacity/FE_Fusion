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
def get_index(frame_path,output_file):
    # 获取 frame_path 目录下的所有文件
    file_names = os.listdir(frame_path)
    
    # 过滤出 PNG 文件并去掉后缀，得到时间戳
    timestamps = [os.path.splitext(file_name)[0] for file_name in file_names if file_name.endswith('.png')]
    
    # 将时间戳排序
    timestamps.sort()
    # 将时间戳写入输出文件
    with open(output_file, 'w') as f:
        for timestamp in timestamps:
            f.write(f"{timestamp}\n")

# 调用get_index      
for name in file_name:
    frame_path = '/root/autodl-tmp/processed_data/'+name+'/frame/'
    output_file = '/root/autodl-tmp/processed_data/'+name+'/timestamp.txt'
    get_index(frame_path,output_file)
