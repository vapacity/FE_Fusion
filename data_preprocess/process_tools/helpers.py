from datetime import datetime
import rospy
from time import mktime
def convert_time_to_rostime(year,month,day,hour,minute,second,millisecond):

    # 创建datetime对象
    dt = datetime(year, month, day, hour-2, minute, second, int(millisecond * 1e3))

    # 使用mktime将datetime对象转换为时间戳
    time_seconds = mktime(dt.timetuple())
    time_nanos = float(millisecond / 1e3)

    # 创建ROS时间戳
    ros_time = rospy.Time.from_sec(time_seconds + time_nanos)
    return ros_time.secs
    # 打印结果
    #print("ROS Timestamp: %d.%09d" % (ros_time.secs, ros_time.nsecs))

def devide_name_to_time(file_name):
    date_part, time_part = file_name.split('_',1)
    year = int(date_part[:4])
    month = int(date_part[4:6])
    day = int(date_part[6:])
    hour = int(time_part[:2])
    minute = int(time_part[2:4])
    second = int(time_part[4:6])
    return year,month,day,hour,minute,second

# 输入gps文件 返回gps数组
def read_gps_data(gps_file_path):
    return np.loadtxt(gps_file_path, delimiter=' ')

# 输入法txt文件，返回timestamp数组
def read_timestamp(timestamp_file):
    timestamp =[]
    with open(timestamp_file,'r') as f:
        for line in f:
            time = float(line.strip())
            timestamp.append(time)
    
    return timestamp

import h5py
import numpy as np
'''
def load_event_volumes_from_hdf5(input_file):
    event_volumes = []
    with h5py.File(input_file, 'r') as f:
        for group_name in f.keys():
            group = f[group_name]
            timestamp = group['timestamp'][()]
            event_volume = group['event_volume'][()]
            event_volumes.append((timestamp, event_volume))
    return event_volumes

def load_event_volumes_from_hdf5(input_file):
    event_volumes = []
    with h5py.File(input_file, 'r') as f:
        for group_name in f.keys():
            group = f[group_name]
            timestamp = group['timestamp'][()]
            print("Timestamp:", timestamp)
            if 'positive' in group:
                pos = f[str(group['positive'][()])]
                pos_timestamp = pos['timestamp'][()]
                pos_lat = pos['interpolated_lat'][()]
                pos_lon = pos['interpolated_lon'][()]
                
                print("Positive Timestamp:", pos_timestamp)
                print("Positive Latitude:", pos_lat)
                print("Positive Longitude:", pos_lon)

            if 'negative' in group:
                neg_sample_keys = [str(k) for k in group['negative'][()]]
                neg_timestamps = [str(f[k]['timestamp'][()]) for k in neg_sample_keys]
                print("Negative Timestamp:", neg_timestamps)
                neg_lat = [str(f[k]['interpolated_lat'][()]) for k in neg_sample_keys if 'interpolated_lat' in f[k]]
                print("Negative Latitude:", neg_lat)
                neg_lon = [str(f[k]['interpolated_lon'][()]) for k in neg_sample_keys if 'interpolated_lon' in f[k]]
                
                
                print("Negative Longitude:", neg_lon)

            #event_volumes.append((timestamp, pos_timestamp, neg_timestamp))
    #return event_volumes

    


import h5py
'interpolated_lat'
def inspect_hdf5_file(file_path):
    def print_attrs(name, obj):
        if isinstance(obj, h5py.Group):
            if 'interpolated_lat' in obj:
                print(f"Group: {name}")
                for ds_name, ds in obj.items():
                    if isinstance(ds, h5py.Dataset):
                        print(f"  Dataset: {ds_name}")
                        print(f"    Shape: {ds.shape}")
                        print(f"    Dtype: {ds.dtype}")
                        print(f"    Data: {ds[()]}")

    with h5py.File(file_path, 'r') as f:
        f.visititems(print_attrs)

def find_max_timestamp(h5_file_path):
    max_timestamp = None

    with h5py.File(h5_file_path, 'r') as f:
        for group_name in f.keys():
            group = f[group_name]
            timestamp = group['timestamp'][()]

            if max_timestamp is None or timestamp > max_timestamp:
                max_timestamp = timestamp
    print(max_timestamp)
    return max_timestamp

def find_min_timestamp(h5_file_path):
    min_timestamp = None

    with h5py.File(h5_file_path, 'r') as f:
        for group_name in f.keys():
            group = f[group_name]
            timestamp = group['timestamp'][()]

            if min_timestamp is None or timestamp < min_timestamp:
                min_timestamp = timestamp
    print(min_timestamp)
    return min_timestamp
h5_file_path = '/root/autodl-tmp/processed_data/event_output/dvs_vpr_2020-04-29-06-20-23.h5'
load_event_volumes_from_hdf5(h5_file_path)

# 示例调用
input_file = 'event_volumes.h5'
event_volumes = load_event_volumes_from_hdf5(input_file)

# 打印解析的数据以验证
for timestamp, event_volume in event_volumes:
    print(f'Timestamp: {timestamp}')
    print(f'Event Volume Shape: {event_volume.shape}')
'''

import numpy as np
import matplotlib.pyplot as plt

def visualize_event_data(event_file):
    """
    从 .npy 文件中加载事件数据，并可视化图像。
    Args:
        event_file (str): .npy 文件的路径
    """
    # 加载 .npy 文件中的事件数据
    event_volume = np.load(event_file)  # 形状为 (2, 260, 346)，第一个维度表示极性
    
    # 分别获取正极性和负极性的事件
    positive_events = event_volume[1]  # 极性为 1 的事件
    negative_events = event_volume[0]  # 极性为 0 的事件

    # 创建一个彩色图像，将不同极性的事件显示为不同的颜色
    img = np.zeros((260, 346, 3), dtype=np.uint8)  # 创建一个 260x346 的RGB图像

    # 对正极性事件着色，例如红色
    img[positive_events > 0] = [255, 0, 0]  # 红色表示正极性事件
    # 对负极性事件着色，例如蓝色
    img[negative_events > 0] = [0, 0, 255]  # 蓝色表示负极性事件

    # 使用 matplotlib 可视化
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.title(f"Event Visualization: {event_file}")
    plt.axis('off')  # 隐藏坐标轴
    plt.savefig("test")
    plt.show()

# # 调用 visualize_event_data 函数，显示某个 .npy 文件
# npy_file = '/root/autodl-tmp/processed_data/dvs_vpr_2020-04-21-17-03-03/event/1587452664.8194156.npy'  # 替换为实际的 .npy 文件路径
# visualize_event_data(npy_file)
