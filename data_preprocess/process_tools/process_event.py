import os
import rosbag
import numpy as np
from pathlib import Path
from tqdm import tqdm
from helpers import read_timestamp
import h5py
file_name=[
#'dvs_vpr_2020-04-21-17-03-03',
#'dvs_vpr_2020-04-22-17-24-21',
'dvs_vpr_2020-04-24-15-12-03',
#'dvs_vpr_2020-04-27-18-13-29',
'dvs_vpr_2020-04-28-09-14-11',
'dvs_vpr_2020-04-29-06-20-23'
]


# def process_event(bag_file, timestamps_file, output_file, time_tolerance=0.0125):
#     timestamps = read_timestamp(timestamps_file)

#     with rosbag.Bag(bag_file, 'r') as bag:
#         it = bag.read_messages(topics=['/dvs/events'])
#         topic, msg, t = next(it, (None, None, None))

#         total_images = len(timestamps)
#         pbar = tqdm(total=total_images, desc='processing events')

#         # 检查文件是否存在
#         if os.path.exists(output_file):
#             print(f"{output_file} already exists. Saving data to existing file.")
#         else:
#             print(f"{output_file} does not exist. Creating new file.")

#         with h5py.File(output_file, 'a') as f:
#             existing_groups = list(f.keys())
#             start_index = len(existing_groups)
            
#             for i, timestamp in enumerate(timestamps):
#                 events = []
#                 event_volume = np.zeros((2, 260, 346))  # 初始化事件数据容器
#                 start_time = timestamp - time_tolerance
#                 end_time = timestamp + time_tolerance

#                 # 跳过早于当前时间窗开始时间的事件
#                 while t and t.to_sec() < start_time:
#                     topic, msg, t = next(it, (None, None, None))

#                 # 收集时间窗内的事件
#                 while t and t.to_sec() <= end_time:
#                     events.append(msg)  # 假设这里我们仅仅收集消息，具体处理根据需求
#                     topic, msg, t = next(it, (None, None, None))
#                 for event_msg in events:
#                     for event in event_msg.events:
#                         x, y, p = event.x, event.y, int(event.polarity)
#                         event_volume[p, y, x] += 1
                
#                 # 将收集的事件保存到文件
#                 group_index = start_index + i
#                 group = f.create_group(str(group_index))
#                 group.create_dataset('timestamp', data=timestamp)
#                 group.create_dataset('event_volume', data=event_volume)
                
#                 # 释放内存
#                 del events
#                 del event_volume
                
#                 pbar.update(1)  # 更新进度条


def process_event(bag_file, timestamps_file, output_dir, time_tolerance=0.0125):
    timestamps = read_timestamp(timestamps_file)

    with rosbag.Bag(bag_file, 'r') as bag:
        it = bag.read_messages(topics=['/dvs/events'])
        topic, msg, t = next(it, (None, None, None))

        total_images = len(timestamps)
        pbar = tqdm(total=total_images, desc='Processing events')  # 初始化tqdm进度条

        # 检查输出目录是否存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"{output_dir} does not exist. Creating new directory.")

        for i, timestamp in enumerate(timestamps):
            events = []
            event_volume = np.zeros((2, 260, 346))  # 初始化事件数据容器
            start_time = timestamp - time_tolerance
            end_time = timestamp + time_tolerance

            # 跳过早于当前时间窗开始时间的事件
            while t and t.to_sec() < start_time:
                topic, msg, t = next(it, (None, None, None))

            # 收集时间窗内的事件
            while t and t.to_sec() <= end_time:
                events.append(msg)  # 假设这里我们仅仅收集消息，具体处理根据需求
                topic, msg, t = next(it, (None, None, None))
            for event_msg in events:
                for event in event_msg.events:
                    x, y, p = event.x, event.y, int(event.polarity)
                    event_volume[p, y, x] += 1

            # 将收集的事件保存为 .npy 文件，文件名使用时间戳
            timestamp_str = f"{timestamp}"  # 保留6位小数
            output_file = os.path.join(output_dir, f"{timestamp_str}.npy")
            np.save(output_file, event_volume)

            # 释放内存
            del events
            del event_volume

            # 更新进度条
            pbar.update(1)

        pbar.close()  # 完成所有任务后关闭进度条



for name in file_name:
    bag_file = '/root/autodl-fs/Brizbane_dataset/'+name+'.bag'
    timestamp_file = '/root/autodl-tmp/processed_data/'+name+'/timestamp.txt'
    output_dir = '/root/autodl-tmp/processed_data/'+name+'/event/'
    # 调用函数
    process_event(bag_file, timestamp_file,output_dir)  # 这里设定了frame_interval，根据需要调整




