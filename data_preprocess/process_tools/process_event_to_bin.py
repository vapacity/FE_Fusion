import os
import rosbag
import numpy as np
from pathlib import Path
from tqdm import tqdm
from helpers import read_timestamp
import h5py


def process_event_to_txt(bag_file, timestamps_file, output_dir, time_tolerance=0.0125):
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
            start_time = timestamp - time_tolerance
            end_time = timestamp + time_tolerance

            # 跳过早于当前时间窗开始时间的事件
            while t and t.to_sec() < start_time:
                topic, msg, t = next(it, (None, None, None))

            # 收集时间窗内的事件
            while t and t.to_sec() <= end_time:
                events.append(msg)  # 假设这里我们仅仅收集消息，具体处理根据需求
                topic, msg, t = next(it, (None, None, None))

            # 将收集的事件逐行写入 .txt 文件
            timestamp_str = f"{timestamp}"  # 保留6位小数
            output_file = os.path.join(output_dir, f"bin_{timestamp_str}.txt")
            with open(output_file, 'w') as f:
                for event_msg in events:
                    for event in event_msg.events:
                        x, y, p = event.x, event.y, int(event.polarity)
                        # 写入格式：timestamp polarity y x
                        if t:
                            f.write(f"{t.to_sec()} {x} {y} {p}\n")

            # 更新进度条
            pbar.update(1)

        pbar.close()  # 完成所有任务后关闭进度条


file_name = [
    'dt',
    'mn',
    'sr',
    'ss1',
    'ss2'
]

file_name_dict = {
    'dt':"dvs_vpr_2020-04-24-15-12-03",
    'mn':"dvs_vpr_2020-04-28-09-14-11",
    'sr':"dvs_vpr_2020-04-29-06-20-23",
    'ss1':"dvs_vpr_2020-04-21-17-03-03",
    'ss2':"dvs_vpr_2020-04-22-17-24-21"
}

for name in file_name:
    bag_file = '/root/autodl-tmp/bags/' + file_name_dict[name] + '.bag'
    timestamp_file = '/root/autodl-tmp/processed_data/' + name + '/timestamp.txt'
    output_dir = '/root/autodl-tmp/processed_data/' + name
    # 调用函数
    process_event_to_txt(bag_file, timestamp_file, output_dir)  # 这里设定了frame_interval，根据需要调整