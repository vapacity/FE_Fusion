import os
import rosbag
import numpy as np
from pathlib import Path
from tqdm import tqdm
from .helpers import read_timestamp
import h5py
import torch
import torchvision.transforms as transforms

def normalize_event_volume(tensor):
    current_max = tensor.max()
    
    # 将张量归一化到最大值为 255 的范围
    if current_max > 0:  # 避免除以零
        # 计算缩放因子
        scale_factor = 255.0 / current_max
        # 缩放张量并转换为整型
        tensor = (tensor * scale_factor).floor()
    
    return tensor

def save_volume_to_image(tensor, save_path):
    """
    将 C×H×W 格式的张量转换为图片并保存
    
    参数:
        tensor: 形状为 [C, H, W] 的张量，通常 C=3 表示 RGB 图像
        save_path: 保存图片的路径
    """
    # 找出当前张量的最大值
    tensor = normalize_event_volume(tensor)
    # 如果张量的值范围不在 [0,1] 或 [0,255]，可能需要进行归一化
     # 使用 torchvision 的 ToPILImage 转换
    to_pil = transforms.ToPILImage()
    img = to_pil(tensor)
    # 保存图片
    img.save(save_path)
    # print(f"图片已保存至 {save_path}")

def process_event_to_volume_and_bin(bag_file, timestamps_file, volume_output_dir, bin_output_dir, time_tolerance=0.0125):
    timestamps = read_timestamp(timestamps_file)

    with rosbag.Bag(bag_file, 'r') as bag:
        it = bag.read_messages(topics=['/dvs/events'])
        topic, msg, t = next(it, (None, None, None))

        total_images = len(timestamps)
        pbar = tqdm(total=total_images, desc='Processing events')  # 初始化tqdm进度条

        # 检查输出目录是否存在
        if not os.path.exists(volume_output_dir):
            os.makedirs(volume_output_dir)
            print(f"{volume_output_dir} does not exist. Creating new directory.")
        
        if not os.path.exists(bin_output_dir):
            os.makedirs(bin_output_dir)
            print(f"{bin_output_dir} does not exist. Creating new directory.")

        for i, timestamp in enumerate(timestamps):
            events = []
            bin_output_list = []    # 用来存储bin格式所需要的全部事件列表
            # event_volume 用来存储事件帧
            event_volume = np.zeros((2, 260, 346))
            start_time = timestamp - time_tolerance
            end_time = timestamp + time_tolerance
            break_flag = False
            while t:
                topic, msg, t = next(it, (None, None, None))
                # 里面的event ts 肯定都是小于 t 的
                if t and t.to_sec() < start_time:
                    continue
                if msg:
                    for event in msg.events:
                        # 这里先对列表元素取最大最小值再作判断
                        if event.ts.to_sec() >= start_time:
                            if event.ts.to_sec() <= end_time:
                                events.append(event)
                            else:
                                break_flag = True
                                break
                if break_flag:
                    break

            for event in events:
                secs, nsecs, x, y, p = event.ts.secs, event.ts.nsecs, event.x, event.y, int(event.polarity) # p为0或1
                event_volume[p, y, x] += 1
                bin_output_list.append([secs, nsecs, x, y, p])

            timestamp_str = f"{timestamp}"  # 保留6位小数
            # 把event_volume 可视化为rgb灰度图
            event_volume_RGB = torch.cat((torch.from_numpy(event_volume), torch.zeros((1, 260, 346))), dim=0)
            save_volume_to_image(event_volume_RGB, os.path.join(volume_output_dir, f"{timestamp_str}.jpg"))
            # 将收集的事件保存为 .npy 文件，文件名使用时间戳
            output_file = os.path.join(volume_output_dir, f"{timestamp_str}.npy")
            np.save(output_file, event_volume)
            output_file = os.path.join(bin_output_dir, f"{timestamp_str}.npy")
            np.save(output_file, bin_output_list)

            # 释放内存
            del events
            del event_volume

            # 更新进度条
            pbar.update(1)

        pbar.close()  # 完成所有任务后关闭进度条




