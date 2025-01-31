import os
import numpy as np

def filt(directory_path, remaining_count):
    """
    筛选文件的函数:
    - 先删除所有全零的 .npy 文件
    - 如果剩余的文件数不够，按点数从小到大删除，直到剩余文件数量等于目标数量
    - 删除对应的 frame 和 event 文件
    """
    original_events_path = os.path.join(directory_path, "event")
    original_frames_path = os.path.join(directory_path, "frame")
    
    # 存储每个文件的点数
    file_counts = []
    
    # 遍历目录内的所有 .npy 文件并计算点数
    for filename in os.listdir(original_events_path):
        if filename.endswith(".npy"):
            file_path = os.path.join(original_events_path, filename)
            data = np.load(file_path)
            count_of_ones = np.sum(data != 0)
            
            # 如果文件不全零，存储文件名和点数
            if count_of_ones > 0:
                file_counts.append((filename, count_of_ones))
            else:
                # 如果文件全零，则删除该文件和对应的 frame
                event_path = os.path.join(original_events_path, filename)
                frame_path = os.path.join(original_frames_path, filename.replace(".npy", ".png"))
                
                if os.path.exists(event_path):
                    os.remove(event_path)
                    print(f"已删除全零事件文件：{event_path}")
                if os.path.exists(frame_path):
                    os.remove(frame_path)
                    print(f"已删除对应的 frame 文件：{frame_path}")

    # 如果文件数量已经达到目标数量，直接返回
    if len(file_counts) <= remaining_count:
        print("不需要筛选，目标数量已满足。")
        return

    # 按点数从小到大排序
    file_counts.sort(key=lambda x: x[1])

    # 从点数最少的开始筛除，直到剩下的个数等于目标
    while len(file_counts) > remaining_count:
        filename, _ = file_counts.pop(0)  # 删除点数最少的文件
        
        # 删除对应的 event 和 frame 文件
        event_path = os.path.join(original_events_path, filename)
        frame_path = os.path.join(original_frames_path, filename.replace(".npy", ".png"))
        
        if os.path.exists(event_path):
            os.remove(event_path)
            print(f"已删除：{event_path}")
        if os.path.exists(frame_path):
            os.remove(frame_path)
            print(f"已删除：{frame_path}")
    
