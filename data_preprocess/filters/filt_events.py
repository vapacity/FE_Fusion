import os
import shutil
import numpy as np

# 设置目录路径
directory_path = "/root/autodl-tmp/processed_data/dvs_vpr_2020-04-24-15-12-03"  # 替换为你的目录路径
output_file = directory_path+"output_counts_sorted.txt"
original_events_path = directory_path+"/event_old"
original_frames_path = directory_path+"/frame_old"

# 存储每个文件的点数
file_counts = []

# 遍历目录内的所有 .npy 文件并计算点数
for filename in os.listdir(original_events_path):
    if filename.endswith(".npy"):
        file_path = os.path.join(original_events_path, filename)
        # 加载 .npy 文件
        data = np.load(file_path)
        # 计算点为 1 的个数
        count_of_ones = np.sum(data != 0)
        max_value = np.max(data)
        # 存储文件名和计数
        file_counts.append((filename, count_of_ones))

# 按点数从多到少排序
file_counts.sort(key=lambda x: x[1], reverse=True)

# 写入排序后的结果
with open(output_file, "w") as file:
    for filename, count in file_counts:
        file.write(f"{filename} {count}\n")

print("文件已排序并保存到", output_file)

# 要求用户输入阈值
remaining_count = int(input("请输入保留的文件数量: "))
new_events_path = directory_path+f"/event"
new_frames_path = directory_path+f"/frame"

# 从点数最少的开始筛除，直到剩下的个数等于要求
file_counts = sorted(file_counts, key=lambda x: x[1])  # 按点数从小到大排序
while len(file_counts) > remaining_count:
    file_counts.pop(0)  # 删除点数最少的文件

# 创建新文件夹以保存筛选后的文件
if not os.path.exists(new_events_path):
    os.makedirs(new_events_path)

if not os.path.exists(new_frames_path):
    os.makedirs(new_frames_path)

# 复制筛选后的文件到新文件夹
for filename, count in file_counts:
    original_event_path = os.path.join(original_events_path, filename)
    new_event_path = os.path.join(new_events_path, filename)
    shutil.copy(original_event_path, new_event_path)
for filename,count in file_counts:
    original_frame_path = os.path.join(original_frames_path,filename.replace(".npy",".png"))
    new_frame_path = os.path.join(new_frames_path,filename.replace(".npy",".png"))
    shutil.copy(original_frame_path,new_frame_path)

# 输出筛选后的结果
final_output_file = "filtered_counts.txt"
with open(final_output_file, "w") as file:
    for filename, count in file_counts:
        file.write(f"{filename} {count}\n") 

print("筛选完成")
