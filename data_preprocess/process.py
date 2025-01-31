import os
from process_tools.process_event import process_event
from process_tools.process_frame import process_frame
from process_tools.process_gps import process_all_gps
from process_tools.process_triplet import find_triplet_samples
from process_tools.process_interpolated_gps import interpolate_gps_data
from process_tools.process_timestamp import get_index
from process_tools.filt_events import filt
dataset = {
    'ss1': 'dvs_vpr_2020-04-21-17-03-03',
    'ss2': 'dvs_vpr_2020-04-22-17-24-21',
    'dt' : 'dvs_vpr_2020-04-24-15-12-03',
    'mn' : 'dvs_vpr_2020-04-28-09-14-11',
    'sr' : 'dvs_vpr_2020-04-29-06-20-23'
}
remaining_count = {
    'ss1': 2181,
    'ss2': 1768,
    'dt' : 2234,
    'mn' : 2478,
    'sr' : 2620
}
input_file_path = '/root/autodl-fs/Brizbane_dataset/'
output_file_path = '/root/autodl-tmp/processed_data/'

# 对每个数据集的frame进行预处理
print("processing frame")
for key, name in dataset.items():
    bag_file = f"{input_file_path}{name}.bag"
    output_dir = f"{output_file_path}{key}/frame/"
    if not os.path.exists(output_dir):  # 检查路径是否存在
        os.makedirs(output_dir, exist_ok=True)  # 如果不存在则创建
        process_frame(bag_file, output_dir, frame_interval=0.25)  # 可以根据需要调整frame_interval
    else:
        print(f"Frame processing for {key} already done. Skipping...")

# 对每个数据集的timestamp进行预处理（提取时间戳）
print("processing timestamp")
for key, name in dataset.items():
    frame_path = f"{output_file_path}{key}/frame/"
    output_file = f"{output_file_path}{key}/timestamp.txt"
    if not os.path.exists(output_file):  # 检查时间戳文件是否已经生成
        get_index(frame_path, output_file)
    else:
        print(f"Timestamp for {key} already generated. Skipping...")

# 对每个数据集的event进行预处理（基于时间戳生成事件数据）
print("processing events")
for key, name in dataset.items():
    bag_file = f"{input_file_path}{name}.bag"
    timestamp_file = f"{output_file_path}{key}/timestamp.txt"
    output_dir = f"{output_file_path}{key}/event/"
    if not os.path.exists(output_dir):  # 检查事件数据目录是否存在
        os.makedirs(output_dir, exist_ok=True)
        process_event(bag_file, timestamp_file, output_dir)
    else:
        print(f"Event processing for {key} already done. Skipping...")

print("filtering events and frames")
for key,name in dataset.items():
    filt(f"{output_file_path}{key}",remaining_count[key])
    
# 对每个数据集的timestamp进行预处理（提取时间戳）
print("processing timestamp")
for key, name in dataset.items():
    frame_path = f"{output_file_path}{key}/frame/"
    output_file = f"{output_file_path}{key}/timestamp.txt"
    get_index(frame_path, output_file)
        
# GPS信息的处理，假设process_gps()会处理所有的GPS相关任务
print("processing gps")
process_all_gps()

# 基于时间戳对GPS数据进行插值
print("interpolating gps")
for key, name in dataset.items():
    timestamp_file = f"{output_file_path}{key}/timestamp.txt"
    gps_file = f"{output_file_path}{key}/gps.txt"
    interpolated_gps_file = f"{output_file_path}{key}/interpolated_gps.txt"
    if not os.path.exists(interpolated_gps_file):  # 检查插值文件是否已存在
        interpolate_gps_data(timestamp_file, gps_file, interpolated_gps_file)
    else:
        print(f"Interpolated GPS for {key} already exists. Skipping...")

# 查找三元组用于训练
query_gps_paths = [f"{output_file_path}{'sr'}/interpolated_gps.txt", # experiment1 train
                   f"{output_file_path}{'mn'}/interpolated_gps.txt", # experiment2 train
                   f"{output_file_path}{'sr'}/interpolated_gps.txt", # experiment3 train
                   f"{output_file_path}{'sr'}/interpolated_gps.txt"  # experiment4 train
                  ]

database_gps_paths = [
    [f"{output_file_path}{'dt'}/interpolated_gps.txt", f"{output_file_path}{'mn'}/interpolated_gps.txt"],  # experiment1 train
    [f"{output_file_path}{'ss2'}/interpolated_gps.txt", f"{output_file_path}{'dt'}/interpolated_gps.txt"],  # experiment2 train
    [f"{output_file_path}{'ss2'}/interpolated_gps.txt", f"{output_file_path}{'dt'}/interpolated_gps.txt"],  # experiment3 train
    [f"{output_file_path}{'ss2'}/interpolated_gps.txt", f"{output_file_path}{'mn'}/interpolated_gps.txt"]   # experiment4 train
]

# 根据正负样本的条件查找三元组,挑选最远的正样本和最近的负样本
for idx, (query_gps_path, db_paths) in enumerate(zip(query_gps_paths, database_gps_paths), start=1):
    # 根据索引动态生成实验保存路径
    save_dir = f"{output_file_path}triplets/experiment_{idx}/"  # 保存的目录路径
    save_file = f"{save_dir}triplet_result.txt"  # 保存的文件路径
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)  # 如果不存在则创建
        
    # 检查文件是否已经存在，如果不存在则进行处理
    if not os.path.exists(save_file):  # 检查文件是否已经存在
        # 调用 find_triplet_samples 函数，传入查询GPS路径和数据库路径，保存三元组
        find_triplet_samples(query_gps_path, db_paths, save_file)
    else:
        print(f"Triplet processing for experiment {idx} already done. Skipping...")
