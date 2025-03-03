import os
import shutil
import numpy as np

# 设置目录路径
time_dir_lst = ['dt','mn','sr','ss1','ss2']

def process(time_dir):
    from_dir = os.path.join("/root/autodl-tmp/processed_data", time_dir)
    old_bin_path = os.path.join(from_dir, "full_bin")
    print(old_bin_path)
    output_dir = os.path.join("/root/autodl-tmp/processed_data", time_dir)
    new_bin_path = os.path.join(output_dir, "bin")
    new_npy_path = os.path.join(output_dir, "event")

    os.makedirs(new_bin_path, exist_ok=True)

    # 存储每个文件的点数
    new_filename_list = []

    # 遍历目录内的所有 .npy 文件
    for filename in os.listdir(new_npy_path):
        if filename.endswith(".npy"):
            new_filename_list.append(filename)

    print("len(new_filename_list)")
    print(len(new_filename_list))
    # breakpoint()

    for filename in os.listdir(old_bin_path):
        if filename.endswith(".npy"):
            old_filename = filename.replace("bin_", "")
            print(old_filename)
            if old_filename in new_filename_list:
                old_bin_file = os.path.join(old_bin_path, filename)
                new_bin_file = os.path.join(new_bin_path, filename)
                shutil.copy(old_bin_file, new_bin_file)
                print("new_bin_file")
                print(new_bin_file)


if __name__ == "__main__":
    for time_dir in time_dir_lst:
        process(time_dir)
