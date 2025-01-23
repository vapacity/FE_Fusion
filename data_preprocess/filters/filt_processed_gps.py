import os

# 设置包含 .npy 文件的目录路径
npy_directory = "/root/autodl-tmp/processed_data/dvs_vpr_2020-04-28-09-14-11/event"  # 替换为你的 .npy 文件目录路径

# 获取目录中所有的 .npy 文件名，并提取时间戳
timestamps = set()
for filename in os.listdir(npy_directory):
    if filename.endswith(".npy"):
        timestamp_str = filename.replace(".npy", "")  # 去除 .npy 后缀
        timestamps.add(float(timestamp_str))  # 将时间戳存入集合


# 读取第一个文件并筛选符合条件的内容
interpolated_gps_file = "/root/autodl-tmp/processed_data/dvs_vpr_2020-04-28-09-14-11/interpolated_gps.txt"  # 替换为第一个文件的路径
output_file = "/root/autodl-tmp/processed_data/dvs_vpr_2020-04-28-09-14-11/filtered_interpolated_gps.txt"
count = 0
with open(interpolated_gps_file, "r") as file, open(output_file, "w") as output:
    for line in file:
        parts = line.strip().split()
        if len(parts) == 3:
            timestamp = float(parts[2])  # 获取时间戳部分
            if timestamp in timestamps:  # 检查时间戳是否在集合中
                count+=1
                output.write(line)  # 写入筛选后的内容
print(count)
print(f"筛选完成，结果已保存到 {output_file}")
