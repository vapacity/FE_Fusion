import numpy as np

file_path = "/root/autodl-tmp/processed_data/dt/event/1587705126.649333.npy"

# 加载数据
data = np.load(file_path)

# 查看数据基本信息
print("数据形状:", data.shape)
print("数据类型:", data.dtype)

# 查看前几行数据
print("前几行数据:\n", data[:5])