import os
import numpy as np

def get_max_lines_in_bin_npy(database_dirs):
    """
    获取 `database_dir/bin/` 目录下所有 `.txt` 文件的最大行数。
    如果目录不存在，则直接抛出 FileNotFoundError。
    """
    max_lines = 0
    for database_dir in database_dirs:
        bin_dir = os.path.join(database_dir, "bin")
        
        if not os.path.exists(bin_dir):
            raise FileNotFoundError(f"Error: Directory '{bin_dir}' does not exist!")

        for filename in os.listdir(bin_dir):
            if filename.endswith(".npy"):
                file_path = os.path.join(bin_dir, filename)
                event_volume = np.load(file_path)                    
                num_lines = len(event_volume)
                max_lines = max(max_lines, num_lines)  # 记录最大行数
    
    return max_lines

database_root = '/root/autodl-tmp/processed_data'
database_rel = 'dt mn sr ss1 ss2'
database_dirs = [os.path.join(database_root, d) for d in database_rel.split()]
print(get_max_lines_in_bin_npy(database_dirs))