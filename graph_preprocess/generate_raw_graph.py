# -- coding: utf-8 --**
# convert downsample file to graph

import numpy as np
import os
import scipy.io as sio
import concurrent.futures
import argparse
import sys
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial import cKDTree
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Queue, Process


def process_file(base_dir, file):
    file_path = os.path.join(base_dir, file)
    # 读取数据，期望格式是4列：time, x, y, p
    try:
        # 实际输入是5列
        event_data = np.load(file_path)
        first_secs, first_nsecs = event_data[0, 0], event_data[0, 1]
        rel_time = (event_data[:, 0] - first_secs) * 1e6 + (event_data[:, 1] - first_nsecs) / 1e3
        data = np.stack((rel_time, event_data[:, 2], event_data[:, 3], event_data[:, 4]), axis=1).astype(np.float32)    # axis

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    if data.shape[0] < 1000:
        return None  # 点太少，不处理

    # 标准化时间戳（0 ~ 10）
    time_length = data[-1, 0]   # 最大时间戳
    eps = 0.001 + 1e-6  # 避免达到上界

    data[:, 0] = np.clip(data[:, 0] / time_length * 64, 0.0, 64.0 - eps)
    data[:, 1] = np.clip(data[:, 1] / 346 * 256, 0.0, 256.0 - eps)
    data[:, 2] = np.clip(data[:, 2] / 260 * 256, 0.0, 256.0 - eps)

    return {"points": data} # t x y p


# get polarity as the feature of the node
def extract_feature(data):
    data_size = data.shape[0]
    feature = np.zeros([data_size, 1])
    for i in range(data_size):
        if data[i, 3] == 1:
            feature[i, 0] = +1
        else:
            feature[i, 0] = -1
    return feature


def extract_position(data):
    data_size = data.shape[0]
    position = np.zeros([data_size, 3])
    for i in range(data_size):
        position[i, :] = data[i, 0:3]
    return position


def generate_graph(file, mat, target_path):
    # file = sio.loadmat(origin_path)
    data = mat["points"]
    feature = extract_feature(data)
    position = extract_position(data)
    # edges = calculate_edges(data, 5)
    # if the number of edges is 0 or less than 10, skip this sample
    # if edges.shape[1]<10:
    #     # view this file
    #     print(file+" : "+str(edges.shape[1]))
    #     return
    save_data = {"feature": feature, "pseudo": position}


    sio.savemat(target_path, save_data)
    return target_path

def process_and_generate(base_dir, file, target_dir):
    try:
        mat = process_file(base_dir, file)
        if mat is not None:
            generate_graph(file, mat, os.path.join(target_dir, file.replace(".npy", ".mat")))
        else:
            print(f"[SKIP] {file}: too few points or failed to process")
    except Exception as e:
        print(f"[ERROR] {file}: {e}")

def Worker(input_queue: Queue):
    while True:
        item = input_queue.get()
        if item == "STOP":
            break
        base_dir, file, target_dir = item
        try:
            process_and_generate(base_dir, file, target_dir)
        except Exception as e:
            print(f"Error: {e}")

def main():
    # dt  mn  sr  ss1  ss2
    dirs = ["ss2", "dt", "mn", "sr", "ss1"]
    for dir in dirs:
        base_dir = "/root/autodl-tmp/processed_data/"+dir+"/bin"
        target_dir = "/root/autodl-tmp/processed_data/"+dir+"/raw"
        os.makedirs(target_dir, exist_ok=True)
        MAX_WORKERS = 8
        task_queue = Queue(maxsize=MAX_WORKERS) 
        workers = []
        for _ in range(MAX_WORKERS):
            p = Process(target=Worker, args=(task_queue,))
            p.start()
            workers.append(p)

        for file in tqdm(os.listdir(base_dir)):
            task_queue.put((base_dir, file, target_dir))

        for worker in workers:
            task_queue.put("STOP")

        for worker in workers:
            worker.join()
        

        # # 串行版本
        # for file in tqdm(os.listdir(base_dir)):
        #      process_and_generate(base_dir, file, target_dir)



if __name__ == "__main__":
    main()
    print("genetate graph complete")