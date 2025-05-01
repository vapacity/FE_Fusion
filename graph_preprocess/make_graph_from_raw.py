# -- coding: utf-8 --**
# the dataset class for EV-Gait-3DGraph model


import os
import numpy as np
import glob
import scipy.io as sio
import torch
import torch.utils.data
from torch_geometric.data import Data
from torch.utils.data import Dataset
import torch_geometric.transforms as T
import os.path as osp
from torch_geometric.nn.pool import radius_graph
from torch_geometric.transforms import FixedPoints
from tqdm import tqdm

def sub_sampling(data, n_samples=4096, sub_sample=True) -> Data:
    if sub_sample:
        sampler = FixedPoints(num=n_samples, allow_duplicates=False, replace=False)
        return sampler(data)
    else:
        sample_idx = np.arange(n_samples)
        for key, item in data:
            if torch.is_tensor(item) and item.size(0) != 1:
                data[key] = item[sample_idx]
        return data


class EV_Gait_3DGraph_Dataset(Dataset):
    def __init__(self, root_list, transform=None):
        if isinstance(root_list, str):
            root_list = [root_list]
        self.root_list = root_list
        self.transform = transform

        self._raw_paths = []
        self._processed_paths = []
        for root in root_list:
            raw_files = glob.glob(os.path.join(root, "raw", "*.mat"))
            self._raw_paths += raw_files
            self._processed_paths += [
                os.path.join(root, "processed", os.path.basename(f).replace(".mat", ".pt"))
                for f in raw_files
            ]
        # if not os.path.exists(self._processed_paths[0]):
        self.process()

    def __len__(self):
        return len(self._processed_paths)

    def __getitem__(self, idx):
        data = torch.load(self._processed_paths[idx])
        if self.transform:
            data = self.transform(data)
        return data

    def process(self):
        for idx, raw_path in enumerate(tqdm(self._raw_paths)):
            content = sio.loadmat(raw_path)
            feature = torch.tensor(content["feature"])[:, 0:1].float()
            pos = torch.tensor(np.array(content["pseudo"]), dtype=torch.float32)
            data = Data(x=feature, pos=pos)
            data = sub_sampling(data, n_samples=2048, sub_sample=True)
            data.edge_index = radius_graph(data.pos, r=10, max_num_neighbors=12)
            saved_name = os.path.basename(raw_path).replace(".mat", ".pt")
            processed_dir = raw_path.replace("raw", "processed").replace(os.path.basename(raw_path), "")
            os.makedirs(processed_dir, exist_ok=True)
            torch.save(data, os.path.join(processed_dir, saved_name))
    

if __name__ == "__main__":
    dataset_base = "/root/autodl-tmp/processed_data"
    dataset_dir = ["dt", "mn", "sr", "ss1", "ss2"]
    dataset = EV_Gait_3DGraph_Dataset([os.path.join(dataset_base, d) for d in dataset_dir])
    print(len(dataset))
    data = dataset[0]
    print(data)