import torch
import os
import numpy as np
from torch.utils.data import Dataset

class GestureDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.categories = os.listdir(data_dir)  # 获取文件夹中的所有类别
        self.data = []
        self.labels = []
        
        # 遍历每个类别文件夹
        for label, category in enumerate(self.categories):
            category_path = os.path.join(data_dir, category)
            for file_name in os.listdir(category_path):
                if file_name.endswith('.npy'):
                    file_path = os.path.join(category_path, file_name)
                    self.data.append(file_path)
                    self.labels.append(label)  # 每个文件对应的标签

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 读取数据
        data = np.load(self.data[idx])
        data = torch.tensor(data, dtype=torch.float32)  # 转换为Tensor
        
        # 数据重塑：将每个时间步的 (21, 3) 转换为一个长度为 63 的向量
        data = data.view(-1, 63)  # 将每个时间步展平为一个向量
        
        label = self.labels[idx]
        
        return data, label
