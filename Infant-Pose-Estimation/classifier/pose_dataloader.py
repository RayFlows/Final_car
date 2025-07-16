import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PoseDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file, 'r') as f:
            self.data = json.load(f)["images"]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        keypoints = np.array(item["keypoints"]).flatten()  # 转为一维数组，方便输入
        posture = item["posture"]

        # 转换标签为数字（例如："Sitting" -> 0, "Prone" -> 1 等）
        posture_dict = {"Sitting": 0, "Supine": 1, "Prone": 2, "Standing": 3}
        # prone（俯卧）是危险姿势，supine（仰卧）
        label = posture_dict.get(posture, -1)  # 默认-1表示未知类
        
        if self.transform:
            keypoints = self.transform(keypoints)

        return torch.tensor(keypoints, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# 数据集路径
input_json_path = "../data/SyRIP_Posture/annotations/train600/processed_train_data.json"
# 创建数据加载器
train_dataset = PoseDataset(input_json_path)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
