import torch
import torch.nn as nn

# LSTM模型定义
class GestureLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_classes=8, num_layers=2, dropout=0.5):
        super(GestureLSTM, self).__init__()

        # 单向LSTM层（因为实时监测需要）
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)

        # Batch Normalization 层
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 全连接层
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # LSTM 输出
        out, _ = self.lstm(x)

        # 获取最后一个时间步的输出
        out = out[:, -1, :]  # (batch_size, hidden_size)

        # Batch Normalization 和 Dropout
        out = self.batch_norm(out)
        out = self.dropout(out)

        # 全连接层进行分类
        out = self.fc(out)
        return out
