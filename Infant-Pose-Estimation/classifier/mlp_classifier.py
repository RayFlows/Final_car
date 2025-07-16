import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# 定义改进后的多层感知器模型
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.3):
        super(MLPClassifier, self).__init__()
        
        # 添加更多隐藏层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)  # 增加一个额外的隐藏层
        
        # BatchNorm 层
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)  # 对第四层也进行 BatchNorm
        
        # 激活函数和其他
        self.relu = nn.LeakyReLU(negative_slope=0.01)  # 使用 LeakyReLU
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=1)
        
        # 输出层
        self.fc_out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # 第一层：FC + BatchNorm + LeakyReLU + Dropout
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        # 第二层：FC + BatchNorm + LeakyReLU + Dropout
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        # 第三层：FC + BatchNorm + LeakyReLU + Dropout
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)

        # 第四层：FC + BatchNorm + LeakyReLU + Dropout
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)

        # 输出层
        x = self.fc_out(x)
        return self.softmax(x)

# 训练模型函数
def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize

            running_loss += loss.item()

        # 调整学习率
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

# 评估模型函数
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy}%')
    return accuracy
