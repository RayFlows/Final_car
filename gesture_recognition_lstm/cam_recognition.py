import cv2
import numpy as np
import torch
import mediapipe as mp
from gesture_lstm import GestureLSTM  # 导入你定义的LSTM模型

# 初始化摄像头和模型
cap = cv2.VideoCapture(0)

# 初始化模型（确保你先定义了GestureLSTM模型结构）
model = GestureLSTM(input_size=63, hidden_size=128, num_classes=8, num_layers=2, dropout=0.5)

# 加载训练好的权重
model.load_state_dict(torch.load('final_model.pth'))  # 加载保存的模型权重
model.eval()  # 设置模型为评估模式

# 初始化MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 用来存储手部关键点数据
all_landmarks = []

# 映射标签到手势名称的字典
gesture_labels = {
    0: "Point down",
    1: "Point up",
    2: "Paml",
    3: "Point left",
    4: "Two",
    5: "None",
    6: "Fist",
    7: "Point right"
}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 转换为RGB格式并处理
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 提取关键点坐标
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            all_landmarks.append(landmarks)

    # 每30帧进行预测
    if len(all_landmarks) >= 30:
        # 取最近的30帧
        landmarks_array = np.array(all_landmarks[-30:])  # 形状: (30, 21, 3)

        # 转换为 (1, 30, 63) 形状: 1个样本，30个时间步，每个时间步有63个特征
        landmarks_array = landmarks_array.reshape(1, 30, 63)

        # 转换为Tensor并进行预测
        landmarks_tensor = torch.tensor(landmarks_array, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(landmarks_tensor)  # 现在可以正确调用模型进行预测
            _, predicted = torch.max(outputs, 1)
            predicted_label = predicted.item()

            # 获取执行度（模型输出的softmax概率）
            probabilities = torch.softmax(outputs, dim=1)
            execution_degree = probabilities[0, predicted_label].item() * 100  # 转换为百分比

        # 显示预测结果
        gesture_name = gesture_labels.get(predicted_label, "Unknown")
        cv2.putText(frame, f"Prediction: {gesture_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Execution Degree: {execution_degree:.2f}%", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示视频帧，不再镜像
    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # 按Esc退出
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
