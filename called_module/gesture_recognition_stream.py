import cv2
import torch
import os
import numpy as np
import mediapipe as mp
import time
import requests
import threading
from collections import deque
from gesture_recognition_lstm.gesture_lstm import GestureLSTM

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

current_directory = os.path.dirname(__file__)
model_path = os.path.join(current_directory, '../gesture_recognition_lstm/final_model.pth')
model = GestureLSTM(input_size=63, hidden_size=128, num_classes=8, num_layers=2, dropout=0.5)
model.load_state_dict(torch.load(model_path))
model.eval()

mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1  # 只检测一只手以提升性能
)
mp_drawing = mp.solutions.drawing_utils

gesture_labels = {
    0: "Point down",
    1: "One",
    2: "Palm",
    3: "Point left",
    4: "Two",
    5: "None",
    6: "Fist",
    7: "Point right"
}

all_landmarks = []
current_gesture = "None"
execution_degree = 0.0
frame_lock = threading.Lock()

def handle_gesture_switch(gesture_name):
    """发送手势识别结果到前端并切换模式"""
    # 判断手势并设置相应的模式
    if gesture_name == "One":
        mode = "cruise"  # 巡航模式
    elif gesture_name == "Two":
        mode = "pose"  # 姿势模式
    else:
        return

    # 发送模式切换的请求
    url = "http://localhost:5000/set_mode"
    payload = {"mode": mode}
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            print(f"模式切换成功：{mode}")
        else:
            print("模式切换失败", response.status_code)
    except Exception as e:
        print("请求失败:", e)

def process_frames():
    """在后台线程中处理手势识别"""
    global current_gesture, execution_degree
    
    while True:
        # 检查是否有足够的关键点进行处理
        if len(all_landmarks) < 30:
            time.sleep(0.05)
            continue
            
        with frame_lock:
            # 复制当前关键点以避免阻塞主线程
            landmarks_copy = list(all_landmarks)[-30:]
        
        landmarks_array = np.array(landmarks_copy)
        landmarks_array = landmarks_array.reshape(1, 30, 63)

        landmarks_tensor = torch.tensor(landmarks_array, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(landmarks_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_label = predicted.item()
            probabilities = torch.softmax(outputs, dim=1)
            execution_degree = probabilities[0, predicted_label].item() * 100

        gesture_name = gesture_labels.get(predicted_label, "Unknown")
        current_gesture = gesture_name
        
        # 处理手势切换
        handle_gesture_switch(gesture_name)
        
        # 降低处理频率
        time.sleep(0.1)

# 启动后台处理线程
processing_thread = threading.Thread(target=process_frames, daemon=True)
processing_thread.start()

# def gen_gesture_camera():
#     """生成手势识别视频流"""
#     last_time = time.time()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(rgb_frame)

#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#                 landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
#                 all_landmarks.append(landmarks)

#         if len(all_landmarks) >= 30:
#             landmarks_array = np.array(all_landmarks[-30:])
#             landmarks_array = landmarks_array.reshape(1, 30, 63)

#             landmarks_tensor = torch.tensor(landmarks_array, dtype=torch.float32)
#             with torch.no_grad():
#                 outputs = model(landmarks_tensor)
#                 _, predicted = torch.max(outputs, 1)
#                 predicted_label = predicted.item()
#                 probabilities = torch.softmax(outputs, dim=1)
#                 execution_degree = probabilities[0, predicted_label].item() * 100

#             gesture_name = gesture_labels.get(predicted_label, "Unknown")
#             cv2.putText(frame, f"Prediction: {gesture_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.putText(frame, f"Execution Degree: {execution_degree:.2f}%", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             handle_gesture_switch(gesture_name)
            
#         ret, jpeg = cv2.imencode('.jpg', frame)
#         frame_bytes = jpeg.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

#         time.sleep(0.05)

#     cap.release()
def gen_gesture_camera():
    """生成手势识别视频流"""
    last_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 降低处理分辨率
        small_frame = cv2.resize(frame, (320, 240))
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # 处理手势
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 在原始帧上绘制（不是缩小后的帧）
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                with frame_lock:
                    all_landmarks.append(landmarks)
        
        # 添加识别结果到帧
        cv2.putText(frame, f"Prediction: {current_gesture}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Execution Degree: {execution_degree:.2f}%", (50, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 编码帧
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
        
        # 控制帧率 (15-20 FPS)
        elapsed = time.time() - last_time
        sleep_time = max(0.05 - elapsed, 0)
        time.sleep(sleep_time)
        last_time = time.time()

# 注册释放资源的函数
import atexit
@atexit.register
def release_resources():
    cap.release()
    print("摄像头资源已释放")
