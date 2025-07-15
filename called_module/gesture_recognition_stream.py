import cv2
import torch
import os
import numpy as np
import mediapipe as mp
import time
import requests
from gesture_recognition_lstm.gesture_lstm import GestureLSTM

cap = cv2.VideoCapture(0)
current_directory = os.path.dirname(__file__)
model_path = os.path.join(current_directory, '../gesture_recognition_lstm/final_model.pth')
model = GestureLSTM(input_size=63, hidden_size=128, num_classes=8, num_layers=2, dropout=0.5)
model.load_state_dict(torch.load(model_path))
model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
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


def gen_gesture_camera():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                all_landmarks.append(landmarks)

        if len(all_landmarks) >= 30:
            landmarks_array = np.array(all_landmarks[-30:])
            landmarks_array = landmarks_array.reshape(1, 30, 63)

            landmarks_tensor = torch.tensor(landmarks_array, dtype=torch.float32)
            with torch.no_grad():
                outputs = model(landmarks_tensor)
                _, predicted = torch.max(outputs, 1)
                predicted_label = predicted.item()
                probabilities = torch.softmax(outputs, dim=1)
                execution_degree = probabilities[0, predicted_label].item() * 100

            gesture_name = gesture_labels.get(predicted_label, "Unknown")
            cv2.putText(frame, f"Prediction: {gesture_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Execution Degree: {execution_degree:.2f}%", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            handle_gesture_switch(gesture_name)
            
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

        time.sleep(0.05)

    cap.release()
