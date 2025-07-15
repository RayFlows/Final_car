import cv2
import os
import numpy as np
import mediapipe as mp

# 设置手势名称（每个手势都有一个文件夹）
gesture_name = "fist"

# 创建对应手势的文件夹（如果不存在）
dataset_dir = "gesture_dataset"
gesture_dir = os.path.join(dataset_dir, gesture_name)
os.makedirs(gesture_dir, exist_ok=True)

# 设置视频捕捉
cap = cv2.VideoCapture(0)  # 使用摄像头

# 初始化MediaPipe手部模型
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 用来存储提取的手部关键点数据
all_landmarks = []
frame_counter = 0

print(f"正在录制'{gesture_name}'手势数据，按空格开始录制，按Esc退出。")

# 录制过程
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为RGB格式
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # 如果检测到手部关键点，绘制骨架
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 提取手部关键点的 (x, y, z) 坐标
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            all_landmarks.append(landmarks)  # 保存所有帧的关键点数据

    # 显示视频帧（关闭镜像效果）
    frame = cv2.flip(frame, 1)  # 翻转图像，确保视频不镜像
    cv2.imshow("Recording", frame)

    # 按空格开始录制数据
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # 空格键开始录制
        print("开始录制数据...")

        # 继续读取视频帧并提取关键点，直到按下Esc
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 转换为RGB格式
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # 如果检测到手部关键点，绘制骨架
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # 提取手部关键点的 (x, y, z) 坐标
                    landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                    all_landmarks.append(landmarks)  # 保存所有帧的关键点数据

            frame = cv2.flip(frame, 1)  # 确保每帧视频都不镜像
            cv2.imshow("Recording", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # 按下Esc键退出录制
                print("录制结束。")
                break

        # 保存录制的数据
        landmarks_array = np.array(all_landmarks)          # 形状: (num_frames, 21, 3)

        # === 计算序号并生成文件名 =========================
        existing_files = [f for f in os.listdir(gesture_dir) if f.endswith('_landmarks.npy')]
        file_index = len(existing_files) + 1               # 序号从 1 开始递增
        save_name = f"{gesture_name}_{file_index}_landmarks.npy"
        save_path = os.path.join(gesture_dir, save_name)
        # =================================================

        np.save(save_path, landmarks_array)
        print(f"数据保存成功！{save_name}")

        all_landmarks = []     # 重置数据，以准备下一个手势

    # 按Esc键退出
    if key == 27:
        print("退出录制。")
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
