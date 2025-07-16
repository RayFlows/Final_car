import sys
import os
import torch
import numpy as np
import cv2
from types import SimpleNamespace
import json
from mlp_classifier import MLPClassifier
from ultralytics import YOLO
import pandas as pd
# 添加项目根目录到 sys.path
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # 获取项目根目录
sys.path.insert(0, PROJECT_DIR)  # 将项目根目录添加到路径

# 如果需要可以添加 lib 和 models 目录
LIB_DIR = os.path.join(PROJECT_DIR, 'lib')
MODELS_DIR = os.path.join(PROJECT_DIR, 'lib', 'models')

sys.path.insert(0, LIB_DIR)  # 添加 lib 目录
sys.path.insert(0, MODELS_DIR)  # 添加 models 目录

import _init_paths
from config import cfg
from config import update_config
from lib.models.pose_hrnet import get_pose_net  # 使用绝对路径导入
from core.inference import get_final_preds
from transforms import get_affine_transform

# 加载 HRNet 模型
def load_model(cfg, model_path, device=torch.device('cpu')):
    model = get_pose_net(cfg, is_train=False)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model

# 加载 MLP 分类器模型
def load_mlp_model(model_path, input_size, hidden_size, output_size, device=torch.device('cpu')):
    model = MLPClassifier(input_size, hidden_size, output_size)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

# 预处理图像
def preprocess(image_path, cfg):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    center = np.array([w / 2, h / 2], dtype=np.float32)
    aspect_ratio = cfg.MODEL.IMAGE_SIZE[0] / cfg.MODEL.IMAGE_SIZE[1]
    pixel_std = 200.0
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w / pixel_std, h / pixel_std], dtype=np.float32) * 1.25

    trans = get_affine_transform(center, scale, 0, cfg.MODEL.IMAGE_SIZE)
    inp = cv2.warpAffine(img_rgb, trans, (cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1]), flags=cv2.INTER_LINEAR)

    inp = inp.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    inp = (inp - mean) / std

    inp = inp.transpose(2, 0, 1)[None, ...]  # HWC -> NCHW
    return torch.from_numpy(inp), center, scale, img

# 获取关键点
def get_keypoints(image_path, model, cfg, device):
    inp_tensor, center, scale, orig_bgr = preprocess(image_path, cfg)
    inp_tensor = inp_tensor.to(device)

    with torch.no_grad():
        output = model(inp_tensor)
    preds, _ = get_final_preds(cfg, output.cpu().numpy(), np.array([center], dtype=np.float32), np.array([scale], dtype=np.float32))

    return preds[0]

# 姿势分类（基于 MLP 分类器）
def classify_pose_with_mlp(keypoints, mlp_model, device):
    keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32).to(device)
    keypoints_tensor = keypoints_tensor.flatten()  # 将 2D 数组展平为一维

    # 增加一个维度，使得输入的形状为 (1, input_size)，这样 BatchNorm1d 可以处理
    keypoints_tensor = keypoints_tensor.unsqueeze(0)

    # 使用训练好的 MLP 分类器进行预测
    with torch.no_grad():
        output = mlp_model(keypoints_tensor)
        _, predicted = torch.max(output, 1)
    
    # 返回预测的姿势类别
    return predicted.item()

# 对单张图片进行姿势预测
def predict_single_image(image_path, pose_model, mlp_model, cfg, device):
    # 提取图像关键点
    keypoints = get_keypoints(image_path, pose_model, cfg, device)
    keypoints_xy = [(float(x), float(y)) for x, y in keypoints]  # 将 keypoints 转换为原生 Python float 类型

    # 使用 MLP 分类器进行姿势分类
    posture = classify_pose_with_mlp(keypoints_xy, mlp_model, device)

    # 姿势标签（假设有四类：Sitting=0, Supine=1, Prone=2, Standing=3）
    posture_dict = {0: "Sitting", 1: "Supine", 2: "Prone", 3: "Standing"}

    print(f"Predicted Posture for {image_path}: {posture_dict.get(posture, 'Unknown')}")
    return posture_dict.get(posture, 'Unknown')

# 打开摄像头，使用 YOLO 检测婴儿并做预测
import cv2
import pandas as pd
from ultralytics import YOLO
import time

import threading

# 将检测和预测放在另一个线程中执行
def detect_and_predict_from_camera(pose_model, mlp_model, cfg, device):
    cap = cv2.VideoCapture(0)
    yolo_model = YOLO("../models/detect/train2/weights/last.pt")

    target_fps = 2
    frame_interval = 1.0 / target_fps
    last_process_time = time.time()

    # 用于存储检测结果
    detections = pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2', 'confidence', 'class'])

    def process_frame(frame):
        nonlocal detections, last_process_time
        current_time = time.time()

        if current_time - last_process_time >= frame_interval:
            # YOLO 推理
            results = yolo_model(frame)
            boxes = results[0].boxes.data.cpu().numpy()

            # 检测过滤
            detections = pd.DataFrame(boxes, columns=['x1', 'y1', 'x2', 'y2', 'confidence', 'class'])
            detections = detections[detections['confidence'] >= 0.7]
            last_process_time = current_time

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧")
            break

        # 通过线程异步处理帧
        threading.Thread(target=process_frame, args=(frame,)).start()

        # 绘制检测框和姿势标签
        for index, detection in detections.iterrows():
            x1, y1, x2, y2 = detection['x1'], detection['y1'], detection['x2'], detection['y2']
            conf = detection['confidence']
            baby_image = frame[int(y1):int(y2), int(x1):int(x2)]

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {conf:.2f}", (int(x1), int(y1)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            baby_image_path = "temp_baby_image.jpg"
            cv2.imwrite(baby_image_path, baby_image)

            posture = predict_single_image(baby_image_path, pose_model, mlp_model, cfg, device)
            cv2.putText(frame, f"Posture: {posture}", (int(x1), int(y2)+20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 加载配置
    args = SimpleNamespace(
        cfg="../experiments/coco/hrnet/w48_384x288_adam_lr1e-3_custom.yaml",
        opts=[],
        modelDir="", logDir="", dataDir="", prevModelDir=""
    )
    update_config(cfg, args)

    # 准备设备和姿势模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pose_model = load_model(cfg, "../models/hrnet_fidip.pth", device)

    # 加载训练好的 MLP 分类器模型
    mlp_model_path = "../classifier/pose_classifier.pth"  # 你训练好的 MLP 模型路径
    mlp_model = load_mlp_model(mlp_model_path, input_size=34, hidden_size=256, output_size=4, device=device)

    # 使用摄像头进行实时预测
    detect_and_predict_from_camera(pose_model, mlp_model, cfg, device)
