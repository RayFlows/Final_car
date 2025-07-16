import sys

import os

# 添加项目根目录到 sys.path
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # 获取项目根目录
sys.path.insert(0, PROJECT_DIR)  # 将项目根目录添加到路径
# 如果需要可以添加 lib 和 models 目录
LIB_DIR = os.path.join(PROJECT_DIR, 'lib')
MODELS_DIR = os.path.join(PROJECT_DIR, 'lib', 'models')

sys.path.insert(0, LIB_DIR)  # 添加 lib 目录
sys.path.insert(0, MODELS_DIR)  # 添加 models 目录
import numpy as np
from types import SimpleNamespace
import torch
import cv2
from lib.models.pose_hrnet import get_pose_net
from core.inference import get_final_preds
from transforms import get_affine_transform
from mlp_classifier import MLPClassifier
from config import cfg, update_config

# 追加项目根目录到路径
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_DIR)














# 加载姿势估计模型
def load_pose_model(cfg, model_path, device=torch.device('cpu')):
    print("开始加载姿势估计模型...")
    model = get_pose_net(cfg, is_train=False)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print("姿势估计模型加载完成。")
    return model

# 加载训练好的 MLP 分类器模型
def load_mlp_model(model_path, input_size, hidden_size, output_size, device=torch.device('cpu')):
    print("开始加载 MLP 分类器模型...")
    model = MLPClassifier(input_size, hidden_size, output_size)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    print("MLP 分类器模型加载完成。")
    return model

# 图像预处理
def preprocess_image(image_path, cfg):
    print(f"开始预处理图像: {image_path}")
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
    print(f"图像预处理完成，图像形状: {inp.shape}")
    return torch.from_numpy(inp), center, scale, img

# 获取图像的关键点
def get_keypoints(image_path, model, cfg, device):
    print(f"开始提取关键点: {image_path}")
    inp_tensor, center, scale, orig_bgr = preprocess_image(image_path, cfg)
    inp_tensor = inp_tensor.to(device)

    with torch.no_grad():
        output = model(inp_tensor)
    preds, _ = get_final_preds(cfg, output.cpu().numpy(), np.array([center], dtype=np.float32), np.array([scale], dtype=np.float32))

    print(f"提取的关键点: {preds[0]}")
    return preds[0]

# 使用 MLP 分类器进行姿势分类
def classify_pose_with_mlp(keypoints, mlp_model, device):
    print(f"开始进行姿势分类，关键点: {keypoints}")
    keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32).to(device)
    keypoints_tensor = keypoints_tensor.flatten()  # 将 2D 数组展平为一维

    keypoints_tensor = keypoints_tensor.unsqueeze(0)

    with torch.no_grad():
        output = mlp_model(keypoints_tensor)
        _, predicted = torch.max(output, 1)

    print(f"姿势分类结果: {predicted.item()}")
    return predicted.item()

# 姿势预测函数
def predict_pose(image_path, pose_model, mlp_model, cfg, device):
    print(f"开始进行姿势预测: {image_path}")
    keypoints = get_keypoints(image_path, pose_model, cfg, device)
    keypoints_xy = [(float(x), float(y)) for x, y in keypoints]  # 将 keypoints 转换为原生 Python float 类型

    posture = classify_pose_with_mlp(keypoints_xy, mlp_model, device)

    posture_dict = {0: "Sitting", 1: "Supine", 2: "Prone", 3: "Standing"}

    print(f"预测的姿势标签: {posture_dict.get(posture, 'Unknown')}")
    return posture_dict.get(posture, 'Unknown')

# 提供一个函数用于加载配置并进行姿势预测
def run_pose_prediction(image_path, device=None):
    print("开始运行姿势预测...")
    pose_model_path = os.path.join(PROJECT_DIR, "models", "hrnet_fidip.pth")
    mlp_model_path = os.path.join(PROJECT_DIR, "classifier", "pose_classifier.pth")
    config_path = os.path.join(PROJECT_DIR, "experiments", "coco", "hrnet", "w48_384x288_adam_lr1e-3_custom.yaml")

    print(f"Pose model path: {pose_model_path}")
    print(f"MLP model path: {mlp_model_path}")
    print(f"Config path: {config_path}")

    args = SimpleNamespace(cfg=config_path, opts=[], modelDir="", logDir="", dataDir="", prevModelDir="")
    update_config(cfg, args)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pose_model = load_pose_model(cfg, pose_model_path, device)
    mlp_model = load_mlp_model(mlp_model_path, input_size=34, hidden_size=256, output_size=4, device=device) 

    posture = predict_pose(image_path, pose_model, mlp_model, cfg, device)

    return posture


def draw_pose(image):
    """
    在图像上绘制姿势关键点。

    :param image: 输入图像
    :return: 绘制了关键点的图像
    """
    # 获取项目根目录
    PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # 配置路径
    pose_model_path = os.path.join(PROJECT_DIR, "models", "hrnet_fidip.pth")
    config_path = os.path.join(PROJECT_DIR, "experiments", "coco", "hrnet", "w48_384x288_adam_lr1e-3_custom.yaml")

    # 加载配置
    args = SimpleNamespace(cfg=config_path, opts=[], modelDir="", logDir="", dataDir="", prevModelDir="")
    update_config(cfg, args)

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    pose_model = load_pose_model(cfg, pose_model_path, device)

    # 提取关键点
    keypoints = get_keypoints(image, pose_model, cfg, device)
    
    # 绘制关键点
    print("开始绘制姿势关键点...")
    for x, y in keypoints:
        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)  # 绘制绿色圆圈
    print("姿势关键点绘制完成。")
    
    return image
