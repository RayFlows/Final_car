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

# 现在可以正确导入模块了
import torch
import numpy as np
import cv2
from types import SimpleNamespace

import _init_paths
from config import cfg
from config import update_config
from lib.models.pose_hrnet import get_pose_net  # 使用绝对路径导入
from core.inference import get_final_preds
from transforms import get_affine_transform
import json

def load_model(cfg, model_path, device=torch.device('cpu')):
    """加载预训练模型，map 到 CPU/GPU."""
    model = get_pose_net(cfg, is_train=False)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def preprocess(image_path, cfg):
    """读取图像 → Affine 变换 → 归一化 → 转 Tensor."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # 计算 center 和 scale（与训练脚本一致）
    center = np.array([w / 2, h / 2], dtype=np.float32)
    aspect_ratio = cfg.MODEL.IMAGE_SIZE[0] / cfg.MODEL.IMAGE_SIZE[1]
    pixel_std = 200.0
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w / pixel_std, h / pixel_std], dtype=np.float32) * 1.25

    trans = get_affine_transform(center, scale, 0, cfg.MODEL.IMAGE_SIZE)
    inp = cv2.warpAffine(img_rgb, trans,
                         (cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1]),
                         flags=cv2.INTER_LINEAR)

    inp = inp.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    inp = (inp - mean) / std

    inp = inp.transpose(2, 0, 1)[None, ...]  # HWC -> NCHW
    return torch.from_numpy(inp), center, scale, img  # 返回图像和中心点、缩放因子


def get_keypoints(image_path, model, cfg, device):
    """对单张图片进行预测，输出关键点"""
    inp_tensor, center, scale, orig_bgr = preprocess(image_path, cfg)
    inp_tensor = inp_tensor.to(device)

    # 前向推理
    with torch.no_grad():
        output = model(inp_tensor)
    preds, _ = get_final_preds(
        cfg,
        output.cpu().numpy(),
        np.array([center], dtype=np.float32),
        np.array([scale], dtype=np.float32)
    )

    return preds[0]  # 返回17个关键点

def default_converter(o):
    """转换 numpy 类型到原生 Python 类型"""
    if isinstance(o, np.generic):
        return o.item()  # 转换为原生 Python 类型
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

def process_json(input_json_path, output_json_path, model, cfg, device):
    """处理 JSON 文件，提取关键点并保存到新 JSON 文件"""
    # 读取原始 JSON 文件
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    output_data = []

    for item in data['images']:
        # 拼接文件路径
        image_path = os.path.join("../data/SyRIP_Posture/images/validate100", item['file_name'])
        print(f"Processing {image_path}...")
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        # 提取图像的特征点
        keypoints = get_keypoints(image_path, model, cfg, device)
        keypoints_xy = [(float(x), float(y)) for x, y in keypoints]  # 将 keypoints 转换为原生 Python float 类型

        # 将每条数据更新到新的字典中
        new_item = {
            "file_name": item["file_name"],
            "image_path": image_path,
            "posture": item["posture"],  # 直接复制原来的姿势
            "keypoints": keypoints_xy
        }

        output_data.append(new_item)

    # 将数据保存到新 JSON 文件时，使用 default 函数
    with open(output_json_path, 'w') as f:
        json.dump({"images": output_data}, f, indent=4, default=default_converter)

    print(f"Results saved to {output_json_path}")
    print(f"Processed {len(output_data)} images.")


if __name__ == "__main__":
    # 加载配置
    args = SimpleNamespace(
        cfg="../experiments/coco/hrnet/w48_384x288_adam_lr1e-3_custom.yaml",
        opts=[],
        modelDir="", logDir="", dataDir="", prevModelDir=""
    )
    update_config(cfg, args)

    # 准备设备和模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, "../models/hrnet_fidip.pth", device)

    # 输入 JSON 文件路径
    # input_json_path = "../data/syrip/annotations/person_keypoints_validate_infant.json"  # 你的输入 JSON 文件路径
    # output_json_path = "../data/syrip/annotations/processed_val_data.json"  # 输出的新的 JSON 文件路径
    input_json_path = "../data/SyRIP_Posture/annotations/validate100/person_keypoints_validate_infant.json"  # 你的输入 JSON 文件路径
    output_json_path = "../data/SyRIP_Posture/annotations/validate100/processed_validate_data.json"  # 输出的新的 JSON 文件路径
    # 处理数据并保存新的 JSON 文件
    process_json(input_json_path, output_json_path, model, cfg, device)
