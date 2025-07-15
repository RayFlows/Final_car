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
    center = np.array([w/2, h/2], dtype=np.float32)
    aspect_ratio = cfg.MODEL.IMAGE_SIZE[0] / cfg.MODEL.IMAGE_SIZE[1]
    pixel_std = 200.0
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w/pixel_std, h/pixel_std], dtype=np.float32) * 1.25

    trans = get_affine_transform(center, scale, 0, cfg.MODEL.IMAGE_SIZE)
    inp = cv2.warpAffine(img_rgb, trans,
                         (cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1]),
                         flags=cv2.INTER_LINEAR)

    inp = inp.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    inp = (inp - mean) / std

    inp = inp.transpose(2,0,1)[None, ...]  # HWC -> NCHW
    return torch.from_numpy(inp), center, scale, img  # 返回图像和中心点、缩放因子

def draw_results(orig_img, keypoints, save_path=None):
    """在图片上画出关键点."""
    img = orig_img.copy()
    for x, y in keypoints:
        cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
    
    # 可视化并保存图像
    cv2.imshow("Keypoints", img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if save_path:
        cv2.imwrite(save_path, img)

def predict_single_image(image_path, model, cfg, device):
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

    keypoints = preds[0]  # 17个关键点
    print("Keypoints (x, y):")
    for i, (x, y) in enumerate(keypoints):
        print(f"{i:2d}: ({x:6.1f}, {y:6.1f})")

    # 可视化
    save_path = os.path.splitext(image_path)[0] + "_result.jpg"
    draw_results(orig_bgr, keypoints, save_path)
    print(f"Result saved to: {save_path}")


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

    # 预测单张图像
    image_path = "../data/images_to_test/test100.jpg"  # 修改为你的图片路径
    predict_single_image(image_path, model, cfg, device)
