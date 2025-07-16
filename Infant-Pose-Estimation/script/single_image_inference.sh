#!/bin/bash

# 设置项目路径
PROJECT_DIR="/Users/ruiyuhan/Desktop/Advanced/Infant-Pose-Estimation"
MODEL_PATH="$PROJECT_DIR/models/hrnet_fidip.pth"
CONFIG_PATH="$PROJECT_DIR/experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml"
IMAGE_PATH="$1"  # 从命令行参数获取图片路径

# 确保图片路径提供
if [ -z "$IMAGE_PATH" ]; then
    echo "错误：请提供图片路径！"
    echo "用法：./run_single_inference.sh /path/to/your/image.jpg"
    exit 1
fi

# 确保模型文件和配置文件存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "模型文件不存在: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "配置文件不存在: $CONFIG_PATH"
    exit 1
fi

# 运行推理脚本
python $PROJECT_DIR/tools/inference_single_image.py \
    --cfg $CONFIG_PATH \
    TEST.MODEL_FILE $MODEL_PATH \
    TEST.USE_GT_BBOX True \
    IMAGE_PATH $IMAGE_PATH
