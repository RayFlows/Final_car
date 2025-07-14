#camera_recevier.py

import cv2
import zmq
import base64
import numpy as np
import threading
from queue import Queue
from final_object_detection import autocontroller 

# 全局帧队列
frame_queue = Queue(maxsize=5)
last_frame = None
processed_frame = None

def run():
    """运行视频接收器"""
    global last_frame, processed_frame
    print("启动视频接收器...")

    # 创建自动控制器
    controller = autocontroller.AutoController()
    
    # 创建ZMQ对象
    context = zmq.Context()
    footage_socket = context.socket(zmq.PAIR)
    footage_socket.bind('tcp://*:5555')
    print("等待树莓派视频流连接...")
    
    try:
        while True:
            # 接收数据
            jpg_as_text = footage_socket.recv()
            
            # Base64解码
            img_data = base64.b64decode(jpg_as_text)
            
            # 转换为numpy数组
            npimg = np.frombuffer(img_data, dtype=np.uint8)
            
            # 解码图像
            frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # 保存原始帧
                last_frame = frame

                # 处理帧（目标检测）
                processed_frame = controller.process_frame(frame.copy())

                # 放入队列供模型处理
                if not frame_queue.full():
                    frame_queue.put(processed_frame)
                
                # # 保存最后一帧供显示
                # global last_frame
                # last_frame = frame
                
    except Exception as e:
        print(f"视频接收错误: {e}")
    finally:
        footage_socket.close()
        context.term()

# def get_frame():
#     """获取最新帧"""
#     global last_frame
#     return last_frame

# def get_frame_queue():
#     """获取帧队列"""
#     return frame_queue

def get_frame():
    """获取最新帧（带目标检测）"""
    global processed_frame
    return processed_frame if processed_frame is not None else last_frame

def get_frame_queue():
    """获取帧队列"""
    return frame_queue