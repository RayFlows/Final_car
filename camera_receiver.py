# #camera_recevier.py

# import cv2
# import zmq
# import base64
# import numpy as np
# import threading
# from queue import Queue
# from final_object_detection import autocontroller 

# # 全局帧队列
# frame_queue = Queue(maxsize=5)
# last_frame = None
# processed_frame = None

# def run():
#     """运行视频接收器"""
#     global last_frame, processed_frame
#     print("启动视频接收器...")

#     # 创建自动控制器
#     controller = autocontroller.AutoController()
    
#     # 创建ZMQ对象
#     context = zmq.Context()
#     footage_socket = context.socket(zmq.PAIR)
#     footage_socket.bind('tcp://*:5555')
#     print("等待树莓派视频流连接...")
    
#     try:
#         while True:
#             # 接收数据
#             jpg_as_text = footage_socket.recv()
            
#             # Base64解码
#             img_data = base64.b64decode(jpg_as_text)
            
#             # 转换为numpy数组
#             npimg = np.frombuffer(img_data, dtype=np.uint8)
            
#             # 解码图像
#             frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            
#             if frame is not None:
#                 # 保存原始帧
#                 last_frame = frame

#                 # 处理帧（目标检测）
#                 processed_frame = controller.process_frame(frame.copy())

#                 # 放入队列供模型处理
#                 if not frame_queue.full():
#                     frame_queue.put(processed_frame)
                
#                 # # 保存最后一帧供显示
#                 # global last_frame
#                 # last_frame = frame
                
#     except Exception as e:
#         print(f"视频接收错误: {e}")
#     finally:
#         footage_socket.close()
#         context.term()

# # def get_frame():
# #     """获取最新帧"""
# #     global last_frame
# #     return last_frame

# # def get_frame_queue():
# #     """获取帧队列"""
# #     return frame_queue

# def get_frame():
#     """获取最新帧（带目标检测）"""
#     global processed_frame
#     return processed_frame if processed_frame is not None else last_frame

# def get_frame_queue():
#     """获取帧队列"""
#     return frame_queue




# import cv2
# import zmq
# import base64
# import numpy as np
# import torch
# import os
# from queue import Queue
# from app import current_mode  # 引用app.py中的current_mode
# import sys
# from final_object_detection import autocontroller
# import requests

# # 获取当前脚本的路径
# PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# # 追加项目根目录到路径
# POSE_ESTIMATION_PATH = os.path.join(PROJECT_DIR, 'Infant-Pose-Estimation', 'tools')  # 确保此路径是正确的
# if POSE_ESTIMATION_PATH not in sys.path:
#     sys.path.insert(0, POSE_ESTIMATION_PATH)

# # 导入姿势预测相关函数
# from yolo_classifier_pipeline import predict_single_image  # 使用run_pose_prediction函数

# # 全局帧队列
# frame_queue = Queue(maxsize=5)
# last_frame = None
# processed_frame = None

# # 初始化姿势检测模型
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # 使用相对路径加载姿势检测模型
# pose_model_path = os.path.join(PROJECT_DIR, 'Infant-Pose-Estimation', 'models', 'hrnet_fidip.pth')

# # 定期获取当前模式的函数
# def get_current_mode():
#     try:
#         # 向 Flask 应用的 /get_mode 路由发起 GET 请求
#         response = requests.get('http://127.0.0.1:5000/get_mode')
#         if response.status_code == 200:
#             # 解析 JSON 响应中的 mode 字段
#             mode = response.json().get('mode')
#             print(f"成功获取当前模式: {mode}")  # 输出获取的模式
#             return mode
#         else:
#             print(f"无法获取模式，状态码: {response.status_code}")
#             return "cruise"  # 默认返回 "cruise"
#     except Exception as e:
#         print(f"获取模式时出错: {e}")
#         return "cruise"  # 如果请求失败，返回默认模式

# def run():
#     """运行视频接收器"""
#     global last_frame, processed_frame
#     print("启动视频接收器...")

#     # 创建自动控制器
#     controller = autocontroller.AutoController()

#     # 创建ZMQ对象
#     context = zmq.Context()
#     footage_socket = context.socket(zmq.PAIR)
#     footage_socket.bind('tcp://*:5555')
#     print("等待树莓派视频流连接...")

#     try:
#         while True:
#             # 获取当前模式
#             mode = get_current_mode()  # 获取最新的 current_mode
#             print(f"当前模式：{mode}")  # 输出当前模式（调试）

#             # 接收数据
#             jpg_as_text = footage_socket.recv()

#             # Base64解码
#             img_data = base64.b64decode(jpg_as_text)

#             # 转换为numpy数组
#             npimg = np.frombuffer(img_data, dtype=np.uint8)

#             # 解码图像
#             frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

#             # 检查是否成功解码
#             if frame is not None:
#                 # 保存原始帧
#                 last_frame = frame

#                 # 根据当前模式处理视频流
#                 if mode == "cruise":
#                     # 处理物品检测
#                     processed_frame = controller.process_frame(frame)  # 直接传递frame，而不是frame.copy()
#                     print(f"当前模式：{mode}，绘制红框")  # 输出调试信息
#                     cv2.rectangle(processed_frame, (50, 50), (400, 400), (0, 0, 255), 2)  # 红框 (BGR)
#                 elif mode == "pose":
#                     # 处理姿势检测
#                     posture = predict_single_image(frame, device=device)  # 直接传递frame作为图像对象
#                     print(f"当前模式：{mode}，姿势预测结果: {posture}")  # 打印出预测的姿势结果
#                     processed_frame = frame
#                     print(f"当前模式：{mode}，绘制绿框")  # 输出调试信息
#                     cv2.rectangle(processed_frame, (50, 50), (400, 400), (0, 255, 0), 2)  # 绿框 (BGR)

#                 # 放入队列供模型处理
#                 if not frame_queue.full():
#                     frame_queue.put(processed_frame)

#             else:
#                 print("图像解码失败，正在跳过该帧")

#     except Exception as e:
#         print(f"视频接收错误: {e}")
#     finally:
#         footage_socket.close()
#         context.term()

# def get_frame():
#     """获取最新帧（带目标检测或姿势检测）"""
#     global processed_frame
#     return processed_frame if processed_frame is not None else last_frame

# def get_frame_queue():
#     """获取帧队列"""
#     return frame_queue


import os
import cv2
import base64
import numpy as np
import torch
import time
import threading
import zmq
import sys
from queue import Queue
import requests
from final_object_detection import autocontroller


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..\Final_car'))
sys.path.insert(0, PROJECT_DIR)  # 将根目录添加到 sys.path 中
# print("insert project dir:", PROJECT_DIR)
# 调试：打印 sys.path，检查是否包含项目根目录
# print("当前的 sys.path:", sys.path)

# 确保包含 'Infant-Pose-Estimation' 根目录的路径
pose_estimation_path = os.path.join(PROJECT_DIR, 'Infant-Pose-Estimation')
pose_path = os.path.join(pose_estimation_path, 'tools')
sys.path.insert(0, pose_path)
# print("insert pose estimation path:", pose_path)

# 调试：检查 sys.path 是否包含 'Infant-Pose-Estimation'
# print("当前的 sys.path（包含 Infant-Pose-Estimation 路径检查）:", sys.path)
# 导入 pose_estimation.py 中的函数或类
from pose_estimation import run_pose_prediction , draw_pose
from extensions import socketio

frame_queue = Queue(maxsize=5)
last_frame = None
processed_frame = None

# 临时存储图像的路径
TEMP_IMAGE_PATH = "temp_frame.jpg"

# 初始化姿势检测模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载姿势检测模型
pose_model_path = os.path.join(PROJECT_DIR, 'Infant-Pose-Estimation', 'models', 'hrnet_fidip.pth')

# 定期获取当前模式的函数
def get_current_mode():
    try:
        # 向 Flask 应用的 /get_mode 路由发起 GET 请求
        response = requests.get('http://127.0.0.1:5000/get_mode')
        if response.status_code == 200:
            # 解析 JSON 响应中的 mode 字段
            mode = response.json().get('mode')
            print(f"成功获取当前模式: {mode}")  # 输出获取的模式
            return mode
        else:
            print(f"无法获取模式，状态码: {response.status_code}")
            return "cruise"  # 默认返回 "cruise"
    except Exception as e:
        print(f"获取模式时出错: {e}")
        return "cruise"  # 如果请求失败，返回默认模式





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
            # 获取当前模式
            mode = get_current_mode()  # 获取最新的 current_mode
            print(f"当前模式：{mode}")  # 输出当前模式（调试）

            # 接收数据
            jpg_as_text = footage_socket.recv()

            # Base64解码
            img_data = base64.b64decode(jpg_as_text)

            # 转换为numpy数组
            npimg = np.frombuffer(img_data, dtype=np.uint8)

            # 解码图像
            frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            # 检查是否成功解码
            if frame is not None:
                # 保存原始帧
                last_frame = frame

                if mode == "cruise":
                    # 处理物品检测
                    processed_frame = controller.process_frame(frame)  # 直接传递frame，而不是frame.copy()
                    print(f"当前模式：{mode}，绘制红框")  # 输出调试信息
                    cv2.rectangle(processed_frame, (50, 50), (400, 400), (0, 0, 255), 2)  # 红框 (BGR)
                
                elif mode == "pose":
                    # 将当前帧保存为临时 JPEG 文件
                    cv2.imwrite(TEMP_IMAGE_PATH, frame)

                    # 处理姿势检测
                    posture = run_pose_prediction(TEMP_IMAGE_PATH, device)  # 传递文件路径
                    print(f"当前模式：{mode}，姿势预测结果: {posture}")  # 打印出预测的姿势结果
                    # processed_frame = draw_pose(frame.copy())
                    socketio.emit('update_prediction', {'posture': posture})
                    print(f"当前模式：{mode}，绘制绿框")  # 输出调试信息
                    cv2.rectangle(processed_frame, (50, 50), (400, 400), (0, 255, 0), 2)  # 绿框 (BGR)
                    
                # 放入队列供模型处理
                if not frame_queue.full():
                    frame_queue.put(processed_frame)

            else:
                print("图像解码失败，正在跳过该帧")

    except Exception as e:
        print(f"视频接收错误: {e}")
    finally:
        footage_socket.close()
        context.term()

def get_frame():
    """获取最新帧（带目标检测或姿势检测）"""
    global processed_frame
    return processed_frame if processed_frame is not None else last_frame

def get_frame_queue():
    """获取帧队列"""
    return frame_queue
