# # cam_recognition.py
# import time
# import cv2
# import numpy as np
# from retinaface import Retinaface

# def main(cam_id: int = 0, view_size: tuple | None = None):
#     retinaface = Retinaface()

#     cap = cv2.VideoCapture(cam_id)
#     if not cap.isOpened():
#         raise RuntimeError(f"Can't open camera {cam_id}")

#     if view_size:
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH,  view_size[0])
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, view_size[1])

#     print("🔹 摄像头已打开，按 <Space> 进行人脸识别，Esc/q 退出。")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("⚠️  无法读取摄像头帧，退出。")
#             break

#         cv2.imshow("Camera Preview (press SPACE to detect)", frame)
#         key = cv2.waitKey(1) & 0xFF

#         if key in (27, ord('q')):  # Esc 或 q 退出
#             break

#         if key == 32:  # 空格键
#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#             # 你需要在 retinaface.detect_image 中支持 return_info
#             result_rgb, boxes, probs, names = retinaface.detect_image(rgb, return_info=True)

#             # ===== 修改点1：多人脸异常处理 =====
#             # 如果检测到多于1张人脸，则在视频中提示并直接返回 False
#             if len(boxes) > 1:
#                 # 在原始预览帧上写提示文字
#                 cv2.putText(frame,
#                             "Only one face permitted!",
#                             (30, 60),  # 文字位置
#                             cv2.FONT_HERSHEY_SIMPLEX, 
#                             1.2,         # 字体大小
#                             (0, 0, 255), # 红色
#                             3)           # 线宽
#                 # 重新展示加了提示的帧
#                 cv2.imshow("Camera Preview (press SPACE to detect)", frame)
#                 cv2.waitKey(1000)  # 停留1秒钟，让用户看到提示
#                 return False
#             # ===== 修改点1 结束 =====
            
#             result_bgr = cv2.cvtColor(np.array(result_rgb), cv2.COLOR_RGB2BGR)

#             # 绘制 bbox + 名字 + conf
#             for box, prob, name in zip(boxes, probs, names):
#                 box = [int(b) for b in box]
#                 cv2.rectangle(result_bgr, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
#                 label = f"{name} {prob:.2f}"
#                 cv2.putText(result_bgr, label, (box[0], box[1] - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#             cv2.imshow("Detection Result", result_bgr)
#             cv2.waitKey(1500)

#             if any(n != "Unknown" for n in names):
#                 unlocked = frame.copy()
#                 cv2.putText(unlocked, "Unlocked Successfully!", (40, 60),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
#                 cv2.imshow("Unlocked", unlocked)
#                 cv2.waitKey(1000)
#                 return True
#             return False

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main(cam_id=0, view_size=None)

import time
import cv2
import numpy as np
from .retinaface import Retinaface

# 全局变量，用于存储当前帧
current_frame = None

def main(cam_id: int = None, view_size: tuple | None = None):
    global current_frame
    retinaface = Retinaface()

    # 自动检测可用摄像头
    if cam_id is None:
        for i in range(0, 5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                print(f"使用摄像头索引: {i}")
                cam_id = i
                break
        else:
            raise RuntimeError("未找到可用的摄像头")
    else:
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {cam_id}")

    # 设置分辨率
    target_width = 640
    target_height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
    
    # 获取实际分辨率
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头分辨率: {actual_width}x{actual_height}")

    # 尝试自动检测人脸，无需按空格
    start_time = time.time()
    detection_interval = 3  # 每3秒尝试一次检测
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 无法读取摄像头帧，尝试重新打开...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print("❌ 无法重新打开摄像头")
                break
            continue
        
        frame_count += 1
        
        # 更新全局帧变量
        current_frame = frame.copy()
        
        # 自动检测（每隔一定时间）
        current_time = time.time()
        if current_time - start_time > detection_interval:
            print("尝试人脸检测...")
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 检测人脸
                result_rgb, boxes, probs, names = retinaface.detect_image(rgb, return_info=True)
                
                # 多人脸异常处理
                if len(boxes) > 1:
                    print("检测到多个人脸")
                    cv2.putText(frame, "Only one face permitted!", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                # 绘制结果
                result_bgr = cv2.cvtColor(np.array(result_rgb), cv2.COLOR_RGB2BGR)
                for box, prob, name in zip(boxes, probs, names):
                    box = [int(b) for b in box]
                    cv2.rectangle(result_bgr, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    label = f"{name} {prob:.2f}"
                    cv2.putText(result_bgr, label, (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # 更新全局帧为检测结果
                current_frame = result_bgr
                
                # 检查是否识别到已知人脸
                if any(n != "Unknown" for n in names):
                    print("识别到已知人脸!")
                    unlocked = frame.copy()
                    cv2.putText(unlocked, "Unlocked Successfully!", (40, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    current_frame = unlocked
                    cap.release()
                    return True
                
            except Exception as e:
                print(f"人脸检测错误: {e}")
            
            start_time = current_time  # 重置计时器

        # 检查退出条件（不再等待按键）
        # 如果需要退出，应该由主线程控制

    cap.release()
    return False

def get_current_frame():
    """获取当前帧"""
    global current_frame
    return current_frame