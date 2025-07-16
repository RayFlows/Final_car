# final_object_detection/auto_controller.py
import cv2
import numpy as np
import time
import socket
from ultralytics import YOLO

class AutoController:
    def __init__(self):
        print("初始化自动控制器...")
        # 加载目标检测模型
        self.model = YOLO("final_object_detection/detect 2/train2/weights/last.pt")
        # self.model = YOLO("final_object_detection/detect/train\weights/last.pt")
        self.model.fuse()
        
        # 目标区域配置
        self.TARGET_REGION = [
            (300, 220),
            (300, 260),
            (340, 260),
            (340, 220)
        ]
        
        # 计算目标区域参数
        self.x_min = min(p[0] for p in self.TARGET_REGION)
        self.x_max = max(p[0] for p in self.TARGET_REGION)
        self.y_min = min(p[1] for p in self.TARGET_REGION)
        self.y_max = max(p[1] for p in self.TARGET_REGION)
        self.target_center_x = (self.x_min + self.x_max) // 2
        self.target_center_y = (self.y_min + self.y_max) // 2
        
        # 创建左右调整区域（比目标区域宽）
        self.ADJUST_X_MIN = self.x_min - 50
        self.ADJUST_X_MAX = self.x_max + 50
        
        # 小车控制配置
        self.RASPBERRY_IP = "frp-fit.com"
        self.CONTROL_PORT = 26669
        self.control_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.control_sock.settimeout(0.1)
        
        # 控制状态
        self.last_control_time = 0
        self.control_interval = 0.5  # 控制间隔
        self.current_command = None
        self.command_start_time = 0
        self.command_duration = 0
        self.last_adjustment = None

        # 抓取状态
        self.grab_command_sent = False  # 是否已发送抓取命令
        self.grab_target_class = None   # 抓取目标的类别

        # 类别到抓取命令的映射
        self.CLASS_TO_COMMAND = {
            "double skin milk pudding": "1",
            "porridge": "1",
            "water bottle": "2",
            "pacifier": "2",
            "green ball": "3",
            "purple ball": "3",
            "shaker": "3",
            "pig": "4",
            "bear": "4"
        }
        
        print("自动控制器初始化完成")
    
    def is_in_vertical_range(self, cy):
        """检查垂直方向是否在可接受范围内"""
        return self.y_min <= cy <= self.y_max
    
    def is_in_horizontal_range(self, cx):
        """检查水平方向是否在可接受范围内"""
        return self.ADJUST_X_MIN <= cx <= self.ADJUST_X_MAX
    
    def decide_move(self, cx, cy):
        """根据目标位置决定移动方向"""
        # 首先检查是否在目标区域内
        if self.x_min <= cx <= self.x_max and self.y_min <= cy <= self.y_max:
            return "in position"
        
        # 优先调整左右方向（如果水平位置不在可接受范围内）
        if not self.is_in_horizontal_range(cx):
            return "left" if cx < self.ADJUST_X_MIN else "right"
        
        # 然后调整前后方向
        if not self.is_in_vertical_range(cy):
            return "forward" if cy < self.y_min else "backward"
        
        # 如果都在可接受范围内但不在目标区域，可能是轻微偏移
        # 优先调整左右方向
        if cx < self.x_min or cx > self.x_max:
            return "left" if cx < self.target_center_x else "right"
        
        # 最后调整前后方向
        return "forward" if cy < self.target_center_y else "backward"
    
    def send_control_command(self, cmd, duration=None):
        """发送控制指令到小车"""
        try:
            self.control_sock.sendto(cmd.encode(), (self.RASPBERRY_IP, self.CONTROL_PORT))
            print(f"已发送控制指令: {cmd}")
            self.current_command = cmd
            self.command_start_time = time.time()
            
            # 设置命令持续时间
            if duration is not None:
                self.command_duration = duration
            else:
                # 默认持续时间
                self.command_duration = 0.3 if cmd in ['a', 'd'] else 0.2
        except Exception as e:
            print(f"发送控制指令失败: {e}")

    def send_grab_command(self, class_name):
        """发送抓取命令"""
        if class_name in self.CLASS_TO_COMMAND:
            grab_cmd = self.CLASS_TO_COMMAND[class_name]
            try:
                self.control_sock.sendto(grab_cmd.encode(), (self.RASPBERRY_IP, self.CONTROL_PORT))
                print(f"✅ 发送抓取命令: {grab_cmd} (类别: {class_name})")
                self.grab_command_sent = True
                self.grab_target_class = class_name
                return True
            except Exception as e:
                print(f"发送抓取命令失败: {e}")
                return False
        else:
            print(f"⚠️ 未找到类别 '{class_name}' 对应的抓取命令")
            return False
    
    def stop_movement(self):
        """停止所有移动"""
        if self.current_command != 'x':
            self.send_control_command('x')
        self.current_command = None
    
    def check_command_timeout(self):
        """检查命令是否超时"""
        if self.current_command and time.time() - self.command_start_time > self.command_duration:
            self.stop_movement()
            return True
        return False
    
    def process_frame(self, frame):
        """处理帧并进行目标检测"""
        # 绘制目标区域
        cv2.polylines(frame, [np.array(self.TARGET_REGION + [self.TARGET_REGION[0]], np.int32)],
                      isClosed=True, color=(255, 0, 0), thickness=2)
        
        # 绘制左右调整区域
        adjust_region = [
            (self.ADJUST_X_MIN, self.y_min),
            (self.ADJUST_X_MIN, self.y_max),
            (self.ADJUST_X_MAX, self.y_max),
            (self.ADJUST_X_MAX, self.y_min)
        ]
        cv2.polylines(frame, [np.array(adjust_region + [adjust_region[0]], np.int32)],
                      isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        
        # 绘制中心点
        cv2.circle(frame, (int(self.target_center_x), int(self.target_center_y)), 
                  5, (0, 0, 255), -1)
        
        # 目标检测推理
        results = self.model(frame, conf=0.35, verbose=False)[0]
        names = self.model.names
        
        closest_dist = float('inf')
        closest_instruction = None
        object_in_region = False
        object_center = None
        closest_label = None
        
        for i, (box, conf, cls) in enumerate(zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls)):
            x1, y1, x2, y2 = map(int, box.tolist())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            instruction = self.decide_move(cx, cy)
            class_id = int(cls.item())
            label = names[class_id] if names else f"id{class_id}"
            confidence = conf.item()
            
            # 判断目标是否在区域内
            if self.x_min <= cx <= self.x_max and self.y_min <= cy <= self.y_max:
                object_in_region = True
                closest_label = label
            
            # 更新最近目标信息
            dist = np.hypot(cx - self.target_center_x, cy - self.target_center_y)
            if dist < closest_dist:
                closest_dist = dist
                closest_instruction = instruction
                object_center = (cx, cy)
            
            # 可视化目标
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
            text = f"{label} {confidence:.2f} | {instruction}"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 绘制目标中心到区域中心的连线
        if object_center:
            cv2.line(frame, (int(object_center[0]), int(object_center[1])),
                    (int(self.target_center_x), int(self.target_center_y)),
                    (255, 0, 255), 2)
        
        # 控制逻辑：根据检测结果发送指令
        current_time = time.time()
        
        # 检查命令是否超时
        self.check_command_timeout()
        
        # 检查控制间隔
        if current_time - self.last_control_time < self.control_interval:
            return frame
        
        # 更新最后控制时间
        self.last_control_time = current_time
        
        # 如果目标在区域内，停止所有移动
        if object_in_region:
            self.stop_movement()
            cv2.putText(frame, "TARGET IN POSITION", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 如果有目标在区域内且未发送抓取命令
            if closest_label and not self.grab_command_sent:
                print("✅ 目标在区域内，准备抓取")
                
                # 查找最匹配的类别名称
                best_match = None
                max_similarity = 0
                
                for class_name in self.CLASS_TO_COMMAND.keys():
                    # 简单的相似度计算：检查类别名称是否包含在标签中
                    similarity = sum(1 for word in class_name.split() if word.lower() in label.lower())
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_match = class_name
                
                if best_match:
                    # 发送抓取命令
                    self.send_grab_command(best_match)
                    cv2.putText(frame, f"GRAB COMMAND SENT: {self.CLASS_TO_COMMAND[best_match]}", 
                               (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    print(f"⚠️ 无法匹配标签 '{closest_label}' 到已知类别")
                    cv2.putText(frame, f"UNKNOWN CLASS: {closest_label}", 
                               (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame
        
        # 重置抓取状态（当目标离开区域时）
        self.grab_command_sent = False
        self.grab_target_class = None
        
        # 如果没有检测到目标或指令，跳过
        if not closest_instruction or self.current_command:
            return frame
        
        # 根据指令发送控制命令
        if closest_instruction == "left":
            print("向左调整")
            self.send_control_command('a', duration=0.1)  # 左转
            self.last_adjustment = "horizontal"
        
        elif closest_instruction == "right":
            print("向右调整")
            self.send_control_command('d', duration=0.1)  # 右转
            self.last_adjustment = "horizontal"
        
        elif closest_instruction == "forward":
            print("向前调整")
            self.send_control_command('s', duration=0.1)  # 前进
            self.last_adjustment = "vertical"
        
        elif closest_instruction == "backward":
            print("向后调整")
            self.send_control_command('w', duration=0.1)  # 后退
            self.last_adjustment = "vertical"
        
        return frame