import time
import cv2
import numpy as np
from ultralytics import YOLO
import torch

# 定义目标区域
TARGET_REGION = [
    (270, 190),
    (270, 290),
    (370, 290),
    (370, 190)
]

x_min = min(p[0] for p in TARGET_REGION)
x_max = max(p[0] for p in TARGET_REGION)
y_min = min(p[1] for p in TARGET_REGION)
y_max = max(p[1] for p in TARGET_REGION)

target_center_x = (x_min + x_max) / 2
target_center_y = (y_min + y_max) / 2

def decide_move(cx, cy):
    dx = cx - target_center_x
    dy = cy - target_center_y
    threshold = 10
    if abs(dx) < threshold and abs(dy) < threshold:
        return "in position"
    direction_x = "left" if dx > 0 else "right"
    direction_y = "up" if dy > 0 else "down"
    return f"move {direction_x} and {direction_y}"

def main(weight_path="runs/detect/train/weights/best.pt",
         cam_id=0, view_size=None, conf_thres=0.7):
    model = YOLO(weight_path)
    model.fuse()

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {cam_id}")

    if view_size:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, view_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, view_size[1])

    fps = 0.0
    last_print_time = time.time()

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Failed to read frame.")
            break

        cv2.polylines(frame, [np.array(TARGET_REGION + [TARGET_REGION[0]], np.int32)],
                      isClosed=True, color=(255, 0, 0), thickness=2)

        results = model(frame, conf=conf_thres, verbose=False)[0]
        names = model.names

        closest_dist = float('inf')
        closest_instruction = None
        object_in_region = False   # ✅ 新增变量：是否有目标在区域中

        for i, (box, conf, cls) in enumerate(zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls)):
            x1, y1, x2, y2 = map(int, box.tolist())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            instruction = decide_move(cx, cy)
            class_id = int(cls.item())
            label = names[class_id] if names else f"id{class_id}"
            confidence = conf.item()

            # ✅ 判断是否在区域内
            if x_min <= cx <= x_max and y_min <= cy <= y_max:
                object_in_region = True

            # ✅ 更新最近目标信息
            dist = np.hypot(cx - target_center_x, cy - target_center_y)
            if dist < closest_dist:
                closest_dist = dist
                closest_instruction = instruction

            # ✅ 可视化
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
            text = f"{label} {confidence:.2f} | {instruction}"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # ✅ 每隔0.2秒输出控制台信息
        now = time.time()
        if now - last_print_time > 0.2:
            if object_in_region:
                print("✅ ready to grab")
            elif closest_instruction:
                print(f"[指令] {closest_instruction}")
            last_print_time = now

        # 显示 FPS
        fps = fps * 0.9 + 0.1 / (time.time() - t0)
        cv2.putText(frame, f"FPS:{fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Control", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main(weight_path="detect 2/train2/weights/last.pt",
         cam_id=0, view_size=None, conf_thres=0.35)
