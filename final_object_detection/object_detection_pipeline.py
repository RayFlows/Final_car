# cam_yolo_detect.py
# ---------------------------------------------------
# 实时摄像头 → YOLOv8 检测 → 绘制结果
# ---------------------------------------------------
import time
import cv2
from ultralytics import YOLO

def main(weight_path="runs/detect/train/weights/best.pt",
         cam_id=0, view_size=None, conf_thres=0.35):
    # ① 加载模型
    model = YOLO(weight_path)
    model.fuse()                          # 更快推理

    # ② 打开摄像头
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {cam_id}")

    if view_size:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  view_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, view_size[1])

    fps = 0.0
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Failed to read frame.")
            break

        # ③ YOLO 推理（直接传 BGR ndarray）
        results = model(frame, conf=conf_thres, verbose=False)[0]

        # ④ 可视化
        annotated = results.plot()        # Ultralytics 内置绘制
        fps = fps * 0.9 + 0.1 / (time.time() - t0)
        cv2.putText(annotated, f"FPS:{fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Camera", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):         # ESC / q 退出
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # view_size 可设 (640,480)；不设保持摄像头默认分辨率
    main(weight_path="runs/detect/train/weights/best.pt",
         cam_id=0, view_size=None, conf_thres=0.6)
