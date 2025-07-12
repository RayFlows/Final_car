import time
import cv2
import numpy as np
from retinaface import Retinaface

def main(cam_id: int = 0, view_size: tuple | None = None):
    retinaface = Retinaface()

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Can't open camera {cam_id}")

    if view_size:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  view_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, view_size[1])

    print("🔹 摄像头已打开，按 <Space> 进行人脸识别，Esc/q 退出。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  无法读取摄像头帧，退出。")
            break

        cv2.imshow("Camera Preview (press SPACE to detect)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q')):  # Esc 或 q 退出
            break

        if key == 32:  # 空格键
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 你需要在 retinaface.detect_image 中支持 return_info
            result_rgb, boxes, probs, names = retinaface.detect_image(rgb, return_info=True)
            result_bgr = cv2.cvtColor(np.array(result_rgb), cv2.COLOR_RGB2BGR)

            # 绘制 bbox + 名字 + conf
            for box, prob, name in zip(boxes, probs, names):
                box = [int(b) for b in box]
                cv2.rectangle(result_bgr, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                label = f"{name} {prob:.2f}"
                cv2.putText(result_bgr, label, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Detection Result", result_bgr)
            cv2.waitKey(1500)

            if any(n != "Unknown" for n in names):
                unlocked = frame.copy()
                cv2.putText(unlocked, "Unlocked Successfully!", (40, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.imshow("Unlocked", unlocked)
                cv2.waitKey(1500)
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(cam_id=0, view_size=None)
