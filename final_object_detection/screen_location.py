import cv2

# 鼠标回调函数
def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"🖱️  鼠标点击位置: ({x}, {y})")
        # 记录点击点
        clicked_points.append((x, y))

# 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ 无法打开摄像头")

cv2.namedWindow("Camera View")
cv2.setMouseCallback("Camera View", on_mouse_click)

clicked_points = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 无法读取帧")
        break

    # 绘制已点击的点
    for pt in clicked_points:
        cv2.circle(frame, pt, 5, (0, 255, 255), -1)
        cv2.putText(frame, f"{pt}", (pt[0] + 10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("Camera View", frame)

    # 按 q 或 Esc 键退出
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
