#main.py
from flask import Flask, render_template, Response, jsonify
import threading
import time
import cv2
from facenet_retinaface_pytorch import cam_recognition
import camera_receiver
import keyboard_controller
import os

app = Flask(__name__)

# 全局状态
auth_status = False
auth_message = "等待人脸识别"
local_frame = None
car_frame = None

def run_face_recognition():
    """运行人脸识别线程"""
    global auth_status, auth_message
    
    while not auth_status:
        auth_message = "正在进行人脸识别..."
        result = cam_recognition.main(cam_id=0, view_size=(640, 480))
        
        if result:
            auth_status = True
            auth_message = "认证成功！"
            # 启动小车相关模块
            threading.Thread(target=camera_receiver.run, daemon=True).start()
            keyboard_controller.start()  # 启动键盘控制器
        else:
            auth_message = "认证失败，5秒后重试..."
            time.sleep(5)

# 在程序退出时停止键盘控制器
import atexit
atexit.register(keyboard_controller.stop)

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/unlocked')
def unlocked():
    """解锁后页面"""
    if not auth_status:
        return render_template('index.html')
    return render_template('unlocked.html')

@app.route('/auth_status')
def get_auth_status():
    """获取认证状态"""
    return jsonify({
        "authenticated": auth_status,
        "message": auth_message
    })

def gen_local_camera():
    """生成本地摄像头视频流（用于人脸识别）"""
    while True:
        frame = cam_recognition.get_current_frame()
        if frame is not None:
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
        else:
            # 返回占位符或空帧
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + b'\xff'*1024 + b'\r\n\r\n')
        time.sleep(0.05)  # 控制帧率

@app.route('/local_video_feed')
def local_video_feed():
    """本地视频流路由"""
    return Response(gen_local_camera(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_car_camera():
    """生成小车摄像头视频流"""
    placeholder = None
    placeholder_path = 'static/images/placeholder.jpg'
    if os.path.exists(placeholder_path):
        placeholder = open(placeholder_path, 'rb').read()
    
    while True:
        frame = camera_receiver.get_frame()
        if frame is not None:
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
        else:
            if placeholder:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n\r\n')
            else:
                # 返回黑色图像
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + b'\xff'*1024 + b'\r\n\r\n')
        time.sleep(0.05)  # 控制帧率

@app.route('/car_video_feed')
def car_video_feed():
    """小车视频流路由"""
    return Response(gen_car_camera(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/active_keys')
def get_active_keys():
    """获取当前按下的键"""
    return keyboard_controller.get_active_keys()

if __name__ == '__main__':
    # 启动人脸识别线程
    threading.Thread(target=run_face_recognition, daemon=True).start()
    
    # 启动Flask应用
    app.run(host='0.0.0.0', port=5000, threaded=True)