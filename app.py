# main.py
from flask import Flask, render_template, Response, jsonify, send_from_directory, url_for, request
import threading
import time

import cv2
from facenet_retinaface_pytorch import cam_recognition

import keyboard_controller
import os
import camera_receiver
import atexit

import uuid
from called_module.voice_ai import chat_with_ollama, generate_tts_mp3, recognize_voice_google

import torch


app = Flask(__name__)
# 全局状态
# auth_status = True
# auth_message = "已跳过人脸识别，直接进入控制界面"
auth_status = False
auth_message = "人脸识别"
local_frame = None
car_frame = None

# def run_face_recognition():
#     """运行人脸识别线程"""
#     global auth_status, auth_message
    
#     while not auth_status:
#         auth_message = "正在进行人脸识别..."
#         result = cam_recognition.main(cam_id=0, view_size=(640, 480))
        
#         if result:
#             auth_status = True
#             auth_message = "认证成功！"
#             # 启动小车相关模块
#             threading.Thread(target=camera_receiver.run, daemon=True).start()
#             keyboard_controller.start()  # 启动键盘控制器
#         else:
#             auth_message = "认证失败，5秒后重试..."
#             time.sleep(5)
def run_face_recognition():
    """运行人脸识别线程"""
    global auth_status, auth_message
    
    # 跳过人脸识别，直接认证成功
    auth_status = True
    auth_message = "认证成功！"
    
    # 启动小车相关模块
    threading.Thread(target=camera_receiver.run, daemon=True).start()
    # keyboard_controller.start()  # 启动键盘控制器


import atexit
atexit.register(keyboard_controller.stop)

@app.route('/')
def index():
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
def get_active_keys_route():
    """返回当前被按下的键"""
    return jsonify({"keys": keyboard_controller.get_active_keys()})


AUDIO_DIR = os.path.join(app.static_folder, 'ai_audio')
os.makedirs(AUDIO_DIR, exist_ok=True)

@app.route('/ai/text', methods=['POST'])
def ai_text():
    text = request.json.get('text', '')
    answer = chat_with_ollama(text)
    mp3_path = generate_tts_mp3(answer, AUDIO_DIR)
    audio_url = url_for('static', filename=f'ai_audio/{mp3_path}')
    return jsonify({'reply': answer, 'audio_url': audio_url})

@app.route('/ai/voice', methods=['POST'])
def ai_voice():
    wav = request.files['voice']
    tmpname = os.path.join('/tmp', f'{uuid.uuid4()}.wav')
    wav.save(tmpname)

    question = recognize_voice_google(tmpname)
    os.remove(tmpname)

    answer = chat_with_ollama(question)
    mp3_path = generate_tts_mp3(answer, AUDIO_DIR)
    audio_url = url_for('static', filename=f'ai_audio/{mp3_path}')
    return jsonify({'reply': answer, 'audio_url': audio_url})



# 音频类别识别
from called_module.model_infer import predict_sequential_samples
@app.route('/predict_cry_once')
def predict_cry_once():
    try:
        result = predict_sequential_samples()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 手势识别视频流路由
from called_module.gesture_recognition_stream import gen_gesture_camera
@app.route('/gesture_video_feed')
def gesture_video_feed():
    return Response(gen_gesture_camera(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 手势识别控制模式的路由
import threading

# 添加一个锁来同步 current_mode 的更新
current_mode = "pose"
mode_lock = threading.Lock()

@app.route('/set_mode', methods=['POST'])
def set_processing_mode():
    """设置当前模式"""
    global current_mode
    mode = request.json.get("mode", "")
    if mode in ["cruise", "pose"]:
        current_mode = mode
        print(f"切换至 {current_mode} 模式")
        return jsonify({"status": "success", "mode": current_mode})
    return jsonify({"status": "error", "message": "无效的模式"}), 400

@app.route('/get_mode', methods=['GET'])
def get_mode():
    """获取当前模式"""
    return jsonify({"status": "success", "mode": current_mode})





import sqlite3
from flask import jsonify

def get_emotion_data():
    """从数据库获取情绪数据"""
    conn = sqlite3.connect('audio_predictions.db')
    cursor = conn.cursor()

    # 获取情绪变化数据：按时间顺序获取最近的情绪
    cursor.execute('''
        SELECT predicted_label, created_at
        FROM predictions
        ORDER BY created_at DESC
        LIMIT 6
    ''')
    emotion_data = cursor.fetchall()

    # 获取情绪频次数据
    cursor.execute('''
        SELECT predicted_label, COUNT(*)
        FROM predictions
        GROUP BY predicted_label
    ''')
    emotion_freq_data = cursor.fetchall()

    # 获取音量趋势数据
    cursor.execute('''
        SELECT duration, AVG(average_volume)
        FROM predictions
        GROUP BY duration
    ''')
    volume_trend_data = cursor.fetchall()

    conn.close()

    # 数据格式化
    emotions = [data[0] for data in emotion_data]
    timestamps = [data[1] for data in emotion_data]
    emotion_frequency = {label: count for label, count in emotion_freq_data}
    volume_trend = {duration: volume for duration, volume in volume_trend_data}

    return {
        'emotionChange': emotions[::-1],  # 最近的情绪倒序显示
        'emotionFrequency': emotion_frequency,
        'volumeTrend': volume_trend
    }

@app.route('/get_emotion_data')
def get_emotion_data_route():
    """返回情绪数据的接口"""
    data = get_emotion_data()
    return jsonify(data)


from extensions import socketio
socketio.init_app(app)


from called_module.model_infer import create_and_insert_initial_data
if __name__ == '__main__':
    # 启动人脸识别线程
    threading.Thread(target=run_face_recognition, daemon=True).start()
    
    # keyboard_controller.start()

    # 启动Flask应用
    app.run(host='0.0.0.0', port=5000, threaded=True)
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)


















