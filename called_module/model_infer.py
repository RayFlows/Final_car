import os
import random
import time
import librosa
import torch
import sqlite3
import pygame
import joblib
from transformers import ASTFeatureExtractor, ASTForAudioClassification

SAMPLE_RATE = 16000
DEVICE = torch.device("cpu")

model = ASTForAudioClassification.from_pretrained("cry_emotion/ast_babycry_model").to(DEVICE)
model.eval()

feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
label_encoder = joblib.load("cry_emotion/label_encoder.pkl")

subfolders = ['awake', 'hungry', 'sleepy', 'uncomfortable']
current_subfolder_index = 3

dataset_dir = "cry_emotion/dataset_cry"


def play_audio_with_pygame(audio_path):
    """用 pygame 播放音频"""
    pygame.mixer.init()
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


def create_and_insert_initial_data():
    # 初始化 SQLite 数据库并插入示例数据
    conn = sqlite3.connect('audio_predictions.db')
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TEXT,
                        audio_path TEXT,
                        predicted_label TEXT,
                        average_volume REAL,
                        duration REAL)''')

    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    initial_data = [
        ("path/to/audio1.wav", "awake", 0.12, 5.6),
        ("path/to/audio2.wav", "hungry", 0.15, 6.3),
        ("path/to/audio3.wav", "sleepy", 0.08, 4.2),
        ("path/to/audio4.wav", "uncomfortable", 0.10, 5.0),
        ("path/to/audio5.wav", "awake", 0.20, 5.8),
        ("path/to/audio6.wav", "hungry", 0.18, 6.0),
        ("path/to/audio7.wav", "sleepy", 0.09, 4.5),
        ("path/to/audio8.wav", "uncomfortable", 0.11, 4.8),
        ("path/to/audio9.wav", "awake", 0.13, 5.3),
        ("path/to/audio10.wav", "hungry", 0.16, 5.2)
    ]

    for data in initial_data:
        cursor.execute('''INSERT INTO predictions (created_at, audio_path, predicted_label, average_volume, duration)
                          VALUES (?, ?, ?, ?, ?)''',
                       (current_time, data[0], data[1], data[2], data[3]))

    conn.commit()
    conn.close()


def calculate_average_volume(waveform):
    """计算音频的平均音量（RMS）"""
    rms = librosa.feature.rms(y=waveform)
    return rms.mean()


def get_audio_duration(waveform, sample_rate):
    """计算音频的持续时间"""
    return len(waveform) / sample_rate


def insert_to_database(audio_path, predicted_label, average_volume, duration):
    """将音频预测数据插入数据库"""
    conn = sqlite3.connect('audio_predictions.db')
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TEXT,
                        audio_path TEXT,
                        predicted_label TEXT,
                        average_volume REAL,
                        duration REAL)''')

    created_at = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    cursor.execute('''INSERT INTO predictions (created_at, audio_path, predicted_label, average_volume, duration)
                      VALUES (?, ?, ?, ?, ?)''',
                   (created_at, audio_path, predicted_label, average_volume, duration))

    conn.commit()

    print(f"数据已插入: {audio_path}, {predicted_label}, {average_volume}, {duration}, {created_at}")

    conn.close()


def predict_sequential_samples():
    global current_subfolder_index

    current_subfolder = subfolders[current_subfolder_index]
    subfolder_path = os.path.join(dataset_dir, current_subfolder)

    if not os.path.isdir(subfolder_path):
        raise FileNotFoundError(f"子文件夹 {current_subfolder} 不存在")

    wav_files = [file for file in os.listdir(subfolder_path) if file.lower().endswith('.wav')]

    if not wav_files:
        raise FileNotFoundError(f"{current_subfolder} 中没有 .wav 文件")

    sample = random.choice(wav_files)
    path = os.path.join(subfolder_path, sample)
    true_label = current_subfolder

    print(f"正在播放音频: {os.path.basename(path)}")
    play_audio_with_pygame(path)

    waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    inputs = feature_extractor(waveform, sampling_rate=SAMPLE_RATE)
    input_tensor = torch.tensor(inputs["input_values"][0], dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_values=input_tensor)
        pred = torch.argmax(output.logits, dim=-1).item()

    pred_label = label_encoder.inverse_transform([pred])[0]

    print(f"预测标签: {pred_label} | {'正确' if pred_label == true_label else '错误'}")

    average_volume = calculate_average_volume(waveform)
    duration = get_audio_duration(waveform, SAMPLE_RATE)

    insert_to_database(path, pred_label, average_volume, duration)

    current_subfolder_index = (current_subfolder_index + 1) % 4

    return {
        "filename": os.path.basename(path),
        "actual": true_label,
        "predicted": pred_label,
        "correct": true_label == pred_label
    }
