import os
import random
import torch
import librosa
import joblib
from transformers import ASTFeatureExtractor, ASTForAudioClassification
import pygame

SAMPLE_RATE = 16000
DEVICE = torch.device("cpu")

model = ASTForAudioClassification.from_pretrained("cry_emotion/ast_babycry_model").to(DEVICE)
model.eval()

feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
label_encoder = joblib.load("cry_emotion/label_encoder.pkl")

subfolders = ['awake', 'hungry', 'sleepy', 'uncomfortable']

current_subfolder_index = 0

dataset_dir = "cry_emotion/dataset_cry"

def play_audio_with_pygame(wav_file):
    pygame.mixer.init()
    sound = pygame.mixer.Sound(wav_file)
    sound.play()
    while pygame.mixer.get_busy():
        pygame.time.delay(100)

def predict_sequential_samples():
    global current_subfolder_index

    print("predict_random_sample() 被调用")

    current_subfolder = subfolders[current_subfolder_index]
    subfolder_path = os.path.join(dataset_dir, current_subfolder)

    if not os.path.isdir(subfolder_path):
        print(f"警告: 子文件夹 {current_subfolder} 不存在或不是目录")
        raise FileNotFoundError(f"子文件夹 {current_subfolder} 未找到")

    wav_files = []
    for file in os.listdir(subfolder_path):
        if file.lower().endswith('.wav'):
            wav_files.append({
                "path": os.path.join(subfolder_path, file),
                "label": current_subfolder
            })

    if not wav_files:
        print("没有找到任何 .wav 文件")
        raise FileNotFoundError(f"子文件夹 {current_subfolder} 中未找到 .wav 文件")

    sample = random.choice(wav_files)
    path = sample["path"]
    true_label = sample["label"]

    print(f"当前子文件夹: {current_subfolder} | 随机选取音频: {os.path.basename(path)} (实际标签: {true_label})")

    print(f"正在播放音频: {os.path.basename(path)}")
    play_audio_with_pygame(path)

    waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    inputs = feature_extractor(waveform, sampling_rate=SAMPLE_RATE)
    input_tensor = torch.tensor(inputs["input_values"][0], dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_values=input_tensor)
        pred = torch.argmax(output.logits, dim=-1).item()

    pred_label = label_encoder.inverse_transform([pred])[0]

    print(f"模型预测标签: {pred_label} | {'正确' if pred_label == true_label else '错误'}")

    current_subfolder_index = (current_subfolder_index + 1) % 4

    return {
        "filename": os.path.basename(path),
        "actual": true_label,
        "predicted": pred_label,
        "correct": true_label == pred_label
    }
