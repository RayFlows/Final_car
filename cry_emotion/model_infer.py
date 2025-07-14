import os
import random
import torch
import librosa
import joblib
from transformers import ASTFeatureExtractor, ASTForAudioClassification

SAMPLE_RATE = 16000
DEVICE = torch.device("cpu")


model = ASTForAudioClassification.from_pretrained("cry_emotion/ast_babycry_model").to(DEVICE)
model.eval()

feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
label_encoder = joblib.load("cry_emotion/label_encoder.pkl")

def predict_random_sample(dataset_dir="cry_emotion/dataset_cry"):
    print("predict_random_sample() 被调用")

    wav_files = []
    for label in os.listdir(dataset_dir):
        label_path = os.path.join(dataset_dir, label)
        if not os.path.isdir(label_path):
            continue
        for file in os.listdir(label_path):
            if file.lower().endswith('.wav'):
                wav_files.append({
                    "path": os.path.join(label_path, file),
                    "label": label
                })

    if not wav_files:
        print("没有找到任何 .wav 文件")
        raise FileNotFoundError("数据集目录中未找到 .wav 文件")

    sample = random.choice(wav_files)
    path = sample["path"]
    true_label = sample["label"]

    print(f"随机选取音频: {os.path.basename(path)} (实际标签: {true_label})")

    waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    inputs = feature_extractor(waveform, sampling_rate=SAMPLE_RATE)
    input_tensor = torch.tensor(inputs["input_values"][0], dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_values=input_tensor)
        pred = torch.argmax(output.logits, dim=-1).item()

    pred_label = label_encoder.inverse_transform([pred])[0]

    print(f"模型预测标签: {pred_label} | {'正确' if pred_label == true_label else '错误'}")

    return {
        "filename": os.path.basename(path),
        "actual": true_label,
        "predicted": pred_label,
        "correct": true_label == pred_label
    }
