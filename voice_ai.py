import os
import re
import time
import json
import uuid
import requests
from gtts import gTTS
import speech_recognition as sr


MODEL_NAME = "deepseek-r1:1.5b"
OLLAMA_URL = "http://localhost:11434/api/generate"


# 把语音文件转成文本
def recognize_voice_google(filepath):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filepath) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio, language="en-US")
        print("识别结果:", text)
        return text
    except Exception as e:
        print("语音识别错误:", e)
        return ""


# 和模型交互
def chat_with_ollama(prompt):
    print("正在向 Ollama 提问...")
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": True}
    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True)
        response.raise_for_status()
        reply = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                reply += data.get("response", "")
        reply_cleaned = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
        return reply_cleaned
    except Exception as e:
        print("Ollama 请求失败:", e)
        return "Couldn't process the request."
    
# 回答转成mp3
def generate_tts_mp3(text, save_dir):
    try:
        filename = f"{int(time.time())}_{uuid.uuid4().hex[:6]}.mp3"
        filepath = os.path.join(save_dir, filename)
        tts = gTTS(text=text, lang='en')
        tts.save(filepath)
        print("TTS 生成完成:", filename)
        return filename
    except Exception as e:
        print("TTS 生成失败:", e)
        return ""

# 英文提问英文回答
# Generate a short story, no more than 80 words, preferably one that children can understand.