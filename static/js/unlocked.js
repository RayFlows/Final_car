const AI_TEXT_API  = '/ai/text';   // POST JSON {text:"..."}
const AI_VOICE_API = '/ai/voice';  // POST multipart/form-data {voice: wav}





















document.addEventListener('DOMContentLoaded', () => {
  const buttons = document.querySelectorAll('.controls button');
  buttons.forEach(button => {
    button.addEventListener('click', () => {
      const mode = button.dataset.mode;
      setControlMode(mode);
      buttons.forEach(btn => btn.classList.remove('active'));
      button.classList.add('active');
    });
  });

  setControlMode('gesture');
  document.querySelector('[data-mode="gesture"]')?.classList.add('active');
});



// unlocked页面第一行右侧板块控制按钮
let currentMode = '';

function setControlMode(mode) {
  const container = document.getElementById('control-content');
  container.innerHTML = '';

  if (mode === 'gesture') {
    container.innerHTML = `
      <h3>手势识别摄像头</h3>
      <img src="/gesture_video_feed" alt="手势摄像头" class="video-feed">
    `;
  }
  else if (mode === 'voice') {
    container.innerHTML = `
        <h3>AI 回答区域</h3>

        <!-- 返回区 -->
        <div id="ai-response"
            style="height:40%;overflow-y:auto;border:1px solid #ccc;padding:5px;background:#f0f0f0;">
          （等待回答）
        </div>

        <!-- 输入行 -->
        <div style="margin-top:10px;">
          <input id="ai-input" type="text" placeholder="你想说什么..." style="width:60%;">
          <button onclick="submitToAI()">发送</button>
          <!-- 录音按钮：点一下开始，再点一下结束 -->
          <button id="mic-btn" onclick="toggleRecord()">🎤</button>
        </div>

        <!-- 把 TTS 回来的 mp3 播出来 -->
        <audio id="ai-audio" style="margin-top:8px;width:100%;" controls></audio>
    `;
  } else if (mode === 'keyboard') {
    container.innerHTML = `
      <h3>当前按键:</h3>
      <div id="active-keys" style="font-size: 18px;">无按键按下</div>
    `;
    updateKeyStatus();
  }

  currentMode = mode;
}






// 从后端获取当前按下的键
function updateKeyStatus() {
    fetch('/active_keys')
        .then(response => response.json())
        .then(data => {
            const keysDiv = document.getElementById('active-keys');
            // 检查按键显示区域是否存在 (只在键盘模式下存在)
            if (keysDiv) { 
                if (data.keys && data.keys.length > 0) {
                    keysDiv.textContent = data.keys.join(', ');
                } else {
                    keysDiv.textContent = '无按键按下';
                }
            }
        });
}
setInterval(updateKeyStatus, 100);






















// unlocked页面第一行右侧板块语音ai交互功能


async function submitToAI() {
  const inp = document.getElementById('ai-input');
  const msg = inp.value.trim();
  if (!msg) return;

  document.getElementById('ai-response').innerText = '⏳ 正在思考…';
  inp.value = '';

  const res = await fetch(AI_TEXT_API, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text: msg})
  });
  const data = await res.json();
  showAIResult(data);
}




/* 浏览器端录音：点一次开始，再点一次停止并上传 */
let mediaRecorder = null;
let chunks = [];

async function toggleRecord() {
  const btn = document.getElementById('mic-btn');

  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
    btn.textContent = '🎤';
    return;
  }

  // 第一次点击：申请麦克风并开始录
  const stream = await navigator.mediaDevices.getUserMedia({audio: true});
  mediaRecorder = new MediaRecorder(stream);
  chunks = [];

  mediaRecorder.ondataavailable = e => chunks.push(e.data);
  mediaRecorder.onstop = async () => {
    const wav = new Blob(chunks, {type: 'audio/wav'});
    const fd  = new FormData();
    fd.append('voice', wav, 'voice.wav');

    document.getElementById('ai-response').innerText = '⏳ 语音上传中…';

    const res  = await fetch(AI_VOICE_API, {method: 'POST', body: fd});
    const data = await res.json();
    showAIResult(data);
  };

  mediaRecorder.start();
  btn.textContent = '■';
}



function showAIResult(data) {
  const respDiv = document.getElementById('ai-response');
  respDiv.innerText = data.reply || '(空回复)';

  // 播放 TTS
  if (data.audio_url) {
    const audio = document.getElementById('ai-audio');
    audio.src = data.audio_url + '?t=' + Date.now();
    audio.play().catch(() => {});
  }
}


























// 通用JavaScript功能

// 模拟按键功能
function simulateKey(key, action) {
    fetch(`/simulate_key?key=${key}&action=${action}`)
        .then(response => {
            if (!response.ok) {
                console.error(`Failed to simulate key ${key} ${action}`);
            }
        });
}

// 虚拟按键处理
document.addEventListener('DOMContentLoaded', () => {
    // 为虚拟按键添加事件监听器
    document.querySelectorAll('[data-key]').forEach(button => {
        button.addEventListener('mousedown', () => {
            const key = button.getAttribute('data-key');
            simulateKey(key, 'down');
        });
        
        button.addEventListener('mouseup', () => {
            const key = button.getAttribute('data-key');
            simulateKey(key, 'up');
        });
        
        // 触摸设备支持
        button.addEventListener('touchstart', (e) => {
            e.preventDefault();
            const key = button.getAttribute('data-key');
            simulateKey(key, 'down');
        });
        
        button.addEventListener('touchend', (e) => {
            e.preventDefault();
            const key = button.getAttribute('data-key');
            simulateKey(key, 'up');
        });
    });
});























async function triggerCryPrediction() {
  const span = document.getElementById('latest-emotion');
  span.innerText = '⏳ 正在识别...';

  try {
    const res = await fetch('/predict_cry_once');
    const data = await res.json();

    if (data.error) {
      span.innerText = '❌ 错误：' + data.error;
      return;
    }

    span.innerText = `${data.predicted}（${data.correct ? '正确' : '错误'}）`;
  } catch (err) {
    span.innerText = '❌ 请求失败';
    console.error(err);
  }
}
