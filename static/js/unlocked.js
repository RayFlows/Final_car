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
  container
  

  if (mode === 'gesture') {
    // const currentModeElement = document.getElementById('current-mode');
    // if (currentModeElement) {
    //   currentModeElement.textContent = mode;
    // }
    container.innerHTML = `
      <h3>Gesture Recognition Camera</h3>
      <img src="/gesture_video_feed" alt="Gesture camera" class="video-feed">
      <div class="current-mode">
        <h3>Current mode: <span id="current-mode">Unset</span></h3>
      </div>
    `;
  }
  else if (mode === 'voice') {
    container.innerHTML = `
        <h3>AI Response Area</h3>

        <!-- Response Area -->
        <div id="ai-response"
            style="height:40%;overflow-y:auto;border:1px solid #ccc;padding:5px;background:#f0f0f0;">
          （Waiting for response）
        </div>

        <!-- Input Row -->
        <div style="margin-top:10px;">
          <input id="ai-input" type="text" placeholder="What would you like to say..." style="width:60%;">
          <button onclick="submitToAI()">Send</button>
          <!-- Record Button: Click to start, click again to stop -->
          <button id="mic-btn" onclick="toggleRecord()">🎤</button>
        </div>

        <!-- Play the returned TTS mp3 -->
        <audio id="ai-audio" style="margin-top:8px;width:100%;" controls></audio>
    `;
  } else if (mode === 'keyboard') {
    container.innerHTML = `
      <h3>Current pressed key:</h3>
      <div id="active-keys" style="font-size: 18px;">No key pressed</div>
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
                    keysDiv.textContent = 'No pressed key';
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

  document.getElementById('ai-response').innerText = '⏳ Thinking…';
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

    document.getElementById('ai-response').innerText = '⏳ Uploading voice...';

    const res  = await fetch(AI_VOICE_API, {method: 'POST', body: fd});
    const data = await res.json();
    showAIResult(data);
  };

  mediaRecorder.start();
  btn.textContent = '■';
}



function showAIResult(data) {
  const respDiv = document.getElementById('ai-response');
  respDiv.innerText = data.reply || '(Empty response)';

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
  span.innerText = '⏳ Recognizing...';

  try {
    const res = await fetch('/predict_cry_once');
    const data = await res.json();

    if (data.error) {
      span.innerText = '❌ Error：' + data.error;
      return;
    }

    span.innerText = `${data.predicted}（${data.correct ? 'Correct' : 'Incorrect'}）`;
  } catch (err) {
    span.innerText = '❌ Request failed!';
    console.error(err);
  }
}




async function getCurrentMode() {
  const res = await fetch('/get_mode', {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json'
    }
  });

  const data = await res.json();

  if (data.status === "success") {
    const currentMode = data.mode;
    const currentModeElement = document.getElementById('current-mode');
    if (currentModeElement) {
      currentModeElement.textContent = currentMode;
    }
  } else {
    console.error('Failed to get current mode:', data.message);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  // 页面加载完成时，获取当前模式
  getCurrentMode();

  // 每隔 500 毫秒获取一次当前模式并更新显示
  setInterval(getCurrentMode, 500);
});


// 获取数据库数据并生成图表
async function fetchEmotionData() {
  try {
    const response = await fetch('/get_emotion_data');
    const data = await response.json();

    // 更新三个图表
    updateEmotionChangeChart(data.emotionChange);
    updateEmotionFrequencyChart(data.emotionFrequency);
    updateVolumeTrendChart(data.volumeTrend);
  } catch (error) {
    console.error('Error fetching emotion data:', error);
  }
}

// 情绪变化折线图
// 情绪变化折线图
function updateEmotionChangeChart(emotionChange) {
  // 创建情绪到高度的映射
  const emotionHeightMap = {
    "uncomfortable": 1,
    "hungry": 2,
    "awake": 4,
    "sleepy": 3
  };

  // 使用情绪标签映射为相应的高度
  const emotionValues = emotionChange.map(emotion => emotionHeightMap[emotion] || 0);  // 默认为 0 如果情绪不在映射中

  // 设置图表数据
  const data = {
    labels: emotionChange,  // 情绪标签作为 X 轴
    datasets: [{
      label: 'Emotion Change',
      data: emotionValues,  // 根据情绪映射的数值作为 Y 轴
      borderColor: 'rgba(75, 192, 192, 1)',  // 线的颜色
      backgroundColor: 'rgba(75, 192, 192, 0.2)',  // 背景颜色
      fill: true,  // 填充区域
    }]
  };

  // 获取图表的上下文
  const ctx = document.getElementById('emotionChart').getContext('2d');
  
  // 创建或更新图表
  new Chart(ctx, {
    type: 'line',  // 折线图
    data: data,
    options: {
      responsive: true,
      scales: {
        x: {
          beginAtZero: true
        },
        y: {
          beginAtZero: true,
          ticks: {
            stepSize: 1,  // 每个步骤为 1
            max: 5  // Y 轴的最大值为 5，确保能够显示所有情绪的高度
          }
        }
      }
    }
  });
}

// 情绪频次饼图
function updateEmotionFrequencyChart(emotionFrequency) {
  const labels = Object.keys(emotionFrequency);
  const data = {
    labels: labels,
    datasets: [{
      label: 'Emotion Frequency',
      data: Object.values(emotionFrequency),
      backgroundColor: ['rgba(255, 99, 132, 0.2)', 'rgba(54, 162, 235, 0.2)', 'rgba(255, 206, 86, 0.2)', 'rgba(75, 192, 192, 0.2)'],
      borderColor: ['rgba(255, 99, 132, 1)', 'rgba(54, 162, 235, 1)', 'rgba(255, 206, 86, 1)', 'rgba(75, 192, 192, 1)'],
      borderWidth: 1
    }]
  };

  const ctx = document.getElementById('emotionPie').getContext('2d');
  new Chart(ctx, {
    type: 'pie',
    data: data,
    options: {
      responsive: true
    }
  });
}

// 音量趋势折线图
function updateVolumeTrendChart(volumeTrend) {
  const labels = Object.keys(volumeTrend);
  const data = {
    labels: labels,
    datasets: [{
      label: 'Volume Trend',
      data: Object.values(volumeTrend),
      borderColor: 'rgba(153, 102, 255, 1)',
      backgroundColor: 'rgba(153, 102, 255, 0.2)',
      fill: true,
    }]
  };

  const ctx = document.getElementById('volumeChart').getContext('2d');
  new Chart(ctx, {
    type: 'line',
    data: data,
    options: {
      responsive: true,
      scales: {
        x: {
          beginAtZero: true
        },
        y: {
          beginAtZero: true
        }
      }
    }
  });
}

// 页面加载时获取情绪数据并更新图表
document.addEventListener('DOMContentLoaded', () => {
  fetchEmotionData();
});



function setupSocket() {
  // 连接到后端的 SocketIO 服务
  var socket = io.connect('http://' + document.domain + ':' + location.port);

  // 当从后端接收到新的姿势识别结果时，更新前端页面
  socket.on('update_prediction', function(data) {
      // 获取姿势预测结果
      var posture = data.posture;
      console.log("Received posture: ", posture);

      // 更新页面上的姿势识别结果
      document.getElementById('detected-objects').innerText = posture;
  });
}

document.addEventListener("DOMContentLoaded", function() {

    // 1. 初始化与后端服务器的 Socket.IO 连接
    const socket = io();

    // (可选) 监听连接成功事件，方便在浏览器控制台调试
    socket.on('connect', function() {
        console.log('Successfully connected to the server');
    });

    /**
     * 这是核心函数部分：监听来自后端的 'update_prediction' 事件。
     * @param {object} data - 从后端接收到的数据对象，格式为 { posture: 'some_value' }
     */
    socket.on('update_prediction', function(data) {
        console.log('Received posture update:', data); // 在控制台打印接收到的数据，便于调试

        // 从数据对象中安全地获取姿势名称
        const posture = data.posture;

        // 找到页面上用于显示结果的 <span> 元素
        const displayElement = document.getElementById('detected-objects');

        // 检查是否成功找到了该元素
        if (displayElement) {
            // 如果 posture 有有效值，则更新元素的文本内容；
            // 否则，显示默认的 "——"
            displayElement.innerText = posture || '——';
        } else {
            console.error('Error: Element with ID "detected-objects" not found on the page.');
        }
    });

    // (可选) 监听断开连接事件
    socket.on('disconnect', function() {
        console.log('Disconnected from the server.');
    });

});
