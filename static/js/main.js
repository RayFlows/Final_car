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