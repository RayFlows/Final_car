import json
from collections import defaultdict

def count_keys(json_file):
    # 打开并读取 JSON 文件
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 使用 defaultdict 来统计每个键的出现次数
    key_count = defaultdict(int)

    # 遍历所有图像条目
    for item in data.get('images', []):
        # 遍历每个图像条目的键
        for key in item:
            key_count[key] += 1

    # 输出每个键的出现次数
    for key, count in key_count.items():
        print(f"键: {key} | 出现次数: {count}")

if __name__ == "__main__":
    # 设置你的 JSON 文件路径
    input_json_path = "../data/syrip/annotations/person_keypoints_train_infant.json"  # 替换为你的文件路径

    # 调用函数统计每个键的出现次数
    count_keys(input_json_path)
