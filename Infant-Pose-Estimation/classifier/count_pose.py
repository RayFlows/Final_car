import json

def count_postures(json_file):
    # 读取 JSON 文件
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 创建一个字典来统计每个姿势的出现次数
    posture_counts = {}

    # 遍历所有图片条目，统计每个姿势
    for item in data.get('images', []):
        posture = item.get('posture')
        if posture:
            if posture not in posture_counts:
                posture_counts[posture] = 1
            else:
                posture_counts[posture] += 1

    # 打印结果
    for posture, count in posture_counts.items():
        print(f"Posture: {posture}, Count: {count}")

if __name__ == "__main__":
    # 设置你的 JSON 文件路径
    input_json_path = "/Users/ruiyuhan/Desktop/Advanced/Infant-Pose-Estimation/data/SyRIP_Posture/annotations/train600/processed_train_data.json"  # 替换为实际路径

    # 调用函数统计姿势
    count_postures(input_json_path)
