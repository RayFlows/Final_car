import json

def check_labeled_data(json_file):
    # 打开并读取 JSON 文件
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 遍历所有图像条目，检查 is_labeled 是否为 "false"（字符串）
    for item in data.get('images', []):
        # 检查 is_labeled 字段是否为 "false"（字符串）
        if str(item.get('is_labeled', 'false')).lower() == 'false':
            print(f"图片 {item['file_name']} 没有标签")
            # 打印该条目的其他信息，帮助你检查
            print(f"File Name: {item['file_name']}")
            print(f"Height: {item['height']}, Width: {item['width']}")
            print(f"Original File Name: {item['original_file_name']}")
            print(f"Posture: {item['posture']}")
            print("-" * 30)

if __name__ == "__main__":
    # 设置你的 JSON 文件路径
    input_json_path = "../data/syrip/annotations/person_keypoints_train_infant.json"  # 替换为你的文件路径

    # 调用函数检查没有标签的数据
    check_labeled_data(input_json_path)
