import os

def check_labels(directory):
    for filename in os.listdir(directory):
        if not filename.endswith(".txt"): continue
        file_path = os.path.join(directory, filename)
        with open(file_path, "r") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if not parts: continue # 跳过空行
                if len(parts) != 17:
                    print(f"❌ 发现问题: {filename} 的第 {idx+1} 行")
                    print(f"   内容: {line.strip()}")
                    print(f"   长度: {len(parts)} (应为 17)")

print("--- 检查训练集 ---")
check_labels("/content/yolov8_dataset/labels/train")
print("--- 检查验证集 ---")
check_labels("/content/yolov8_dataset/labels/val")