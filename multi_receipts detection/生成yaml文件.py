import os

# 这是你原本脚本中定义的根目录
DATASET_ROOT = "/Users/baiwenbin/Documents/MyProjects/DeepLearning Project/work/multi_receipts detection/yolo_dataset"


def generate_yaml(dataset_path):
    # YAML 内容模板
    yaml_content = f"""
# 1. 数据集根目录绝对路径
path: {dataset_path}

# 2. 训练集和验证集的相对路径
train: images/train
val: images/val

# 3. 类别名称映射
names:
  0: receipt

# 4. 关键点配置 (这一行对于 17 值格式至关重要)
kpt_shape: [4, 2] # 代表 4 个关键点，每个点有 (x, y) 2 个坐标
"""

    yaml_path = os.path.join(dataset_path, "data.yaml")

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content.strip())

    print(f"✅ 成功生成 YOLO 配置文件: {yaml_path}")


# 调用生成
generate_yaml(DATASET_ROOT)