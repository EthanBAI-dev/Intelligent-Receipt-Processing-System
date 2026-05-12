from inference_sdk import InferenceHTTPClient
from PIL import Image
import os
import json

# ===== 配置 =====
INPUT_DIR = "/Users/baiwenbin/Documents/MyProjects/DeepLearning Project/work/processed data2"
OUTPUT_DIR = "output"
API_KEY = "zvFWvtuVgoJHzpG0tFp0"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== 初始化 API =====
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)

# ===== 工具函数 =====
def yolo_to_xyxy(box):
    x, y, w, h = box['x'], box['y'], box['width'], box['height']
    return int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)

# ===== 主循环 =====
for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(INPUT_DIR, filename)
    print(f"处理: {filename}")

    try:
        # 1. 调用 API
        result = client.run_workflow(
            workspace_name="ras-workspace-uhnsw",
            workflow_id="custom-workflow",
            images={"image": image_path},
            use_cache=True
        )

        preds = result[0]['predictions']['predictions']

        # 2. 找 receipt
        receipt = next((p for p in preds if p['class'] == 'receipt'), None)
        if receipt is None:
            print("  ⚠️ 未检测到 receipt，跳过")
            continue

        # 3. 裁剪
        image = Image.open(image_path)
        rx1, ry1, rx2, ry2 = yolo_to_xyxy(receipt)
        cropped = image.crop((rx1, ry1, rx2, ry2))

        # 保存裁剪图
        base_name = os.path.splitext(filename)[0]
        cropped_path = os.path.join(OUTPUT_DIR, f"{base_name}_crop.jpg")
        cropped.save(cropped_path)

        # 4. 转换其他框
        annotations = []
        for p in preds:
            if p['class'] == 'receipt':
                continue

            x1, y1, x2, y2 = yolo_to_xyxy(p)

            # 坐标转换
            x1 -= rx1
            y1 -= ry1
            x2 -= rx1
            y2 -= ry1

            # 边界保护
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(rx2 - rx1, x2), min(ry2 - ry1, y2)

            annotations.append({
                "class": p['class'],
                "bbox": [x1, y1, x2, y2],
                "confidence": p['confidence']
            })

        # 保存 JSON
        json_path = os.path.join(OUTPUT_DIR, f"{base_name}.json")
        with open(json_path, "w") as f:
            json.dump(annotations, f, indent=2)

    except Exception as e:
        print(f"  ❌ 出错: {e}")

print("✅ 全部处理完成")