import cv2
import os
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ================= 1. 配置路径 =================
INPUT_DIR = "./input_images"
OUTPUT_DIR = "./cropped_receipts"
MODEL_PATH = "best.pt"


# --- 手动 NMS 函数：过滤重复框 ---
def nms(boxes, scores, iou_threshold=0.3):
    if len(boxes) == 0: return []
    indices = np.argsort(scores)[::-1]
    keep = []
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        if len(indices) == 1: break

        x1 = np.maximum(boxes[current, 0], boxes[indices[1:], 0])
        y1 = np.maximum(boxes[current, 1], boxes[indices[1:], 1])
        x2 = np.minimum(boxes[current, 2], boxes[indices[1:], 2])
        y2 = np.minimum(boxes[current, 3], boxes[indices[1:], 3])
        w = np.maximum(0, x2 - x1)
        h = np.maximum(0, y2 - y1)
        intersection = w * h
        area = (boxes[indices[1:], 2] - boxes[indices[1:], 0]) * (boxes[indices[1:], 3] - boxes[indices[1:], 1])
        iou = intersection / (area + (boxes[current, 2] - boxes[current, 0]) * (
                    boxes[current, 3] - boxes[current, 1]) - intersection)

        indices = indices[1:][iou < iou_threshold]
    return keep


def detect_crop_and_pad(input_dir, output_dir, model_path):
    # 加载 SAHI 模型
    print("正在加载 SAHI 模型...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=0.2,  # 初始门槛
        device='cpu'
    )

    os.makedirs(output_dir, exist_ok=True)
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if not images:
        print(f"❌ 错误：在 {input_dir} 中未找到图片")
        return

    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        h_orig, w_orig = img.shape[:2]
        base_name = os.path.splitext(img_name)[0]

        # --- 预处理：缩小大图 ---
        MAX_DIM = 2000
        if max(h_orig, w_orig) > MAX_DIM:
            scale = MAX_DIM / max(h_orig, w_orig)
            img = cv2.resize(img, (int(w_orig * scale), int(h_orig * scale)))
            temp_path = os.path.join(output_dir, f"temp_{img_name}")
            cv2.imwrite(temp_path, img)
            infer_path = temp_path
        else:
            infer_path = img_path

        # --- 执行 SAHI 切片推理 ---
        result = get_sliced_prediction(
            infer_path,
            detection_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.25,
            overlap_width_ratio=0.25
        )

        # 提取框并进行 NMS 去重
        all_preds = result.object_prediction_list
        boxes, scores = [], []
        for pred in all_preds:
            boxes.append([pred.bbox.minx, pred.bbox.miny, pred.bbox.maxx, pred.bbox.maxy])
            scores.append(pred.score.value)

        keep_indices = nms(np.array(boxes), np.array(scores), iou_threshold=0.3)

        # 提取保留下来的框
        final_boxes = [boxes[i] for i in keep_indices]

        # 【极其重要】按 x1 坐标从小到大排序，保证输出文件是从左到右的 1, 2, 3, 4
        final_boxes.sort(key=lambda b: b[0])

        print(f"--- {img_name} 检测并去重后保留 {len(final_boxes)} 张小票，开始裁剪 ---")

        for idx, box in enumerate(final_boxes):
            x1, y1, x2, y2 = map(int, box)

            # 边界保护，防止框超出图片范围报错
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

            # 1. 精确裁剪：完全按照模型识别的框来切，一点多余的原图都不带
            exact_crop = img[y1:y2, x1:x2]

            # 容错：如果框的宽高为0则跳过
            if exact_crop.size == 0:
                continue

            # 2. 灰色填充扩充：使用 cv2.copyMakeBorder 给图片加上灰色的边框
            # pad 像素数可以根据需要调整，10~20 都可以
            pad = 10
            # BGR 格式的灰色: (128, 128, 128)
            gray_color = (128, 128, 128)

            padded_crop = cv2.copyMakeBorder(
                exact_crop,
                top=pad, bottom=pad, left=pad, right=pad,
                borderType=cv2.BORDER_CONSTANT,
                value=gray_color
            )

            # 保存加上灰底边的图片
            save_name = f"{base_name}_receipt_{idx + 1}.jpg"
            save_path = os.path.join(output_dir, save_name)
            cv2.imwrite(save_path, padded_crop)
            print(f"  ✅ 已裁剪并加灰色边框: {save_name}")


if __name__ == "__main__":
    detect_crop_and_pad(INPUT_DIR, OUTPUT_DIR, MODEL_PATH)