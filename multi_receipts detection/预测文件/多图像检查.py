import cv2
from ultralytics import YOLO
import os
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


# --- 手动 NMS 函数：这是解决重复框的终极办法 ---
def nms(boxes, scores, iou_threshold=0.3):
    if len(boxes) == 0: return []
    indices = np.argsort(scores)[::-1]
    keep = []
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        if len(indices) == 1: break

        # 计算 IoU
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


def visualize_with_slicing(input_dir, output_dir, model_path):
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=0.2,  # 初始门槛
        device='cpu'
    )

    os.makedirs(output_dir, exist_ok=True)
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))]

    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        h_orig, w_orig = img.shape[:2]

        # 预处理：缩小大图 (保持不变)
        MAX_DIM = 2000
        if max(h_orig, w_orig) > MAX_DIM:
            scale = MAX_DIM / max(h_orig, w_orig)
            img = cv2.resize(img, (int(w_orig * scale), int(h_orig * scale)))
            temp_path = os.path.join(output_dir, f"temp_{img_name}")
            cv2.imwrite(temp_path, img)
            infer_path = temp_path
        else:
            infer_path = img_path

        # 【核心修正】：只调用一次，且只保留基础参数
        result = get_sliced_prediction(
            infer_path,
            detection_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.25,
            overlap_width_ratio=0.25
        )

        # 提取数据进行 NMS 处理
        all_preds = result.object_prediction_list
        boxes = []
        scores = []
        for pred in all_preds:
            boxes.append([pred.bbox.minx, pred.bbox.miny, pred.bbox.maxx, pred.bbox.maxy])
            scores.append(pred.score.value)

        # 应用 NMS 过滤重复
        keep_indices = nms(np.array(boxes), np.array(scores), iou_threshold=0.3)

        print(f"--- {img_name} 检测到 {len(all_preds)} 个，去重后保留 {len(keep_indices)} 个 ---")

        # 遍历保留的框
        for i in keep_indices:
            box = boxes[i]
            x1, y1, x2, y2 = map(int, box)
            conf = scores[i]

            # 画图
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 4)
            cv2.putText(img, f"{conf:.2f}", (x1, y1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imwrite(os.path.join(output_dir, f"sliced_debug_{img_name}"), img)
        print(f"✅ 已保存: {img_name}")


visualize_with_slicing("./input_images", "./output_annotated", "best.pt")