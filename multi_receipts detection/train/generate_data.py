"""
生成YOLOv8-Detect角点检测训练数据的合成图片脚本（第三版）。
功能：将多个票据图片通过透视变换、旋转后拼接到灰色背景画布上，
      每张票据标注5个框：1个receipt大框 + 4个角点小框（corner_tl/tr/br/bl），
      输出标准YOLOv8-Detect格式的训练/验证数据。
与第一版的区别：角点不再作为关键点（Pose），而是作为独立的检测目标（Detect）。
"""

import os
import glob
import math
import random
import numpy as np
from PIL import Image
import yaml


# ====================== 1. 透视变换工具函数 ======================

def compute_homography(src_pts, dst_pts):
    A = []
    for (x, y), (u, v) in zip(src_pts, dst_pts):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def get_pil_coeffs_from_homography(H):
    H_inv = np.linalg.inv(H)
    H_inv = H_inv / H_inv[2, 2]
    return tuple(H_inv.flatten()[:8])


def apply_homography(pt, H):
    x, y = pt
    v = np.array([x, y, 1.0])
    r = H @ v
    return (r[0] / r[2], r[1] / r[2])


def trim_image_with_offset(img):
    bbox = img.getbbox()
    if not bbox:
        return img, (0, 0)
    return img.crop(bbox), (bbox[0], bbox[1])


def get_bbox_from_corners(points):
    xs, ys = zip(*points)
    return [min(xs), min(ys), max(xs), max(ys)]


def global_perspective_normalized(max_pitch=12, max_yaw=12, focal_len=1.5):
    pitch = math.radians(random.uniform(-max_pitch, max_pitch))
    yaw = math.radians(random.uniform(-max_yaw, max_yaw))
    pts_3d = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]])
    Rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
    Ry = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    R = Ry @ Rx
    pts_cam = pts_3d @ R.T
    pts_cam[:, 2] += focal_len
    src = [(0, 0), (1, 0), (1, 1), (0, 1)]
    dst = [(X / Z + 0.5, Y / Z + 0.5) for X, Y, Z in pts_cam]
    return compute_homography(src, dst)


def local_perspective(w, h, max_offset=20):
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [
        (random.uniform(0, max_offset), random.uniform(0, max_offset)),
        (w + random.uniform(-max_offset, 0), random.uniform(0, max_offset)),
        (w + random.uniform(-max_offset, 0), h + random.uniform(-max_offset, 0)),
        (random.uniform(0, max_offset), h + random.uniform(-max_offset, 0)),
    ]
    return compute_homography(src, dst)


def rotate_points(points, angle_deg, center):
    angle = math.radians(-angle_deg)
    cx, cy = center
    rotated = []
    for x, y in points:
        x0, y0 = x - cx, y - cy
        xr = x0 * math.cos(angle) - y0 * math.sin(angle)
        yr = x0 * math.sin(angle) + y0 * math.cos(angle)
        rotated.append((xr + cx, yr + cy))
    return rotated


# ====================== 2. YOLOv8 Detect 标签格式 ======================

def to_yolo_detect_line(cls_id, bbox, canvas_w, canvas_h):
    """
    bbox: [xmin, ymin, xmax, ymax] 绝对像素坐标
    返回 YOLO 格式: class x_center y_center width height (归一化)
    """
    xmin, ymin, xmax, ymax = bbox
    xmin = max(0, min(xmin, canvas_w - 1))
    ymin = max(0, min(ymin, canvas_h - 1))
    xmax = max(0, min(xmax, canvas_w - 1))
    ymax = max(0, min(ymax, canvas_h - 1))

    x_center = ((xmin + xmax) / 2) / canvas_w
    y_center = ((ymin + ymax) / 2) / canvas_h
    width = (xmax - xmin) / canvas_w
    height = (ymax - ymin) / canvas_h

    return f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def corner_square_bbox(cx, cy, half_size):
    """
    以角点 (cx, cy) 为中心，生成小正方形 bbox
    返回: [xmin, ymin, xmax, ymax]
    """
    return [cx - half_size, cy - half_size, cx + half_size, cy + half_size]


# ====================== 3. 配置参数 ======================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "source_images")
DATASET_ROOT = os.path.join(BASE_DIR, "yolo_dataset")

NUM_SAMPLES = 10  # 示例：10 张合成图。正式训练改为 300
TRAIN_RATIO = 0.8
NUM_PER_IMG = 4
OVERLAP = 120

CORNER_SQUARE_RATIO = 2 / 15
CORNER_SQUARE_MIN = 40
CORNER_SQUARE_MAX = 90

CLASS_NAMES = ['receipt', 'corner_tl', 'corner_tr', 'corner_br', 'corner_bl']

# 收集源票据图片
IMAGE_FILES = []
for ext in ["*.JPG", "*.jpg", "*.png"]:
    IMAGE_FILES.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
random.shuffle(IMAGE_FILES)

print(f"找到 {len(IMAGE_FILES)} 张源票据图片")

# 创建目录
for split in ["train", "val"]:
    os.makedirs(os.path.join(DATASET_ROOT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(DATASET_ROOT, "labels", split), exist_ok=True)


# ====================== 4. 主流程：生成合成图 ======================

for i in range(NUM_SAMPLES):
    split = "train" if i < int(NUM_SAMPLES * TRAIN_RATIO) else "val"
    file_name = f"receipt_corners_{i:04d}"

    sample_files = random.sample(IMAGE_FILES, min(NUM_PER_IMG, len(IMAGE_FILES)))
    H_global = global_perspective_normalized()
    processed_imgs_data = []

    for f in sample_files:
        img = Image.open(f).convert("RGBA")
        orig_w, orig_h = img.size

        src_corners = [(0, 0), (orig_w, 0), (orig_w, orig_h), (0, orig_h)]

        angle = random.uniform(-6, 6) if random.random() < 0.7 else 0
        img = img.rotate(angle, expand=True)
        w, h = img.size

        cx, cy = orig_w / 2, orig_h / 2
        rotated_corners = rotate_points(src_corners, angle, (cx, cy))
        dx, dy = (w - orig_w) / 2, (h - orig_h) / 2
        src_corners = [(x + dx, y + dy) for x, y in rotated_corners]

        H_local = local_perspective(w, h, max_offset=w * 0.05)
        S_norm = np.array([[1 / w, 0, 0], [0, 1 / h, 0], [0, 0, 1]])
        S_pix = np.array([[w, 0, 0], [0, h, 0], [0, 0, 1]])
        H_total = (S_pix @ H_global @ S_norm) @ H_local

        warped_img_corners = [apply_homography(pt, H_total) for pt in [(0, 0), (w, 0), (w, h), (0, h)]]
        min_x = min(p[0] for p in warped_img_corners)
        min_y = min(p[1] for p in warped_img_corners)
        max_x = max(p[0] for p in warped_img_corners)
        max_y = max(p[1] for p in warped_img_corners)

        new_w = int(math.ceil(max_x - min_x))
        new_h = int(math.ceil(max_y - min_y))

        T = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
        H_total_translated = T @ H_total

        coeffs = get_pil_coeffs_from_homography(H_total_translated)
        transformed_img = img.transform((new_w, new_h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

        trimmed_img, offset = trim_image_with_offset(transformed_img)

        warped_corners = [apply_homography(pt, H_total_translated) for pt in src_corners]
        rel_corners = [(x - offset[0], y - offset[1]) for x, y in warped_corners]

        processed_imgs_data.append({
            "image": trimmed_img,
            "relative_corners": rel_corners,
        })

    total_w = sum(d["image"].width for d in processed_imgs_data) + (len(processed_imgs_data) - 1) * OVERLAP
    max_h = max(d["image"].height for d in processed_imgs_data)

    canvas = Image.new("RGBA", (total_w + 200, max_h + 200), (128, 128, 128, 255))
    canvas_w, canvas_h = canvas.size

    yolo_lines = []
    x_pos = 100
    y_center = canvas_h // 2

    for data in processed_imgs_data:
        img = data["image"]
        rel_corners = data["relative_corners"]
        y_pos = y_center - (img.height // 2)
        canvas.paste(img, (x_pos, y_pos), img)

        abs_corners = [(x + x_pos, y + y_pos) for x, y in rel_corners]

        receipt_bbox = get_bbox_from_corners(abs_corners)
        yolo_lines.append(to_yolo_detect_line(0, receipt_bbox, canvas_w, canvas_h))

        shortest_side = min(
            abs_corners[1][0] - abs_corners[0][0],
            abs_corners[3][1] - abs_corners[0][1],
        )
        half_size = max(CORNER_SQUARE_MIN, min(CORNER_SQUARE_MAX, int(shortest_side * CORNER_SQUARE_RATIO)))

        corner_classes = [1, 2, 3, 4]
        for cls_id, (cx, cy) in zip(corner_classes, abs_corners):
            corner_bbox = corner_square_bbox(cx, cy, half_size)
            yolo_lines.append(to_yolo_detect_line(cls_id, corner_bbox, canvas_w, canvas_h))

        x_pos += img.width + OVERLAP

    canvas_rgb = canvas.convert("RGB")
    canvas_rgb.save(os.path.join(DATASET_ROOT, "images", split, f"{file_name}.jpg"))

    with open(os.path.join(DATASET_ROOT, "labels", split, f"{file_name}.txt"), "w") as f:
        for line in yolo_lines:
            f.write(line + "\n")

    print(f"[{i+1}/{NUM_SAMPLES}] 已生成 {file_name}.jpg  (共 {len(yolo_lines)} 个标签)")

print(f"\n✅ 数据生成完成！")
print(f"   训练集: {int(NUM_SAMPLES * TRAIN_RATIO)} 张")
print(f"   验证集: {NUM_SAMPLES - int(NUM_SAMPLES * TRAIN_RATIO)} 张")
print(f"   每张包含: 4×5=20 个标签 (4个receipt大框 + 16个角点小框)")

# ====================== 5. 生成 data.yaml ======================

def generate_yaml(dataset_root):
    yaml_path = os.path.join(dataset_root, "data.yaml")
    config = {
        'path': dataset_root,
        'train': 'images/train',
        'val': 'images/val',
        'nc': 5,
        'names': ['receipt', 'corner_tl', 'corner_tr', 'corner_br', 'corner_bl'],
    }
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"\n✅ data.yaml 已生成: {yaml_path}")

generate_yaml(DATASET_ROOT)

# --- 自动压缩图片（最长边 ≤ 1280，匹配 YOLOv8 训练） ---
print("\n📐 正在压缩图片至最长边 1280px...")
resize_count = 0
for split in ["train", "val"]:
    img_dir = os.path.join(DATASET_ROOT, "images", split)
    for fname in os.listdir(img_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(img_dir, fname)
            img = Image.open(path)
            w, h = img.size
            if max(w, h) > 1280:
                scale = 1280 / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
                img.save(path, quality=90)
                resize_count += 1
print(f"   已压缩 {resize_count} 张图片")

print("\n🎉 全部完成！")
print(f"   📁 数据集根目录: {DATASET_ROOT}")
print(f"   📸 训练集图片:   {DATASET_ROOT}/images/train/")
print(f"   🏷️  YOLO标签:    {DATASET_ROOT}/labels/train/")
print(f"   📄 配置文件:     {DATASET_ROOT}/data.yaml")
print(f"\n" + "=" * 65)
print("  用 YOLOv8 训练命令:")
print(f"  yolo detect train data={DATASET_ROOT}/data.yaml model=yolov8n.pt epochs=200 imgsz=1280")
print("=" * 65)
