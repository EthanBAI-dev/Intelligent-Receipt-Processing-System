import os
import glob
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import shutil


# ====================== 1. 工具函数 (保留核心数学逻辑，新增标注工具) ======================
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
    """前向 H -> PIL 逆透视系数"""
    H_inv = np.linalg.inv(H)
    H_inv = H_inv / H_inv[2, 2]
    return tuple(H_inv.flatten()[:8])


def apply_homography(pt, H):
    """将 H 矩阵应用到单个点"""
    x, y = pt
    v = np.array([x, y, 1.0])
    r = H @ v
    return (r[0] / r[2], r[1] / r[2])


def trim_image_with_offset(img):
    """✅ 新增：裁剪透明边框，并返回裁剪的 (xmin, ymin) 偏移量"""
    bbox = img.getbbox()
    if not bbox:
        return img, (0, 0)
    # bbox 格式为 (xmin, ymin, xmax, ymax)
    return img.crop(bbox), (bbox[0], bbox[1])


def get_bbox_from_corners(points):
    """✅ 新增：根据角点计算 Axis-Aligned BBox"""
    xs, ys = zip(*points)
    return [min(xs), min(ys), max(xs), max(ys)]


def global_perspective_normalized(max_pitch=10, max_yaw=10, focal_len=1.5):
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
    # ❗ 必须加负号：让数学计算的顺时针转换为和 PIL 同步的逆时针！
    angle = math.radians(-angle_deg)
    cx, cy = center

    rotated = []
    for x, y in points:
        x0 = x - cx
        y0 = y - cy
        xr = x0 * math.cos(angle) - y0 * math.sin(angle)
        yr = x0 * math.sin(angle) + y0 * math.cos(angle)
        rotated.append((xr + cx, yr + cy))

    return rotated

# ====================== 2. 配置与字体 ======================
INPUT_DIR = "/Users/baiwenbin/Documents/MyProjects/DeepLearning Project/work/data/"
DATASET_ROOT = "/Users/baiwenbin/Documents/MyProjects/DeepLearning Project/work/multi_receipts detection/data"
os.makedirs(DATASET_ROOT, exist_ok=True)
IMAGE_FILES = [f for ext in ["*.JPG", "*.jpg", "*.png"] for f in glob.glob(os.path.join(INPUT_DIR, ext))]

# 用于标注的大字体，DejaVuSans-Bold 在高分屏上清晰
try:
    FONT = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
except:
    FONT = ImageFont.load_default()

# 参数配置
NUM_SAMPLES = 10
TRAIN_RATIO = 0.8
NUM_PER_IMG = 4
OVERLAP = 20
DEBUG_DRAW = True # 生成数据集时建议设为 False

# 目录创建
for split in ["train", "val"]:
    os.makedirs(os.path.join(DATASET_ROOT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(DATASET_ROOT, "labels", split), exist_ok=True)

def to_yolo_format(bbox, canvas_w, canvas_h):
    """
    纯粹负责坐标转换：Converts [xmin, ymin, xmax, ymax] to YOLO [class, x_center, y_center, w, h]
    """
    xmin, ymin, xmax, ymax = bbox
    # Normalize coordinates
    x_center = ((xmin + xmax) / 2) / canvas_w
    y_center = ((ymin + ymax) / 2) / canvas_h
    width = (xmax - xmin) / canvas_w
    height = (ymax - ymin) / canvas_h
    return [0, x_center, y_center, width, height] # class_id 0


# ====================== 3. 主流程 ======================
for i in range(NUM_SAMPLES):
    # 确定当前属于训练集还是验证集
    split = "train" if i < int(NUM_SAMPLES * TRAIN_RATIO) else "val"
    file_name = f"receipt_{i:04d}"

    sample_files = random.sample(IMAGE_FILES, NUM_PER_IMG)
    H_global = global_perspective_normalized()
    processed_imgs_data = []

    # 直接使用原始分辨率处理
    # 直接使用原始分辨率处理
    for f in sample_files:
        img = Image.open(f).convert("RGBA")
        orig_w, orig_h = img.size
        src_corners = [(0, 0), (orig_w, 0), (orig_w, orig_h), (0, orig_h)]  # 原始角点

        # 随机旋转图像
        angle = random.uniform(-4, 4) if random.random() < 0.7 else 0
        img = img.rotate(angle, expand=True)
        w, h = img.size

        # 旋转角点逻辑
        cx, cy = orig_w / 2, orig_h / 2
        rotated_corners = rotate_points(src_corners, angle, (cx, cy))
        dx, dy = (w - orig_w) / 2, (h - orig_h) / 2
        src_corners = [(x + dx, y + dy) for x, y in rotated_corners]

        H_local = local_perspective(w, h, max_offset=w * 0.05)
        S_norm = np.array([[1 / w, 0, 0], [0, 1 / h, 0], [0, 0, 1]])
        S_pix = np.array([[w, 0, 0], [0, h, 0], [0, 0, 1]])
        H_total = (S_pix @ H_global @ S_norm) @ H_local

        # =========================================================
        # ❗ 核心修复点：预判图像透视后的真实边界，生成平移矩阵防止裁剪
        # =========================================================
        # 1. 算出完整画布的 4 个角落被透视到了哪里
        img_corners = [(0, 0), (w, 0), (w, h), (0, h)]
        warped_img_corners = [apply_homography(pt, H_total) for pt in img_corners]

        # 2. 找到极值，这决定了新画布究竟需要多大，以及要往右下方推多少
        min_x = min(p[0] for p in warped_img_corners)
        min_y = min(p[1] for p in warped_img_corners)
        max_x = max(p[0] for p in warped_img_corners)
        max_y = max(p[1] for p in warped_img_corners)

        new_w = int(math.ceil(max_x - min_x))
        new_h = int(math.ceil(max_y - min_y))

        # 3. 构造平移矩阵 T，把最小 x,y 拉回 0,0
        T = np.array([
            [1, 0, -min_x],
            [0, 1, -min_y],
            [0, 0, 1]
        ])

        # 4. 把平移合并进总透视矩阵中！
        H_total_translated = T @ H_total

        # =========================================================

        # 5. 用带平移的矩阵去获取系数，并赋予图片全新的宽和高 (new_w, new_h)
        coeffs = get_pil_coeffs_from_homography(H_total_translated)
        transformed_img = img.transform(
            (new_w, new_h),  # ✅ 使用扩展后的安全画布尺寸，再也不会被切了
            Image.PERSPECTIVE,
            coeffs,
            Image.BICUBIC
        )

        # 因为此时图片四边可能有多余透明像素，重新 Trim 一下
        trimmed_img, offset = trim_image_with_offset(transformed_img)

        # 6. 对角点应用同样的带平移的矩阵！这保证了数学坐标和像素绝对同步
        warped_corners = [apply_homography(pt, H_total_translated) for pt in src_corners]
        rel_corners = [(x - offset[0], y - offset[1]) for x, y in warped_corners]

        processed_imgs_data.append({
            "image": trimmed_img,
            "relative_corners": rel_corners
        })

    # 动态计算画布大小
    total_w = sum(d["image"].width for d in processed_imgs_data) + (len(processed_imgs_data) - 1) * OVERLAP
    max_h = max(d["image"].height for d in processed_imgs_data)

    # 创建画布，灰色背景
    canvas = Image.new("RGBA", (total_w + 200, max_h + 200), (128, 128, 128, 255))
    draw = ImageDraw.Draw(canvas)
    canvas_w, canvas_h = canvas.size

    yolo_labels = []
    x_pos = 100
    y_center = canvas.height // 2

    # 粘贴与 ✅ 绘制标注
    x_pos = 100
    y_center = (canvas.height) // 2

    for data in processed_imgs_data:
        img = data["image"]
        rel_corners = data["relative_corners"]
        y_pos = y_center - (img.height // 2)
        canvas.paste(img, (x_pos, y_pos), img)# 粘贴小票图片

        # ✅ 计算画布绝对角点坐标 (相对坐标 + 粘贴起始点)
        abs_corners = [(x + x_pos, y + y_pos) for x, y in rel_corners]

        # 保存标签
        xmin, ymin, xmax, ymax = get_bbox_from_corners(abs_corners)

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(canvas.width, xmax)
        ymax = min(canvas.height, ymax)

        bbox = [xmin, ymin, xmax, ymax]
        label = to_yolo_format(bbox, canvas.width, canvas.height)
        yolo_labels.append(label)

        # ✅ 绘制多边形轮廓 (青色)
        if DEBUG_DRAW:
            draw.polygon(abs_corners, outline=(0, 255, 255), width=8)

        # ✅ 绘制角点圆圈和序号 (红色)
        for idx, (px, py) in enumerate(abs_corners):
            r = 20  # 圆半径
            # 画圆
            if DEBUG_DRAW:
                draw.ellipse((px - r, py - r, px + r, py + r),
                             outline=(255, 0, 0), fill=(255, 255, 255), width=4)

            # ✅ 数字（自动居中）
            text = str(idx + 1)
            bbox_text = draw.textbbox((0, 0), text, font=FONT)
            tw = bbox_text[2] - bbox_text[0]
            th = bbox_text[3] - bbox_text[1]

            if DEBUG_DRAW:
                draw.text(
                    (px - tw / 2, py - th / 2),
                    text,
                    fill=(255, 0, 0),
                    font=FONT
                )

        # ✅ 绘制 Axis-Aligned BBox (蓝色)
        bbox = get_bbox_from_corners(abs_corners)
        if DEBUG_DRAW:
            draw.rectangle(bbox, outline=(0, 0, 255), width=6)

        # 更新下一个坐标
        x_pos += img.width + OVERLAP

    # Save Image
    canvas.convert("RGB").save(os.path.join(DATASET_ROOT, "images", split, f"{file_name}.jpg"))

    # Save Label (.txt)
    with open(os.path.join(DATASET_ROOT, "labels", split, f"{file_name}.txt"), "w") as f:
        for label in yolo_labels:
            f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

    print(f"[{split}] 已生成 {file_name}.jpg")