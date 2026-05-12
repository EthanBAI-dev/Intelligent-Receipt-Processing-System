from PIL import Image, ImageDraw, ImageFont
import math, random, json
import numpy as np

# ====================== Homography 工具 ======================
def compute_homography(src_pts, dst_pts):
    A = []
    for (x, y), (u, v) in zip(src_pts, dst_pts):
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]

def apply_homography(pt, H):
    x, y = pt
    v = np.array([x, y, 1.0])
    r = H @ v
    return (r[0] / r[2], r[1] / r[2])

# ====================== ✅ 全局相机（归一化平面） ======================
def global_perspective_normalized(
    max_pitch_deg=15,
    max_yaw_deg=0,
    focal_len=1
):
    pitch = math.radians(random.uniform(-max_pitch_deg, 0))
    yaw   = math.radians(random.uniform(-max_yaw_deg, max_yaw_deg))

    # 统一归一化平面 [-0.5, 0.5]
    pts_3d = np.array([
        [-0.5, -0.5, 0],
        [ 0.5, -0.5, 0],
        [ 0.5,  0.5, 0],
        [-0.5,  0.5, 0],
    ])

    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch),  math.cos(pitch)],
    ])

    Ry = np.array([
        [ math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)],
    ])

    R = Ry @ Rx
    pts_cam = pts_3d @ R.T
    pts_cam[:, 2] += focal_len

    src = [(0,0), (1,0), (1,1), (0,1)]
    dst = []

    for X, Y, Z in pts_cam:
        x = X / Z + 0.5
        y = Y / Z + 0.5
        dst.append((x, y))

    return compute_homography(src, dst), pitch, yaw

def apply_global_to_receipt(w, h, H_global_norm):
    S_norm = np.array([
        [1/w, 0,   0],
        [0,   1/h, 0],
        [0,   0,   1],
    ])

    S_pix = np.array([
        [w, 0, 0],
        [0, h, 0],
        [0, 0, 1],
    ])

    return S_pix @ H_global_norm @ S_norm

# ====================== 局部轻扰动（非常弱） ======================
def local_perspective(w, h, max_offset=6):
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [
        (random.uniform(0, max_offset), random.uniform(0, max_offset)),
        (w + random.uniform(-max_offset, 0), random.uniform(0, max_offset)),
        (w + random.uniform(-max_offset, 0), h + random.uniform(-max_offset, 0)),
        (random.uniform(0, max_offset), h + random.uniform(-max_offset, 0)),
    ]
    return compute_homography(src, dst)

# ====================== 几何辅助 ======================
def rotate_and_translate(points, cx, cy, angle_rad):
    out = []
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    for x, y in points:
        rx = x * cos_a - y * sin_a + cx
        ry = x * sin_a + y * cos_a + cy
        out.append((rx, ry))
    return out

def bbox(points):
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs), max(ys)

# ====================== 配置 ======================
CANVAS_W, CANVAS_H = 2600, 1000
NUM_RECEIPTS = 4

W_RANGE = (220, 340)
H_RANGE = (600, 800)

ROT_RANGE = 3
SHIFT_Y = 25
SAFE_MARGIN = 30

OUT_IMAGE = "annotated_receipts_shared_camera.png"
OUT_JSON  = "receipt_annotations_shared_camera.json"

# ====================== 字体 ======================
try:
    FONT = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
except:
    FONT = ImageFont.load_default()

# ====================== 主流程 ======================
canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), "white")
draw = ImageDraw.Draw(canvas)

y_center = CANVAS_H // 2
MAX_W = W_RANGE[1]
SLOT_WIDTH = MAX_W + 2 * SAFE_MARGIN
TOTAL_WIDTH = SLOT_WIDTH * NUM_RECEIPTS
START_X = (CANVAS_W - TOTAL_WIDTH) // 2

annotations = []

# ✅ 只生成一次全局相机
H_global_norm, pitch, yaw = global_perspective_normalized()

for i in range(NUM_RECEIPTS):
    w = random.randint(*W_RANGE)
    h = random.randint(*H_RANGE)

    slot_left = START_X + i * SLOT_WIDTH
    slot_center_x = slot_left + SLOT_WIDTH // 2
    max_shift_x = (SLOT_WIDTH - w) // 2 - SAFE_MARGIN

    cx = slot_center_x + random.uniform(-max_shift_x, max_shift_x)
    cy = y_center + random.uniform(-SHIFT_Y, SHIFT_Y)

    angle = random.uniform(-ROT_RANGE, ROT_RANGE)
    a = math.radians(angle)

    # ✅ 同一相机映射到当前尺寸
    H_global = apply_global_to_receipt(w, h, H_global_norm)
    H_local  = local_perspective(w, h)

    H_total = H_global @ H_local

    src = [(0,0), (w,0), (w,h), (0,h)]
    warped = [apply_homography(p, H_total) for p in src]
    warped = [(x - w/2, y - h/2) for x, y in warped]

    SCALE_BACK = 1.12  # 1.08 ~ 1.15 之间


    final_corners = rotate_and_translate(warped, cx, cy, a)

    draw.polygon(final_corners, outline="black", width=3)

    corner_data = []
    for idx, (x, y) in enumerate(final_corners):
        r = 16
        draw.ellipse((x-r, y-r, x+r, y+r),
                     outline="red", fill="white", width=3)
        draw.text((x-8, y-13), str(idx+1),
                  fill="red", font=FONT)
        corner_data.append({"corner_id": idx+1, "x": x, "y": y})

    xmin, ymin, xmax, ymax = bbox(final_corners)
    draw.rectangle((xmin, ymin, xmax, ymax), outline="blue", width=3)

    annotations.append({
        "receipt_id": i,
        "size_px": {"width": w, "height": h},
        "rotation_deg": angle,
        "global_pitch_deg": math.degrees(pitch),
        "global_yaw_deg": math.degrees(yaw),
        "homography_total": H_total.tolist(),
        "corners_px": corner_data,
        "bbox_px": [xmin, ymin, xmax, ymax]
    })

canvas.save(OUT_IMAGE)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(annotations, f, indent=2, ensure_ascii=False)

print("✅ 完成：真正全局相机一致的小票合成")