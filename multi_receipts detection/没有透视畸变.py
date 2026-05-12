from PIL import Image, ImageDraw, ImageFont
import math, random, json

# ====================== 配置 ======================
CANVAS_W, CANVAS_H = 2600, 1000
NUM_RECEIPTS = 4

# 小票尺寸范围（真实比例）
W_RANGE = (220, 340)
H_RANGE = (600, 800)

ROT_RANGE = 4        # ±旋转角度
SHIFT_Y = 25         # Y 方向偏移（安全）
SAFE_MARGIN = 30     # 槽位内安全边界

OUT_IMAGE = "annotated_receipts_safe！！.png"
OUT_JSON = "receipt_corner_annotations_safe！！.json"

# ====================== 字体 ======================
try:
    FONT = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
except:
    FONT = ImageFont.load_default()

# ====================== 几何函数 ======================
def rotate_rect(cx, cy, w, h, deg):
    a = math.radians(deg)
    pts = [(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)]
    return [
        (x*math.cos(a)-y*math.sin(a)+cx,
         x*math.sin(a)+y*math.cos(a)+cy)
        for x, y in pts
    ]

def bbox(points):
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs), max(ys)

# ====================== 主流程 ======================
canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), "white")
draw = ImageDraw.Draw(canvas)

y_center = CANVAS_H // 2

# ✅ 槽位设计（关键）
MAX_W = W_RANGE[1]
SLOT_WIDTH = MAX_W + 2 * SAFE_MARGIN
TOTAL_WIDTH = SLOT_WIDTH * NUM_RECEIPTS
START_X = (CANVAS_W - TOTAL_WIDTH) // 2

annotations = []

for i in range(NUM_RECEIPTS):
    w = random.randint(*W_RANGE)
    h = random.randint(*H_RANGE)

    slot_left = START_X + i * SLOT_WIDTH
    slot_center_x = slot_left + SLOT_WIDTH // 2

    # ✅ X 偏移自动安全限制
    max_shift_x = (SLOT_WIDTH - w) // 2 - SAFE_MARGIN
    shift_x = random.uniform(-max_shift_x, max_shift_x)

    cx = slot_center_x + shift_x
    cy = y_center + random.uniform(-SHIFT_Y, SHIFT_Y)
    angle = random.uniform(-ROT_RANGE, ROT_RANGE)

    corners = rotate_rect(cx, cy, w, h, angle)

    # —— 小票轮廓
    draw.polygon(corners, outline="black", width=3)

    # —— 角点（圆圈 + 数字）
    corner_data = []
    for idx, (x, y) in enumerate(corners):
        r = 16
        draw.ellipse((x-r, y-r, x+r, y+r),
                     outline="red", fill="white", width=3)
        draw.text((x-8, y-13), str(idx+1),
                  fill="red", font=FONT)

        corner_data.append({
            "corner_id": idx + 1,
            "x": x,
            "y": y
        })

    # —— 外接水平框
    xmin, ymin, xmax, ymax = bbox(corners)
    draw.rectangle((xmin, ymin, xmax, ymax),
                   outline="blue", width=3)

    annotations.append({
        "receipt_id": i,
        "size_px": {"width": w, "height": h},
        "rotation_deg": angle,
        "corners_px": corner_data,
        "bbox_px": [xmin, ymin, xmax, ymax]
    })

canvas.save(OUT_IMAGE)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(annotations, f, indent=2, ensure_ascii=False)

print("✅ 完成（保证不重叠）")