"""
v3 评估脚本 — 在 final_test 真实图片上运行推理并展示结果。

使用方法:
    python evaluate.py                  # 仅推理统计
    python evaluate.py --save           # 推理 + 保存可视化结果
    python evaluate.py --perspective    # 推理 + 裁剪 + 透视矫正 + 保存
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_v8_300.pt")
TEST_DIR = os.path.join(BASE_DIR, "final_test")
OUTPUT_DIR = os.path.join(BASE_DIR, "evaluate_output")

CLASS_NAMES = ['receipt', 'corner_tl', 'corner_tr', 'corner_br', 'corner_bl']
CLASS_COLORS = {
    0: (0, 255, 0),    # receipt - green
    1: (255, 0, 0),    # tl - red
    2: (0, 0, 255),    # tr - blue
    3: (255, 255, 0),  # br - yellow
    4: (0, 255, 255),  # bl - cyan
}

try:
    from ultralytics import YOLO
except ImportError:
    print("错误: 需要安装 ultralytics。运行: pip install ultralytics")
    sys.exit(1)


def load_font(size=16):
    for fp in ["/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def filter_receipts_by_aspect(detections, min_ratio=0.20, max_ratio=0.95):
    """过滤宽高比异常的 receipt 检测框。"""
    filtered = []
    removed = 0
    for cls_id, xyxy, conf in detections:
        if cls_id == 0:
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            ratio = w / h if h > 0 else 0
            if ratio < min_ratio or ratio > max_ratio:
                removed += 1
                continue
        filtered.append((cls_id, xyxy, conf))
    return filtered, removed


def group_corners_to_receipt(receipts, corners, x_range_margin=0.4):
    """将角点根据 X 坐标约束匹配到对应的 receipt 框。"""
    groups = {i: [] for i in range(len(receipts))}
    for cls_id, xyxy, conf in corners:
        cx = (xyxy[0] + xyxy[2]) / 2
        cy = (xyxy[1] + xyxy[3]) / 2
        min_dist = float('inf')
        best_idx = -1
        for i, (_, rb, _) in enumerate(receipts):
            rx_w = rb[2] - rb[0]
            if cx < rb[0] - rx_w * x_range_margin or cx > rb[2] + rx_w * x_range_margin:
                continue
            r_cx = (rb[0] + rb[2]) / 2
            r_cy = (rb[1] + rb[3]) / 2
            dist = (cx - r_cx) ** 2 + (cy - r_cy) ** 2
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        if best_idx >= 0:
            groups[best_idx].append((cls_id, (cx, cy), conf, xyxy))
    return groups


def parse_detect_result(r):
    """解析 YOLO Detect 结果。"""
    out = []
    if r.boxes is not None and len(r.boxes) > 0:
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for ci, xy, cf in zip(cls_ids, xyxy, confs):
            out.append((int(ci), xy.tolist(), float(cf)))
    return out


def inference_stats(show_details=True):
    """运行推理并输出统计。"""
    print(f"{'='*60}")
    print(f"模型: {os.path.basename(MODEL_PATH)}")
    print(f"测试图片: {TEST_DIR}")
    print(f"{'='*60}")

    if not os.path.exists(TEST_DIR):
        print(f"错误: 测试目录不存在 - {TEST_DIR}")
        return

    model = YOLO(MODEL_PATH)
    image_files = sorted([f for f in os.listdir(TEST_DIR)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"找到 {len(image_files)} 张测试图片\n")

    total_receipts = 0
    total_corners = 0

    for fname in image_files:
        img_path = os.path.join(TEST_DIR, fname)
        img = Image.open(img_path).convert("RGB")

        results = model.predict(source=img_path, save=False, conf=0.25,
                                iou=0.45, verbose=False)
        r = results[0]
        preds = parse_detect_result(r)
        receipts = [(ci, xy, cf) for ci, xy, cf in preds if ci == 0]
        receipts.sort(key=lambda x: x[1][0])
        corners = [(ci, xy, cf) for ci, xy, cf in preds if ci in (1, 2, 3, 4)]

        receipts, removed = filter_receipts_by_aspect(receipts)

        if show_details:
            print(f"[{fname}]")
            print(f"  原始检测: {len(preds)} 个框")
            print(f"  Receipt框: {len(receipts)} 张 (过滤掉 {removed} 个异常框)")
            print(f"  角点框:    {len(corners)} 个")

            for ri, (_, rb, cf) in enumerate(receipts):
                print(f"    票据 {ri+1}: conf={cf:.3f}, 位置=[{rb[0]:.0f},{rb[1]:.0f},{rb[2]:.0f},{rb[3]:.0f}]")

            # 角点匹配分组
            groups = group_corners_to_receipt(receipts, corners)
            for ri in groups:
                print(f"    票据 {ri+1} 匹配的角点: {len(groups[ri])} 个")
                for cid, (cx, cy), cf, xyxy in groups[ri]:
                    print(f"      {CLASS_NAMES[cid]}: ({cx:.0f},{cy:.0f}) conf={cf:.3f}")
            print()

        total_receipts += len(receipts)
        total_corners += len(corners)

    print(f"{'='*60}")
    print(f"总计: {len(image_files)} 张图, {total_receipts} 张票据, {total_corners} 个角点")
    print(f"平均: 每张图 {total_receipts/len(image_files):.1f} 张票据, "
          f"{total_corners/len(image_files):.1f} 个角点")
    print(f"{'='*60}")


def inference_with_save():
    """推理并保存标注结果。"""
    from tools.perspective_utils import rectify_receipt

    model = YOLO(MODEL_PATH)
    image_files = sorted([f for f in os.listdir(TEST_DIR)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    save_dir = os.path.join(OUTPUT_DIR, "annotated")
    crop_dir = os.path.join(OUTPUT_DIR, "cropped")
    rectify_dir = os.path.join(OUTPUT_DIR, "rectified")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(crop_dir, exist_ok=True)
    os.makedirs(rectify_dir, exist_ok=True)

    font = load_font(14)

    for fname in image_files:
        img_path = os.path.join(TEST_DIR, fname)
        img = Image.open(img_path).convert("RGB")
        w_img, h_img = img.size
        base_name = os.path.splitext(fname)[0]

        results = model.predict(source=img_path, save=False, conf=0.25,
                                iou=0.45, verbose=False)
        preds = parse_detect_result(results[0])
        receipts = [(ci, xy, cf) for ci, xy, cf in preds if ci == 0]
        receipts.sort(key=lambda x: x[1][0])
        receipts, _ = filter_receipts_by_aspect(receipts)
        corners = [(ci, xy, cf) for ci, xy, cf in preds if ci in (1, 2, 3, 4)]

        # 标注原始图
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)

        for ci, xyxy, cf in preds:
            color = CLASS_COLORS.get(ci, (255, 255, 255))
            label = f"{CLASS_NAMES[ci]} {cf:.2f}"
            draw.rectangle(xyxy, outline=color, width=2)
            draw.text((xyxy[0], max(0, xyxy[1] - 16)), label, fill=color, font=font)

        scale = min(1200 / w_img, 800 / h_img, 1.0)
        if scale < 1.0:
            new_size = (int(w_img * scale), int(h_img * scale))
            draw_img = draw_img.resize(new_size, Image.Resampling.LANCZOS)
        draw_img.save(os.path.join(save_dir, f"{base_name}_annotated.jpg"))

        # 裁剪和矫正
        groups = group_corners_to_receipt(receipts, corners)
        for ri, (_, rb, _) in enumerate(receipts):
            # 裁剪
            x1, y1, x2, y2 = [int(v) for v in rb]
            pad = 10
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(img.width, x2 + pad)
            y2 = min(img.height, y2 + pad)
            cropped = img.crop((x1, y1, x2, y2))

            # 灰色填充裁剪区域
            crop_padded = Image.new("RGB", (cropped.width + pad * 2, cropped.height + pad * 2),
                                    (128, 128, 128))
            crop_padded.paste(cropped, (pad, pad))
            crop_padded.save(os.path.join(crop_dir, f"{base_name}_receipt_{ri+1}.jpg"))

            # 透视矫正
            corners_with_class = [
                (x_y[0], x_y[1], cid)
                for cid, x_y, _, _ in groups[ri]
            ]
            if len(corners_with_class) >= 4:
                rectified, method = rectify_receipt(img, corners_with_class)
                if rectified is not None:
                    rectified.save(os.path.join(rectify_dir,
                                                f"{base_name}_receipt_{ri+1}_{method}.jpg"))

        print(f"[{fname}] 已保存: {save_dir}/")

    print(f"\n标注图片: {save_dir}/")
    print(f"裁剪图片: {crop_dir}/")
    print(f"矫正图片: {rectify_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v3 评估脚本")
    parser.add_argument("--save", action="store_true", help="保存标注结果")
    parser.add_argument("--perspective", action="store_true", help="裁剪 + 透视矫正")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型不存在 - {MODEL_PATH}")
        sys.exit(1)

    if args.perspective or args.save:
        inference_with_save()
    else:
        inference_stats(show_details=True)
