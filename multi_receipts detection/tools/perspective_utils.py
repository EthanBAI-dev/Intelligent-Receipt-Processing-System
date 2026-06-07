"""
共享透视矫正工具模块（加强版）。
统一入口 rectify_receipt()：输入裁切后的票据图片 + 角点信息（带class_id），
自动验证角点四边形的合法性，不合法时回退 OpenCV 轮廓法。
"""

import cv2
import numpy as np
from PIL import Image


def rectify_receipt(pil_img, corners_with_class=None):
    """
    统一的票据透视矫正函数。

    Args:
        pil_img: PIL Image (RGB)，裁切后的票据图片
        corners_with_class: list of (class_id, x, y)
                            class_id: 1=TL, 2=TR, 3=BR, 4=BL
                            为 None / 长度≠4 / 验证不通过 → 自动回退 OpenCV

    Returns:
        (rectified_pil_img, method_used)
        method_used: "corner"=角点法, "opencv"=OpenCV回退, "fallback"=返回原图
    """
    # 尝试角点法
    if corners_with_class and len(corners_with_class) == 4:
        result = _try_corner_rectify(pil_img, corners_with_class)
        if result is not None:
            return result, "corner"

    # 回退 OpenCV 轮廓法
    result = _rectify_opencv(pil_img)
    if result is not None and result != pil_img:
        return result, "opencv"

    # 最终回退：返回原图
    return pil_img, "fallback"


def _try_corner_rectify(pil_img, corners_with_class):
    """尝试用 class_id 角点做透视矫正，失败返回 None"""
    w_img, h_img = pil_img.width, pil_img.height

    # 1. 收集各角点坐标
    pts_dict = {}
    for ci, cx, cy in corners_with_class:
        ci = int(ci)
        if ci in (1, 2, 3, 4):
            pts_dict[ci] = (cx, cy)

    if set(pts_dict.keys()) != {1, 2, 3, 4}:
        return None

    tl = np.array(pts_dict[1], dtype=np.float32)
    tr = np.array(pts_dict[2], dtype=np.float32)
    br = np.array(pts_dict[3], dtype=np.float32)
    bl = np.array(pts_dict[4], dtype=np.float32)

    # 2. 验证：角点必须在图片内
    for pt_name, pt in [("TL", tl), ("TR", tr), ("BR", br), ("BL", bl)]:
        if pt[0] < 0 or pt[0] >= w_img or pt[1] < 0 or pt[1] >= h_img:
            return None

    # 3. 验证：四边形不自交（利用面积法）
    quad = np.float32([tl, tr, br, bl])
    area_full = cv2.contourArea(quad)
    # 分成两个三角形，检查面积
    area1 = cv2.contourArea(np.float32([tl, tr, br]))
    area2 = cv2.contourArea(np.float32([tl, br, bl]))
    if abs(area1 + area2 - area_full) > max(area_full * 0.1, 10):
        return None

    # 4. 验证：四边形面积非零且不过小
    if area_full < 100:
        return None

    # 5. 验证：长宽比不过于极端
    max_w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    max_h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if max_w < 30 or max_h < 30:
        return None
    aspect = max(max_w / (max_h + 1), max_h / (max_w + 1))
    if aspect > 8:
        return None

    # 6. 透視变换
    max_w = int(max_w)
    max_h = int(max_h)

    src = np.float32([tl, tr, br, bl])
    dst = np.float32([[0, 0],
                      [max_w - 1, 0],
                      [max_w - 1, max_h - 1],
                      [0, max_h - 1]])

    img = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_bgr, M, (max_w, max_h))

    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))


def _rectify_opencv(pil_img):
    """OpenCV 轮廓法"""
    img = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h_img, w_img = img_bgr.shape[:2]
    total_area = h_img * w_img

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 1)

    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 5)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = _find_quads(contours, total_area * 0.15)

    if not candidates:
        edges = cv2.Canny(blur, 30, 100)
        edges = cv2.dilate(edges, kernel, iterations=3)
        edges = cv2.erode(edges, kernel, iterations=1)
        contours2, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = _find_quads(contours2, total_area * 0.10)

    if not candidates:
        return pil_img

    candidates.sort(key=lambda x: -x[0])
    pts = candidates[0][1]

    tl, tr, br, bl = pts[0], pts[1], pts[2], pts[3]
    max_w = max(int(np.linalg.norm(tr - tl)), int(np.linalg.norm(br - bl)))
    max_h = max(int(np.linalg.norm(bl - tl)), int(np.linalg.norm(br - tr)))
    max_w = max(max_w, 100)
    max_h = max(max_h, 100)

    dst = np.float32([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]])
    M = cv2.getPerspectiveTransform(np.float32(pts), dst)
    warped = cv2.warpPerspective(img_bgr, M, (max_w, max_h))
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))


def _find_quads(contours, min_area):
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            pts = _order_corners(approx.reshape(4, 2))
            candidates.append((area, pts))
    return candidates


def _order_corners(pts):
    """x+y 最小=TL, 最大=BR；剩余中x大=TR，x小=BL"""
    pts = np.array(pts)
    sums = pts.sum(axis=1)
    tl_idx = int(np.argmin(sums))
    br_idx = int(np.argmax(sums))
    remaining = [i for i in range(4) if i not in (tl_idx, br_idx)]
    if pts[remaining[0]][0] > pts[remaining[1]][0]:
        tr_idx, bl_idx = remaining[0], remaining[1]
    else:
        tr_idx, bl_idx = remaining[1], remaining[0]
    return pts[[tl_idx, tr_idx, br_idx, bl_idx]]
