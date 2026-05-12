import cv2
import numpy as np


# 找到最大轮廓
def biggest_contour(contours):
    biggest = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    return biggest, max_area


# 重新排序点 (左上0，右上1，左下2，右下3)
def reorder(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 2), dtype=np.float32)

    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)

    new_points[0] = points[np.argmin(s)]  # 左上
    new_points[3] = points[np.argmax(s)]  # 右下
    new_points[1] = points[np.argmin(diff)]  # 右上
    new_points[2] = points[np.argmax(diff)]  # 左下

    return new_points


# 画矩形
def draw_rectangle(img, points, thickness=2):
    if points is None:
        return
    points = points.astype(int)
    cv2.line(img, tuple(points[0]), tuple(points[1]), (0, 255, 0), thickness)
    cv2.line(img, tuple(points[1]), tuple(points[3]), (0, 255, 0), thickness)
    cv2.line(img, tuple(points[3]), tuple(points[2]), (0, 255, 0), thickness)
    cv2.line(img, tuple(points[2]), tuple(points[0]), (0, 255, 0), thickness)


# 主流程
def main():
    img = cv2.imread("4.jpg")

    if img is None:
        print("未找到图片，请检查路径。")
        return

    img_h, img_w = img.shape[:2]

    # --- Step 1: 图像预处理 ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)

    # --- Step 2: 找轮廓 ---
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_big_contours = img.copy()
    biggest, max_area = biggest_contour(contours)

    img_warp = None

    if biggest is not None and biggest.size != 0:
        biggest = reorder(biggest)
        draw_rectangle(img_big_contours, biggest, 10)

        # --- Step 3: 动态计算真实的宽高比例 (核心改动在这里) ---
        # 提取四个顶点
        tl = biggest[0]  # 左上 (Top-Left)
        tr = biggest[1]  # 右上 (Top-Right)
        bl = biggest[2]  # 左下 (Bottom-Left)
        br = biggest[3]  # 右下 (Bottom-Right)

        # 计算宽度：取 (左上到右上) 和 (左下到右下) 之间的最大距离
        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        max_width = max(int(width_top), int(width_bottom))

        # 计算高度：取 (左上到左下) 和 (右上到右下) 之间的最大距离
        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        max_height = max(int(height_left), int(height_right))

        # --- Step 4: 透视变换 ---
        pts1 = np.float32(biggest)
        # 使用动态计算出的 max_width 和 max_height 作为目标尺寸
        pts2 = np.float32([
            [0, 0],
            [max_width - 1, 0],
            [0, max_height - 1],
            [max_width - 1, max_height - 1]
        ])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_warp = cv2.warpPerspective(img, matrix, (max_width, max_height))
    else:
        print("未能识别到四个顶点的有效轮廓！")

    # --- 结果展示 ---
    scale_percent = 40
    disp_w = int(img_w * scale_percent / 100)
    disp_h = int(img_h * scale_percent / 100)

    img_show = cv2.resize(img_big_contours, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    cv2.imshow("Detected Area", img_show)

    if img_warp is not None:
        # 为了防止小票太长超出屏幕，如果高度大于屏幕高度，也可以把生成图稍微缩小一点展示
        warp_h, warp_w = img_warp.shape[:2]
        if warp_h > 900:  # 假设你的屏幕高度装不下 900 像素
            ratio = 900 / warp_h
            img_warp_show = cv2.resize(img_warp, (int(warp_w * ratio), int(warp_h * ratio)))
            cv2.imshow("Warped Image", img_warp_show)
        else:
            cv2.imshow("Warped Image", img_warp)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()