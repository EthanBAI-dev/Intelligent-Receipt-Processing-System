import cv2
import numpy as np
import os

# ================= 1. 配置路径 =================
# 这里填你上一步裁剪出来的图片所在的文件夹
INPUT_DIR = "./cropped_receipts"
# 这里填你最终想要保存矫正后图片的文件夹
OUTPUT_DIR = "./rectified_receipts"
# ===============================================

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

# 主流程：批量处理文件夹
def batch_rectify(input_dir, output_dir):
    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图片文件
    valid_exts = ('.jpg', '.jpeg', '.png')
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]

    if not images:
        print(f"❌ 在 {input_dir} 中没有找到图片，请检查路径。")
        return

    print(f"找到 {len(images)} 张待处理的图片，开始批量矫正...")

    # 遍历每张图片
    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"❌ 无法读取图片: {img_name}")
            continue

        # --- Step 1: 图像预处理 ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 1)
        edges = cv2.Canny(blur, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)

        # --- Step 2: 找轮廓 ---
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest, max_area = biggest_contour(contours)

        if biggest is not None and biggest.size != 0:
            biggest = reorder(biggest)

            # --- Step 3: 动态计算真实的宽高比例 ---
            tl, tr, bl, br = biggest[0], biggest[1], biggest[2], biggest[3]

            width_top = np.linalg.norm(tr - tl)
            width_bottom = np.linalg.norm(br - bl)
            max_width = max(int(width_top), int(width_bottom))

            height_left = np.linalg.norm(bl - tl)
            height_right = np.linalg.norm(br - tr)
            max_height = max(int(height_left), int(height_right))

            # --- Step 4: 透视变换 ---
            pts1 = np.float32(biggest)
            pts2 = np.float32([
                [0, 0],
                [max_width - 1, 0],
                [0, max_height - 1],
                [max_width - 1, max_height - 1]
            ])

            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            img_warp = cv2.warpPerspective(img, matrix, (max_width, max_height))

            # 保存成功的矫正图
            save_path = os.path.join(output_dir, f"rectified_{img_name}")
            cv2.imwrite(save_path, img_warp)
            print(f"  ✅ 成功矫正: {img_name}")

        else:
            # --- 容错机制 ---
            # 如果找不到四个角，就把未矫正的原图存过去，避免数据丢失
            print(f"  ⚠️ 警告: {img_name} 未找到四边形轮廓，跳过矫正。")
            save_path = os.path.join(output_dir, f"failed_rectify_{img_name}")
            cv2.imwrite(save_path, img)

    print("🎉 批量处理完成！")

if __name__ == "__main__":
    batch_rectify(INPUT_DIR, OUTPUT_DIR)