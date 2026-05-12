import os
from PIL import Image


def batch_resize(image_dir, max_long_edge=1280):
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, file)
                img = Image.open(path)
                w, h = img.size

                # 如果图片比 1280 大，才进行压缩
                if max(w, h) > max_long_edge:
                    scale = max_long_edge / max(w, h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    # 使用 LANCZOS 高质量重采样
                    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    img.save(path, quality=90)
                    print(f"已压缩: {file} -> {new_w}x{new_h}")


# 把路径换成你的 images 文件夹
batch_resize(
    "/Users/baiwenbin/Documents/MyProjects/DeepLearning Project/work/multi_receipts detection/yolo_dataset/images/train")
batch_resize(
    "/Users/baiwenbin/Documents/MyProjects/DeepLearning Project/work/multi_receipts detection/yolo_dataset/images/val")