import os
from PIL import Image


def batch_rotate_images(input_dir, output_dir, angle=90):
    # 如果输出文件夹不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 支持的图片格式
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    count = 0

    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith(valid_extensions):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            try:
                with Image.open(input_path) as img:
                    # 逆时针旋转，expand=True 确保旋转后图像不被裁剪
                    rotated_img = img.rotate(angle, expand=True)

                    # 统一保存，如果是 JPG 建议保持高质量
                    rotated_img.save(output_path, quality=95)

                count += 1
                print(f"已处理: {file_name}")
            except Exception as e:
                print(f"处理 {file_name} 出错: {e}")

    print(f"\n======================")
    print(f"任务完成！总计处理: {count} 张图片")
    print(f"保存至: {output_dir}")
    print(f"======================")


# =========================
# 路径设置
# =========================
# 建议使用你在 PyCharm 项目中的实际路径
input_folder = "/Users/baiwenbin/Documents/MyProjects/DeepLearning Project/work/some file"
output_folder = "/Users/baiwenbin/Documents/MyProjects/DeepLearning Project/work/some file"

batch_rotate_images(input_folder, output_folder)