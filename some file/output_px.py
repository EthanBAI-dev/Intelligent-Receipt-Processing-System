import os
from PIL import Image

input_folder = "/Users/baiwenbin/Documents/MyProjects/DeepLearning Project/work/DATA FILE/VOC2007/JPEGImages"  # 图片文件夹

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        path = os.path.join(input_folder, filename)

        img = Image.open(path)
        width, height = img.size

        print(f"{filename}: {width}*{height}")