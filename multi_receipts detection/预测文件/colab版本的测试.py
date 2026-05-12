from ultralytics import YOLO
import os

# --- 1. 设置路径 (请将下面的路径改为你电脑上的真实路径) ---
# 输入文件夹
INPUT_DIR = "./input_images"
# 输出文件夹
OUTPUT_DIR = "/Users/baiwenbin/Documents/MyProjects/DeepLearning Project/work/multi_receipts detection/预测文件/colab版本预测结果"
# 模型路径
MODEL_PATH = "best.pt"


def batch_predict():
    # 2. 加载模型
    model = YOLO(MODEL_PATH)

    # 3. 执行批量预测
    # save=True: 保存画了框的图片
    # project: 指定输出项目的根目录
    # name: 指定输出的文件夹名称
    # imgsz=1280: 强制大分辨率输入，确保能看清票据
    # conf=0.5: 置信度门槛
    results = model.predict(
        source=INPUT_DIR,
        save=True,
        project=OUTPUT_DIR,
        name='predictions',
        imgsz=1280,
        conf=0.5
    )

    print(f"✅ 预测完成！结果已保存在: {OUTPUT_DIR}/predictions")


if __name__ == "__main__":
    batch_predict()