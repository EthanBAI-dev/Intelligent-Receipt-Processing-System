"""
OCR 识别模块 (PaddleOCR 版本)。
全局单例 PaddleOCR 实例，支持日文+英文识别。
四类业务区域对应关系（由 YOLOv5 检测模型定义）：
  class_0 = 日期 (Date)
  class_1 = 店名 (Store Name)
  class_2 = 電話番号 (Phone)
  class_3 = 合計金額 (Total Amount, 含￥符号)
"""

import os
import re
import pandas as pd
import cv2
import ssl

# PaddleOCR 缓存路径（避免 sandbox 权限问题）
os.environ.setdefault('PADDLEX_HOME',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '.paddlex_cache'))

from paddleocr import PaddleOCR

ssl._create_default_https_context = ssl._create_unverified_context

# ====================== 全局单例 OCR 实例 ======================
_ocr_instance = None


def get_ocr():
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = PaddleOCR(
            lang='japan',
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
    return _ocr_instance


# ====================== 四类业务专属文本清洗 ======================

def clean_date_text(text):
    """class_0 = 日期。仅去除首尾空格，PaddleOCR 自身对日文日期识别已经非常准确。"""
    return text.strip()


def clean_store_text(text):
    """class_1 = 店名。保留日文假名、汉字、英文字母、数字。"""
    text = text.strip()
    text = re.sub(r'[^\w\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff  \-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_phone_text(text):
    """class_2 = 電話番号。保留数字、-、()、.、:、TEL 字样。"""
    text = text.strip()
    text = re.sub(r'[^\d\-\(\)\.: TELtel]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_total_text(text):
    """class_3 = 合計金額。保留数字、￥¥$、逗号、句点、合計/合计文字。"""
    text = text.strip()
    text = re.sub(r'[^\d¥￥$€,.  \-合計合计]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# class_id → (CSV列名, 清洗函数)
CLASS_INFO = {
    0: ("date",   clean_date_text),
    1: ("store",  clean_store_text),
    2: ("phone",  clean_phone_text),
    3: ("total",  clean_total_text),
}


# ====================== CSV 文件命名 ======================

def get_sequential_csv_name(base_name="ocr_results", directory=".", extension=".csv"):
    counter = 1
    while True:
        file_name = f"{base_name}{counter}{extension}"
        file_path = os.path.join(directory, file_name)
        if not os.path.exists(file_path):
            return file_path
        counter += 1


# ====================== 核心 OCR 流程 ======================

def perform_ocr_on_folder(label_dir, image_dir, output_csv_base):
    ocr = get_ocr()
    results = []

    if not os.path.exists(label_dir):
        print(f"Label directory does not exist: {label_dir}")
        return None
    if not os.path.exists(image_dir):
        print(f"Image directory does not exist: {image_dir}")
        return None

    for label_file in sorted(os.listdir(label_dir)):
        if not label_file.endswith(".txt"):
            continue

        image_name = label_file.replace(".txt", ".jpg")
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, label_file)

        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        ocr_results = {"image_name": image_name}
        for cname, _ in CLASS_INFO.values():
            ocr_results[cname] = ""
        h_original, w_original = image.shape[:2]

        for line in lines:
            components = line.strip().split()
            if len(components) < 5:
                continue

            class_id = int(components[0])
            if class_id not in CLASS_INFO:
                continue

            cname, cleaner = CLASS_INFO[class_id]
            x_center, y_center, width, height = map(float, components[1:5])

            # 归一化坐标 → 像素坐标
            x1 = int((x_center - width / 2) * w_original)
            y1 = int((y_center - height / 2) * h_original)
            x2 = int((x_center + width / 2) * w_original)
            y2 = int((y_center + height / 2) * h_original)

            # 边界保护
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_original, x2), min(h_original, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            # class_3 (合計金額): 仅取右侧 50%（左边是"合計"标签文字，右边是金额数值）
            if class_id == 3:
                x1 = int(x1 + (x2 - x1) * 0.5)
                if x1 >= x2:
                    continue

            cropped_image = image[y1:y2, x1:x2]

            # PaddleOCR 推理
            try:
                result = ocr.predict(cropped_image)
                texts = result[0].get("rec_texts", []) if result else []
                extracted_text = " ".join(texts)
            except Exception as e:
                print(f"OCR failed for {label_file} {cname}: {e}")
                extracted_text = ""

            # 文本清洗
            if cleaner and extracted_text:
                extracted_text = cleaner(extracted_text)

            # 按类别累积
            if ocr_results[cname]:
                ocr_results[cname] += " " + extracted_text
            else:
                ocr_results[cname] = extracted_text

        # 清理空白
        for cname, _ in CLASS_INFO.values():
            ocr_results[cname] = ocr_results[cname].strip()

        results.append(ocr_results)

    if not results:
        print("No OCR results to save.")
        return None

    # 输出 CSV
    columns = ["image_name"] + [info[0] for info in CLASS_INFO.values()]
    df = pd.DataFrame(results)[columns]

    output_csv = get_sequential_csv_name(base_name=output_csv_base, directory=".")
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"OCR results saved to {output_csv}")
    return output_csv


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PaddleOCR: OCR on detected regions from YOLO labels.")
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="ocr_results")

    args = parser.parse_args()

    labels_subfolder = os.path.join(args.input_folder, "labels")
    label_dir = labels_subfolder if os.path.isdir(labels_subfolder) else args.input_folder
    print(f"Using labels from: {label_dir}")
    print(f"Image directory: {args.image_dir}")
    print(f"Output CSV base name: {args.output_csv}")

    result_path = perform_ocr_on_folder(label_dir, args.image_dir, args.output_csv)
    if result_path:
        print(f"OCR completed successfully. Results: {result_path}")
    else:
        print("OCR completed with no results.")
