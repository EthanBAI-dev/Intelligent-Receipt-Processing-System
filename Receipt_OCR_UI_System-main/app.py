import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import cv2
import pandas as pd
import numpy as np
import subprocess
import glob
from perspective_utils import rectify_receipt

# ====================== 基础路径配置 ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ====================== 工具函数 ======================

def get_unique_csv_name(base_folder, base_name="ocr_results", extension=".csv"):
    counter = 1
    while True:
        file_name = f"{base_name}{counter}{extension}"
        file_path = os.path.join(base_folder, file_name)
        if not os.path.exists(file_path):
            return file_path
        counter += 1

def get_latest_exp_folder(base_folder=None):
    if base_folder is None:
        base_folder = os.path.join(BASE_DIR, "runs", "detect")
    if not os.path.exists(base_folder):
        return None
    subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir() and f.name.startswith("exp")]
    if not subfolders:
        return None
    return max(subfolders, key=os.path.getmtime)

def filter_receipts_by_aspect(predictions, min_aspect=0.20, max_aspect=0.95):
    filtered, removed = [], 0
    for pred in predictions:
        xyxy = pred[1]
        w, h = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
        if h <= 0:
            removed += 1; continue
        if min_aspect <= w / h <= max_aspect:
            filtered.append(pred)
        else:
            removed += 1
    return filtered, removed

@st.cache_resource
def load_detect_model(model_path):
    return YOLO(model_path)

DETECT_MODEL_PATH = os.path.join(BASE_DIR, "best_v8_300.pt")

METHOD_LABELS = {"corner": "[Corner]", "opencv": "[OpenCV]", "fallback": "[Fallback]"}

def group_corners_to_receipt(receipt_boxes, corner_predictions):
    receipt_corners = [[] for _ in receipt_boxes]
    for ci, xy, cf in corner_predictions:
        cx, cy = (xy[0] + xy[2]) / 2, (xy[1] + xy[3]) / 2
        best_ri, best_dist = -1, 1e9
        for ri, (_, rb, _) in enumerate(receipt_boxes):
            rx_w = rb[2] - rb[0]
            if cx < rb[0] - rx_w * 0.4 or cx > rb[2] + rx_w * 0.4:
                continue
            rx_c, ry_c = (rb[0] + rb[2]) / 2, (rb[1] + rb[3]) / 2
            dist = np.sqrt((cx - rx_c) ** 2 + (cy - ry_c) ** 2)
            if dist < best_dist:
                best_dist, best_ri = dist, ri
        if best_ri >= 0:
            receipt_corners[best_ri].append((ci, (cx, cy), cf))
    return receipt_corners

def detect_crop_and_rectify(img_pil, model):
    img_np = np.array(img_pil)
    h_orig, w_orig = img_np.shape[:2]
    MAX_DIM = 2000
    if max(h_orig, w_orig) > MAX_DIM:
        scale = MAX_DIM / max(h_orig, w_orig)
        img_np = cv2.resize(img_np, (int(w_orig * scale), int(h_orig * scale)))

    results = model.predict(img_np, conf=0.25, iou=0.45, verbose=False)
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []

    cls_ids = r.boxes.cls.cpu().numpy().astype(int)
    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    all_preds = [(int(ci), xy, float(cf)) for ci, xy, cf in zip(cls_ids, xyxy, confs)]

    detect_receipts = [(ci, xy, cf) for ci, xy, cf in all_preds if ci == 0]
    detect_receipts, _ = filter_receipts_by_aspect(detect_receipts)
    detect_receipts.sort(key=lambda x: x[1][0])
    detect_corners = [(ci, xy, cf) for ci, xy, cf in all_preds if ci in (1, 2, 3, 4)]
    corner_groups = group_corners_to_receipt(detect_receipts, detect_corners)

    cv_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    CROP_PAD = 10
    results_out = []
    for ri, (_, rbox, _) in enumerate(detect_receipts):
        x1, y1, x2, y2 = map(int, rbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(cv_img.shape[1], x2), min(cv_img.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop_x1, crop_y1 = max(0, x1 - CROP_PAD), max(0, y1 - CROP_PAD)
        crop_x2, crop_y2 = min(cv_img.shape[1], x2 + CROP_PAD), min(cv_img.shape[0], y2 + CROP_PAD)
        cropped = cv_img[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        padded = np.full(((y2 - y1) + 2 * CROP_PAD, (x2 - x1) + 2 * CROP_PAD, 3), (128, 128, 128), dtype=np.uint8)
        px, py = CROP_PAD - (x1 - crop_x1), CROP_PAD - (y1 - crop_y1)
        padded[py:py + cropped.shape[0], px:px + cropped.shape[1]] = cropped
        cropped_pil = Image.fromarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
        corners_info = corner_groups[ri] if ri < len(corner_groups) else []
        corner_with_class = []
        for ci, (cx, cy), _ in corners_info:
            pcx = max(0, min(padded.shape[1] - 1, CROP_PAD + cx - x1))
            pcy = max(0, min(padded.shape[0] - 1, CROP_PAD + cy - y1))
            corner_with_class.append((ci, pcx, pcy))
        rectified_pil, method = rectify_receipt(cropped_pil, corner_with_class)
        results_out.append({'cropped': cropped_pil, 'rectified': rectified_pil, 'index': ri + 1, 'method': method})
    return results_out


# ====================== 4列展示工具 ======================

def display_4col_row(images, captions, width=280):
    """用4列展示一组图片：两侧留白居中、列间距紧凑、不铺满全屏"""
    n = min(len(images), 4)
    # 6列: [左留白 | 图1 | 图2 | 图3 | 图4 | 右留白], 内容占80%
    ratios = [1.0] + [2.0] * n + [2.0] * (4 - n) + [1.0]
    cols = st.columns(ratios, gap="small")
    for i in range(n):
        with cols[i + 1]:
            if images[i] is not None:
                st.image(images[i], caption=captions[i] if i < len(captions) else "", width=width)


def pad_to_4(items, fill=None):
    """补到4个元素"""
    return (list(items) + [fill] * 4)[:4]


# ====================== UI ======================

st.set_page_config(layout="wide")

# ====================== 全局 UI 样式 ======================
st.markdown("""
<style>
/* 主标题 */
h1 { font-size: 2.4rem !important; font-weight: 800 !important; line-height: 1.3 !important; margin-bottom: 0.6rem !important; }
/* 步骤标题 (st.header) */
h2 { font-size: 1.55rem !important; font-weight: 700 !important; line-height: 1.4 !important; margin: 1.0rem 0 0.5rem 0 !important; padding-bottom: 0.3rem !important; border-bottom: 2px solid #3a7bd5 !important; }
/* 子标题 (st.subheader) */
h3 { font-size: 1.25rem !important; font-weight: 700 !important; line-height: 1.4 !important; margin: 0.8rem 0 0.4rem 0 !important; }
/* 流程步骤徽章 */
.step-badge {
    font-size: 1.5rem !important; font-weight: 800 !important;
    display: inline-block; padding: 0.5rem 1.0rem;
    margin: 1.0rem 0 0.5rem 0;
    background: #b2bec3;
    color: #2d3436; border-radius: 6px;
}
/* 分割子标题 */
.section-title {
    font-size: 1.1rem !important; font-weight: 700 !important;
    margin: 0.8rem 0 0.4rem 0 !important;
    padding-bottom: 0.25rem !important;
    border-bottom: 2px solid #dfe6e9 !important;
}
/* 信息行 */
.info-line { font-size: 0.95rem; font-weight: 600; color: #636e72; margin: 0.3rem 0; }
/* 原始图片说明块 */
.original-note {
    font-size: 0.95rem; font-weight: 600; color: #636e72;
    margin: 0.4rem 0 0.8rem 0;
}
/* 4列图片行：紧凑间距 + 图片居中 */
[data-testid="stHorizontalBlock"] {
    gap: 0.5rem !important;
}
[data-testid="stHorizontalBlock"] img {
    display: block !important;
    margin: 0 auto !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Receipt Processing System")
st.sidebar.title("Function selection")
functionality = st.sidebar.radio(
    "Please select the function",
    ("Full Pipeline", "Multi Receipts Detection", "Classification", "Detection", "OCR")
)

# ====================== FULL PIPELINE ======================

if functionality == "Full Pipeline":
    st.header("Full Pipeline: Detect -> Classify -> Detect -> OCR")
    st.markdown("Upload a photo with multiple receipts. The system will: crop & rectify -> classify direction -> detect text areas -> OCR")
    gif_path = os.path.join(BASE_DIR, "receipt detection.gif")
    if os.path.exists(gif_path):
        with st.expander("Demo Preview", expanded=False):
            st.image(gif_path, caption="OCR Pipeline Demo", width=900)

    uploaded_files = st.file_uploader("Upload images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files and st.button("Start Full Pipeline"):
        model = load_detect_model(DETECT_MODEL_PATH)

        for file_idx, uploaded_file in enumerate(uploaded_files):
            st.markdown(f'<hr style="border: 5px solid red; margin: 20px 0;">', unsafe_allow_html=True)
            st.subheader(f"File: {uploaded_file.name}")

            # ====== [ORIGINAL IMAGE DISPLAY NODE] ======
            img = Image.open(uploaded_file).convert("RGB")
            st.markdown('<p class="step-badge">Step 0/4: Original Image Display</p>', unsafe_allow_html=True)
            st.markdown('<p class="original-note">This is a composite image containing 4 individual receipts stitched together. The subsequent Split step separates them for individual processing.</p>', unsafe_allow_html=True)
            img_cols = st.columns([0.3, 3, 0.3], gap="small")
            with img_cols[1]:
                st.image(img, caption=f"[Original Input] {uploaded_file.name}  ({img.width} x {img.height})", width=1200)
            # ==============================================

            with st.spinner("Step 1/4: Splitting & rectifying receipts..."):
                results = detect_crop_and_rectify(img, model)

            if len(results) == 0:
                st.warning("No receipts detected.")
                continue

            n = len(results)
            st.markdown(f'<p class="step-badge">Step 1/4: Split & Rectify — Detected {n} receipt(s)</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="info-line">Model: YOLOv8-Detect (best_v8_300.pt)</p>', unsafe_allow_html=True)

            st.markdown('<p class="section-title">Cropped</p>', unsafe_allow_html=True)
            cropped_imgs = [r['cropped'] for r in results]
            cropped_caps = [f"#{r['index']} Crop" for r in results]
            display_4col_row(pad_to_4(cropped_imgs), pad_to_4(cropped_caps))

            st.markdown('<p class="section-title">Rectified</p>', unsafe_allow_html=True)
            rect_imgs = [r['rectified'] for r in results]
            rect_caps = [f"#{r['index']} {METHOD_LABELS.get(r['method'], r['method'])}" for r in results]
            display_4col_row(pad_to_4(rect_imgs), pad_to_4(rect_caps))

            with st.spinner("Step 2/4: Classifying orientation..."):
                from Classification import predict_and_correct_images, model as cls_model, data_transforms
                corrected, labels = predict_and_correct_images(
                    [r['rectified'] for r in results], model=cls_model, transform=data_transforms
                )
            label_names = {0: '0 deg (upright)', 1: '90 deg CCW -> corrected', 2: '180 deg -> corrected', 3: '270 deg CCW -> corrected'}
            st.markdown('<p class="step-badge">Step 2/4: Orientation Classification</p>', unsafe_allow_html=True)
            corr_imgs = pad_to_4(corrected)
            corr_caps = pad_to_4([f"#{r['index']} [{label_names.get(l, str(l))}]" for r, l in zip(results, labels)])
            display_4col_row(corr_imgs, corr_caps)

            with st.spinner("Step 3/4: Detecting text areas..."):
                corrected_images_dir = os.path.join(BASE_DIR, "data", "corrected_images")
                detect_weights = os.path.join(BASE_DIR, "runs", "train", "exp", "weights", "best.pt")
                detect_cmd = ["python", "detect.py", "--img", "640", "--conf", "0.5",
                              "--device", "cpu", "--weights", detect_weights,
                              "--source", corrected_images_dir, "--save-txt", "--save-conf"]
                try:
                    result = subprocess.run(detect_cmd, check=True, capture_output=True,
                                            text=True, cwd=BASE_DIR)
                    latest = get_latest_exp_folder()
                    st.caption(f"Detection output: {result.stdout[-300:] if len(result.stdout) > 300 else result.stdout}")
                except subprocess.CalledProcessError as e:
                    st.error(f"Detection failed: {e.stderr}")
                    latest = None

            st.markdown('<p class="step-badge">Step 3/4: Text Area Detection</p>', unsafe_allow_html=True)
            if latest:
                det_imgs = []
                det_caps = []
                for ri in range(1, n + 1):
                    found = None
                    for fname in sorted(os.listdir(latest)):
                        if f"corrected_image_{ri}" in fname:
                            found = fname
                            break
                    if found:
                        det_imgs.append(Image.open(os.path.join(latest, found)))
                        det_caps.append(f"#{ri} Detection")
                if det_imgs:
                    display_4col_row(pad_to_4(det_imgs), pad_to_4(det_caps))
                else:
                    st.warning("No detection results found.")
            else:
                st.warning("Detection step failed.")

            with st.spinner("Step 4/4: Running OCR..."):
                if latest:
                    ocr_cmd = ["python", "OCR.py", "--input_folder", latest,
                               "--image_dir", corrected_images_dir,
                               "--output_csv", "ocr_results"]
                    try:
                        result = subprocess.run(ocr_cmd, check=True, capture_output=True,
                                                text=True, cwd=BASE_DIR)
                        st.caption(f"OCR output: {result.stdout[-300:] if len(result.stdout) > 300 else result.stdout}")
                    except subprocess.CalledProcessError as e:
                        st.error(f"OCR failed: {e.stderr}")

            st.markdown('<p class="step-badge">Step 4/4: OCR Results</p>', unsafe_allow_html=True)
            csv_files = sorted(glob.glob(os.path.join(BASE_DIR, "ocr_results*.csv")), key=os.path.getmtime)
            if csv_files:
                ocr_csv = csv_files[-1]
                df = pd.read_csv(ocr_csv)
                st.dataframe(df, width='stretch')
                with open(ocr_csv, "rb") as f:
                    st.download_button("Download CSV", f, os.path.basename(ocr_csv), "text/csv",
                                       key=f"download_csv_{file_idx}")
            else:
                st.info("OCR CSV not found. Check OCR.py output.")

# ====================== MULTI RECEIPTS DETECTION ======================

elif functionality == "Multi Receipts Detection":
    st.header("Multi Receipts Detection")
    st.markdown("Detect each receipt -> crop -> perspective correction")

    uploaded_files = st.file_uploader("Upload images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.image(Image.open(uploaded_file), caption=uploaded_file.name, width=400)

        if st.button("Start Detection"):
            model = load_detect_model(DETECT_MODEL_PATH)
            for file_idx, uploaded_file in enumerate(uploaded_files):
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    img = Image.open(uploaded_file).convert("RGB")
                    results = detect_crop_and_rectify(img, model)

                if len(results) == 0:
                    st.warning(f"No receipts in {uploaded_file.name}")
                    continue

                st.markdown(f'<hr style="border: 8px solid red; margin: 20px 0;">', unsafe_allow_html=True)
                st.subheader(f"File: {uploaded_file.name}")
                st.markdown(f"Detected **{len(results)}** receipts | YOLOv8-Detect (best_v8_300.pt)")

                st.markdown('<p class="section-title">Cropped</p>', unsafe_allow_html=True)
                display_4col_row(
                    pad_to_4([r['cropped'] for r in results]),
                    pad_to_4([f"#{r['index']} Crop" for r in results])
                )
                st.markdown('<p class="section-title">Rectified</p>', unsafe_allow_html=True)
                display_4col_row(
                    pad_to_4([r['rectified'] for r in results]),
                    pad_to_4([f"#{r['index']} {METHOD_LABELS.get(r['method'], r['method'])}" for r in results])
                )

# ====================== CLASSIFICATION ======================

elif functionality == "Classification":
    st.header("Classification: Orientation Correction")
    uploaded_files = st.file_uploader("Upload images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.image(Image.open(uploaded_file), caption=uploaded_file.name, width=400)

        if st.button("Start Classification"):
            from Classification import predict_and_correct_images, model as cls_model, data_transforms
            for file_idx, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file)
                with st.spinner(f"Classifying {uploaded_file.name}..."):
                    corrected, labels = predict_and_correct_images([image], model=cls_model, transform=data_transforms)
                label_names = {0: '0 deg Upright', 1: '90 deg CCW -> Corrected', 2: '180 deg -> Corrected', 3: '270 deg CCW -> Corrected'}
                st.markdown(f'<hr style="border: 8px solid red; margin: 20px 0;">', unsafe_allow_html=True)
                st.subheader(uploaded_file.name)
                st.image(corrected[0], caption=f"Predicted: {label_names.get(labels[0], labels[0])}", width=400)

elif functionality == "Detection":
    st.header("Object Detection: Text Areas")
    detection_folder = os.path.join(BASE_DIR, "data", "corrected_images")
    base_output_folder = os.path.join(BASE_DIR, "runs", "detect")
    detect_weights = os.path.join(BASE_DIR, "runs", "train", "exp", "weights", "best.pt")

    if os.path.exists(detection_folder) and os.listdir(detection_folder):
        if st.button("Start Detection"):
            command = ["python", "detect.py", "--img", "640", "--conf", "0.5", "--device", "cpu",
                       "--weights", detect_weights,
                       "--source", detection_folder, "--save-txt", "--save-conf"]
            try:
                with st.spinner("Running..."):
                    subprocess.run(command, check=True, cwd=BASE_DIR)
                st.success("Done!")
                latest = get_latest_exp_folder(base_output_folder)
                if latest:
                    detected = [f for f in os.listdir(latest) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
                    for fname in detected:
                        st.image(os.path.join(latest, fname), caption=fname, width=400)
            except subprocess.CalledProcessError as e:
                st.error(f"Detection failed: {e.stderr}")
    else:
        st.warning("No corrected images. Run Classification first.")

elif functionality == "OCR":
    st.header("OCR")
    detection_folder = get_latest_exp_folder()
    corrected_images_dir = os.path.join(BASE_DIR, "data", "corrected_images")

    if st.button("Start OCR"):
        if not detection_folder:
            st.error("No detection results found. Run Detection first.")
        else:
            command = ["python", "OCR.py", "--input_folder", detection_folder,
                       "--image_dir", corrected_images_dir, "--output_csv", "ocr_results"]
            try:
                with st.spinner("Running OCR..."):
                    subprocess.run(command, check=True, cwd=BASE_DIR)
                csv_files = sorted(glob.glob(os.path.join(BASE_DIR, "ocr_results*.csv")), key=os.path.getmtime)
                if csv_files:
                    ocr_csv = csv_files[-1]
                    st.success("Done!")
                    df = pd.read_csv(ocr_csv)
                    st.dataframe(df, width='stretch')
                    with open(ocr_csv, "rb") as f:
                        st.download_button("Download CSV", f, os.path.basename(ocr_csv), "text/csv",
                                           key="download_csv_ocr")
                else:
                    st.warning("OCR CSV not found. Check OCR.py output.")
            except subprocess.CalledProcessError as e:
                st.error(f"OCR failed: {e.stderr}")
