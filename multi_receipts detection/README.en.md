# Multi-receipt Corner Detection System

[日本語](README.md) | [中文](README.zh.md)

---

## 1. Core Functionality & Positioning

This project is a **multi-receipt corner detection system** based on **YOLOv8m-Detect**, designed to precisely detect the bounding boxes and four corners of each receipt from **4-in-1 composite images**, providing high-precision localization for subsequent perspective correction and OCR recognition.

**Key Use Cases**:
- Detect bounding boxes of each receipt from a 4-in-1 composite image
- Detect 4 corner points (top-left, top-right, bottom-right, bottom-left) for each receipt
- Provide corner coordinates for perspective correction to rectify receipts
- Serves as a **foundational detection module** integrated into a full OCR pipeline

---

## 2. Core Technology & Implementation

### Technology Stack

| Technology | Purpose |
|------------|---------|
| **YOLOv8m-Detect** | Object detection model for receipt boxes and corner boxes |
| **PIL / OpenCV** | Image perspective transformation and processing |
| **NumPy** | Matrix operations (homography computation) |
| **SAHI** | Slicing-aided inference (historically used in v1/v2) |

### Model Output (5 Classes)

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | receipt | Receipt bounding box |
| 1 | corner_tl | Top-left corner |
| 2 | corner_tr | Top-right corner |
| 3 | corner_br | Bottom-right corner |
| 4 | corner_bl | Bottom-left corner |

### Training Data Generation

`train/generate_data.py` synthesizes training data through the following pipeline:

1. **Source images** — Load individual receipt images from `train/source_images/`
2. **Perspective transform** — Apply ±12° pitch/yaw 3D perspective + local distortion
3. **Random rotation** — 70% probability of ±6° rotation for augmentation
4. **Trim offset** — Auto-crop blank edges after transformation
5. **Horizontal stitching** — Paste 4 transformed receipts onto a gray canvas (120px gap)
6. **Label generation** — 5 boxes per receipt (1 receipt + 4 corners)
7. **Auto-compression** — Resize to max 1280px on the longest side

### 3-Level Perspective Correction

`tools/perspective_utils.py` provides a three-level fallback strategy:

1. **Corner method** (preferred) — 4-point perspective transform using detected corner box centers (with 5 validation checks)
2. **OpenCV contour method** (fallback) — CLAHE + adaptive threshold + morphology + largest quadrilateral fit
3. **Original image** (last resort)

---

## 3. Development History

### v1 — YOLOv8-Pose Keypoint Scheme (Archived)

- **Approach**: YOLOv8-Pose model, corners regressed as keypoints
- **Data**: ~150 synthetic training images
- **Issue**: Limited keypoint regression accuracy, no confidence output

### v2 — Pose Data Augmented (Archived)

- **Approach**: Same Pose keypoint method as v1
- **Data**: ~500 synthetic training images
- **Model**: `best_2.pt`
- **Issue**: Inherent limitations of the Pose approach remained

### v3 — YOLOv8-Detect Corner Detection (Current Production)

- **Key Change**: Corners changed from Pose keypoints to **independent detection targets (Detect boxes)**
- **Advantages**: Independent confidence per corner, NMS support, higher accuracy
- **Data**: 300 synthetic training images
- **Model**: `model/best_v8_300.pt`
- **Status**: In production, integrated into OCR system

### Version Comparison

| Version | Approach | Data Size | Corner Method | Rectify Success Rate | Status |
|---------|----------|-----------|---------------|---------------------|--------|
| v1 | YOLOv8-Pose | ~150 | Keypoints | Low | Archived |
| v2 | YOLOv8-Pose | ~500 | Keypoints | Low | Archived |
| **v3** | **YOLOv8-Detect** | **300** | **Detect boxes** | **~70% corner + OpenCV fallback** | **Production** |

> Historical versions archived in `back up/项目迭代文件/multi_receipt_detection/`

---

## 4. Installation & Deployment

### Requirements

- Python >= 3.8
- OS: macOS / Linux / Windows

### Directory Structure

```
multi_receipts detection/
├── train/                           # Training pipeline
│   ├── generate_data.py             #     Data generation script
│   ├── source_images/               #     Source receipt images (10 samples)
│   └── yolo_dataset/                #     Generated synthetic dataset
│       ├── data.yaml                #     Dataset config
│       ├── images/train/            #     Training images
│       ├── images/val/              #     Validation images
│       ├── labels/train/            #     Training labels
│       └── labels/val/              #     Validation labels
├── model/                           # Model weights
│   └── best_v8_300.pt               #     Production model
├── tools/                           # Tools
│   └── perspective_utils.py         #     Perspective correction
├── final_test/                      # Real test images
│   ├── Image_1.jpg
│   ├── Image_2.jpg
│   └── Image_3.jpg
├── evaluate.py                      # Evaluation script
├── evaluate_output/                 # Evaluation results
├── README.md                        # Documentation (Japanese)
├── README.zh.md                     # Documentation (Chinese)
├── README.en.md                     # This file (English)
└── REIMPLEMENTATION_PLAN.md         # Reimplementation plan
```

### Installation

```bash
# 1. Enter project directory
cd multi_receipts detection/

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install ultralytics pillow numpy pyyaml opencv-python
```

### Quick Start

```bash
# 1. Generate sample data (10 images)
cd train/
python generate_data.py

# 2. Run inference evaluation
cd ..
python evaluate.py                     # Inference stats
python evaluate.py --save              # Inference + save annotated images

# 3. Train model (full dataset)
yolo detect train data=train/yolo_dataset/data.yaml \
      model=yolov8m.pt epochs=300 imgsz=1280 batch=8

# 4. Run inference with trained model
yolo detect predict model=model/best_v8_300.pt \
      source=final_test/ save=True
```

### Test Images

The `final_test/` directory contains 3 real-world 4-in-1 receipt photos for model validation.

---

## License

This project is provided for learning and research purposes only.
