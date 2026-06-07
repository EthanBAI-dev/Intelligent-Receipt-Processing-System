# Receipt Auto Digitizer

[日本語](README.md) | [中文](README.zh.md) | **English**

---

> This project is an **end-to-end receipt digitization pipeline** that takes a **composite photo of multiple receipts** and automatically generates structured CSV data through **detection, classification, text region recognition, and OCR**.

---

## System Overview

**Receipt Auto Digitizer** transforms a composite photo containing 4 receipts into digital data through 4 steps:

1. **Multi-Receipt Detection & Perspective Correction** — YOLOv8m-Detect detects each receipt's bounding box and 4 corners, applies perspective transform for rectification
2. **Orientation Classification & Correction** — ResNet34 classifies rotation angle (0°/90°/180°/270°) and corrects to upright
3. **Text Region Detection** — YOLOv5/YOLOv9 detects 4 classes: date, store name, phone number, total amount
4. **OCR Recognition & CSV Output** — PaddleOCR (PP-OCRv5) recognizes text in each region, post-processes and exports to structured CSV

### System Architecture

```
                    ┌─────────────────────────────────────┐
  Composite Photo   │  YOLOv8m-Detect                     │
  (4 receipts)   ──▶  Receipt Box & Corner Detection     │
                    │  Perspective Correction & Split      │
                    └──────────────┬──────────────────────┘
                                   │  4 individual receipts
                    ┌──────────────▼──────────────────────┐
                    │  ResNet34 Orientation Classification │
                    │  0°/90°/180°/270° → 0° correction   │
                    └──────────────┬──────────────────────┘
                                   │  4 upright receipts
                    ┌──────────────▼──────────────────────┐
                    │  YOLOv5 / YOLOv9 Text Detection     │
                    │  4 classes: date/store/phone/total   │
                    └──────────────┬──────────────────────┘
                    │  bounding boxes
                    ┌──────────────▼──────────────────────┐
                    │  PaddleOCR (PP-OCRv5)               │
                    │  Class-specific text recognition    │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                          CSV Output (.csv)
```

### Pipeline Steps

1. **Step 0 — Input**: Upload a composite photo of 4 receipts
2. **Step 1 — Split & Perspective Correction**: YOLOv8m-Detect detects receipt positions and 4 corners, generates individual rectified images via perspective transform
3. **Step 2 — Orientation Correction**: ResNet34 classifier determines receipt orientation, rotates to upright (0°)
4. **Step 3 — Text Region Detection**: YOLOv5/YOLOv9 detects date, store name, phone number, and total amount regions
5. **Step 4 — OCR Recognition**: PaddleOCR recognizes text in each region, post-processes and exports to CSV

---

## Results

> *(Images to be added later)*

<!--
![System Overview Result](images/overview_result.png)
-->

---

## Project Structure

```
receipt-auto-digitizer/
├── Receipt_OCR_UI_System-main/       # [Core] OCR Processing System + Streamlit UI
│   ├── app.py                        # Streamlit main entry
│   ├── OCR.py                        # PaddleOCR module
│   ├── Classification.py             # ResNet34 orientation classifier
│   ├── detect.py                     # YOLOv5 text detection
│   ├── perspective_utils.py          # Perspective correction utilities
│   └── README.md                     # Detailed docs (Japanese)
│
├── Receipt_classificaion_model-main/ # Receipt orientation classification model
│   ├── data_load_train.py            # Training script
│   ├── prediction.py                 # Inference & correction script
│   └── README.md                     # Detailed docs (Japanese)
│
├── Receipt_detection_model-main/     # Receipt text region detection model
│   ├── detect.py                     # YOLOv9 inference script
│   ├── train.py                      # Training script
│   ├── best_text.pt                  # Pre-trained weights
│   └── README.md                     # Detailed docs (Japanese)
│
├── multi_receipts detection/         # Multi-receipt corner detection system
│   ├── train/generate_data.py        # Synthetic data generation
│   ├── evaluate.py                   # Evaluation script
│   ├── tools/perspective_utils.py    # Perspective correction tools
│   └── README.md                     # Detailed docs (Japanese)
│
├── back up/                          # Backup data (VOC2007 format)
│
├── README.md                         # Japanese main docs
├── README.zh.md                      # Chinese docs
└── README.en.md                      # This file (English)
```

---

## Technology Stack

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| **Object Detection** | Ultralytics YOLOv8m-Detect | >=8.0 | Receipt box & corner detection |
| **Object Detection** | YOLOv5 / YOLOv9 (GELAN-C) | - | Text region detection |
| **Image Classification** | ResNet34 | - | Orientation classification (4-class) |
| **OCR** | PaddleOCR (PP-OCRv5) | 3.6.0 | Text recognition |
| **Deep Learning** | PaddlePaddle | 3.3.1 | OCR backend |
| **Deep Learning** | PyTorch | >=2.0 | YOLO / ResNet34 backend |
| **Web UI** | Streamlit | >=1.28 | Interactive UI |
| **Image Processing** | OpenCV | >=4.5 | Perspective transform & image processing |
| **Image Processing** | Pillow | >=9.0 | Image manipulation |
| **Data Processing** | NumPy | - | Matrix operations |
| **Data Processing** | Pandas | >=1.5 | CSV processing |

---

## Environment & Dependencies

### System Requirements

- **OS**: macOS / Linux / Windows
- **Python**: >= 3.8
- **CUDA**: Optional (GPU inference recommended)

### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd receipt-auto-digitizer

# 2. Install dependencies for each sub-project
# (Virtual environment per sub-project is recommended)

# --- OCR UI System ---
cd Receipt_OCR_UI_System-main
pip install streamlit ultralytics paddleocr paddlepaddle opencv-python pillow pandas torch torchvision

# --- Receipt Orientation Classification Model ---
cd ../Receipt_classificaion_model-main
pip install torch torchvision pillow

# --- Receipt Text Region Detection Model ---
cd ../Receipt_detection_model-main
pip install torch torchvision pyyaml

# --- Multi-Receipt Corner Detection ---
cd "../multi_receipts detection"
pip install ultralytics pillow numpy pyyaml opencv-python
```

---

## Deployment

### Quick Start (OCR UI System)

```bash
cd Receipt_OCR_UI_System-main
python -m streamlit run app.py --server.port 8503
```

Open `http://localhost:8503` in your browser, upload a composite photo, and run the pipeline.

### Individual Sub-project Execution

Refer to each sub-project's README for details.

---

## Development Standards

### Coding Conventions

- **Language**: Python 3.8+
- **Style**: PEP 8 compliant
- **Naming**: Snake case (`snake_case`)
- **Type Hints**: Add type annotations where possible

### Git Workflow

- **Branch Strategy**: `main` (stable), `develop` (in-progress), `feature/*` (feature branches)
- **Commit Messages**: Concise description in English or Chinese
- **Pull Requests**: Review required, one PR per feature

### Documentation

- Each sub-project includes README.md (Japanese), README.zh.md (Chinese), README.en.md (English)
- Key functions should include docstrings

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Issues & Suggestions

- Bug reports and feature requests are welcome via Issues
- For security concerns, please contact the maintainer directly

---

## License

This project is intended for learning and research purposes. Refer to each sub-project's license for details.

---

## Maintainer

- **Author**: [baiwenbin](https://github.com/baiwenbin)
- **Contact**: Via GitHub Issues

---

## References

- [YOLOv9](https://github.com/WongKinYiu/yolov9)
- [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [Streamlit](https://streamlit.io/)
