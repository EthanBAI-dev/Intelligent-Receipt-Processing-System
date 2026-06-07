# Receipt OCR Processing System

> **Language**: [日本語](README.md) | [中文](README.zh.md)

Multi-receipt detection, classification, text-area detection, and OCR recognition pipeline powered by YOLOv8, YOLOv5, ResNet34, and PaddleOCR.
Built with Streamlit for an intuitive web-based UI.

---

## System Architecture

Processes a **composite photo of 4 receipts** through splitting, orientation correction, text region detection, and OCR to produce structured CSV output.

```
                    ┌─────────────────────────────────────┐
  Composite Photo   │  YOLOv8m-Detect (best_v8_300.pt)   │
  (4 receipts)   ──▶  Split & Perspective Correction     │
                    └──────────────┬──────────────────────┘
                                   │  4 individual receipts
                    ┌──────────────▼──────────────────────┐
                    │  ResNet34 Classification            │
                    │  Orientation correction (0/90/180/270°) │
                    └──────────────┬──────────────────────┘
                                   │  4 upright receipts
                    ┌──────────────▼──────────────────────┐
                    │  YOLOv5 Detection (runs/train/exp)  │
                    │  4 classes: date / store / phone / total │
                    └──────────────┬──────────────────────┘
                                   │  bounding boxes
                    ┌──────────────▼──────────────────────┐
                    │  PaddleOCR (PP-OCRv5, lang=japan)   │
                    │  Class-specific text recognition    │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                          CSV Output (.csv)
```

## Project Structure

```
Receipt_OCR_UI_System-main/
├── app.py                     # Streamlit main entry (Full Pipeline + individual steps)
├── OCR.py                     # PaddleOCR module (global singleton, 4-class post-processing)
├── Classification.py          # ResNet34 orientation classifier (0/90/180/270°)
├── detect.py                  # YOLOv5 inference script (text area detection)
├── perspective_utils.py       # Shared perspective correction (corner + OpenCV fallback)
├── best_v8_300.pt             # YOLOv8m-Detect: receipt + corner detection model
│
├── logs/
│   └── Epoch82-Total_Loss0.0004.pth   # ResNet34 classification weights
│
├── runs/
│   ├── train/exp/weights/best.pt      # YOLOv5 text detection weights
│   └── detect/exp*/                    # Detection output (auto-incremented)
│       ├── corrected_image_*.jpg       #   Annotated detection results
│       └── labels/
│           └── corrected_image_*.txt   #   YOLO-format labels
│
├── data/
│   ├── corrected_images/               # Step 2 output: orientation-corrected receipts
│   ├── test/images/                    # Legacy test images directory
│   ├── pipeline_temp/classified/       # Pipeline temporary classification results
│   ├── prediction/                     # Classification input directory
│   └── ocr_upload/                     # Historical upload directory
│
└── ocr_results*.csv                    # Final OCR output (auto-numbered)
```

## Models

### 1. Receipt Detection & Splitting

| Property | Value |
|----------|-------|
| Model | YOLOv8m-Detect |
| Weights | `best_v8_300.pt` |
| Classes | 5: `receipt(0)`, `corner_tl(1)`, `corner_tr(2)`, `corner_br(3)`, `corner_bl(4)` |
| Training | 300 synthetic composite images, 4 receipts per image |
| Confidence | 0.25 |
| NMS IoU | 0.45 |

**Perspective correction** (`perspective_utils.py`):

| Method | Trigger | Description |
|--------|---------|-------------|
| **Corner-based** | 4 corner points + validation passed | Uses class IDs (TL/TR/BR/BL) to map directly, no geometric sorting needed |
| **OpenCV fallback** | Fewer than 4 corners or validation failed | CLAHE + adaptive threshold + contour quadrilateral detection |

**Validation**: boundary clamp, self-intersection detection, area > 100px, aspect ratio < 8:1.

### 2. Orientation Classification

| Property | Value |
|----------|-------|
| Model | ResNet34 |
| Weights | `logs/Epoch82-Total_Loss0.0004.pth` |
| Input | 224×224, normalized to [-1, 1] |
| Classes | 0=0° upright, 1=90° CCW, 2=180°, 3=270° CCW |

### 3. Text Area Detection

| Property | Value |
|----------|-------|
| Framework | YOLOv5 |
| Weights | `runs/train/exp/weights/best.pt` |
| Input | 640×640 |
| Confidence | 0.5 |

| Class ID | Field | CSV Column | PaddleOCR Post-processing |
|:--------:|-------|:----------:|---------------------------|
| 0 | Date | `date` | Strip only (PP-OCRv5 handles date format natively) |
| 1 | Store Name | `store` | Keep JP kana/kanji + alphanumeric |
| 2 | Phone Number | `phone` | Keep digits + `- ( ) . :` + TEL |
| 3 | Total Amount | `total` | Right 50% crop (numeric value with ¥), keep digits + ¥$ |

### 4. OCR Recognition

| Property | Value |
|----------|-------|
| Engine | PaddleOCR 3.6.0 |
| Models | PP-OCRv5_server_det + PP-OCRv5_server_rec + PP-LCNet |
| Language | `japan` (Japanese + English) |
| Instance | Global singleton (`get_ocr()`), initialized once at module load |
| Cache | Auto-download & cache in `~/.paddlex/` |
| Output | UTF-8 BOM CSV, auto-numbered (`ocr_results1.csv`, …) |

## CSV Output Format

| Column | Type | Example |
|--------|------|---------|
| `image_name` | string | `corrected_image_1.jpg` |
| `date` | string | `2026-03-21` |
| `store` | string | `Beisie 吉井店` |
| `phone` | string | `027-387-6511` |
| `total` | string | `￥1,529` |

## Usage

### Quick Start

```bash
# Install dependencies
pip install streamlit ultralytics paddleocr paddlepaddle pillow opencv-python pandas

# Start the server
cd Receipt_OCR_UI_System-main
python -m streamlit run app.py --server.port 8503
```

### Sidebar Modes

| Mode | Description |
|------|-------------|
| **Full Pipeline** | Upload composite photo → split → classify → detect → OCR → CSV (one-click) |
| **Multi Receipts Detection** | Split composite photo → crop → perspective correction |
| **Classification** | Single-image orientation correction (0/90/180/270°) |
| **Detection** | Run YOLOv5 text-area detection on corrected images |
| **OCR** | Run PaddleOCR on existing detection results → CSV |

### Full Pipeline Display Layout

Each step shows **4 columns** (one per receipt), vertically aligned:

```
┌──────────┬──────────┬──────────┬──────────┐
│   Step 0: Original Composite Image        │
├──────────┼──────────┼──────────┼──────────┤
│   Step 1: Cropped (4 individual receipts) │
├──────────┼──────────┼──────────┼──────────┤
│   Step 1: Rectified [method]              │
├──────────┼──────────┼──────────┼──────────┤
│   Step 2: Orientation Correction          │
├──────────┼──────────┼──────────┼──────────┤
│   Step 3: Text Area Detection             │
├──────────┼──────────┼──────────┼──────────┤
│   Step 4: OCR Results CSV (downloadable)  │
└──────────┴──────────┴──────────┴──────────┘
```

Files separated by **red horizontal dividers**.

## File Storage

| Category | Path | Naming | Retention |
|----------|------|--------|-----------|
| Detection model | `./best_v8_300.pt` | Fixed | Permanent |
| Perspective utils | `./perspective_utils.py` | Fixed | Permanent |
| Classification weights | `./logs/Epoch82-*.pth` | Fixed | Permanent |
| Detection weights | `./runs/train/exp/weights/best.pt` | Fixed | Permanent |
| Corrected images | `./data/corrected_images/` | `corrected_image_{i}.jpg` | Overwritten each run |
| Detection outputs | `./runs/detect/exp{n}/` | Auto-incremented | Permanent |
| Detection labels | `./runs/detect/exp{n}/labels/` | `corrected_image_{i}.txt` | Permanent |
| OCR results | `./` | `ocr_results{n}.csv` | Auto-numbered, permanent |
| PaddleOCR models | `~/.paddlex/official_models/` | Auto-downloaded | Cached indefinitely |
| Uploaded originals | Memory only | — | Session lifetime |

## Key Design Decisions

1. **Global OCR singleton** — `PaddleOCR` initialized once via `get_ocr()`, not per-image.

2. **Corner-based perspective correction** — Uses class IDs (1=TL, 2=TR, 3=BR, 4=BL) directly rather than geometric sorting, preventing failures on rotated/skewed receipts.

3. **Automatic OpenCV fallback** — Transparent fallback to contour-based rectification when corners are insufficient or validation fails.

4. **Aspect ratio filtering** — Discards receipt boxes with aspect ratio outside `[0.20, 0.95]`, eliminating false-positive narrow strips.

5. **X-range constrained corner grouping** — Corners only matched to receipt boxes whose x-range overlaps with sufficient margin, preventing cross-receipt misassignment.

6. **Class-specific OCR post-processing** — Each of the 4 text area classes gets tailored text cleaning (date: minimal, store: JP characters, phone: digits+delimiters, total: right-half crop).

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | >=1.28 | Web UI |
| ultralytics | >=8.0 | YOLOv8 inference |
| paddleocr | 3.6.0 | OCR recognition |
| paddlepaddle | 3.3.1 | PaddlePaddle backend |
| opencv-python | >=4.5 | Image processing & perspective transform |
| pillow | >=9.0 | PIL image operations |
| pandas | >=1.5 | CSV handling & DataFrame display |
| torch | >=2.0 | ResNet34 classification |
| torchvision | >=0.15 | Classification transforms |

## References

- [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics)
- [PaddleOCR / PP-OCRv5](https://github.com/PaddlePaddle/PaddleOCR)
- [Streamlit](https://streamlit.io/)
