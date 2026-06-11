# 收据自动数字化系统（Receipt Auto Digitizer）

[日本語](README.md) | **中文** | [English](README.en.md)

---

> 本项目以**多收据合成照片**为输入，经过**检测、分类、文字区域识别、OCR** 等步骤，自动生成结构化的 CSV 数据，实现端到端的收据数字化流水线。

---

## 系统概述

**Receipt Auto Digitizer** 可将一张包含 4 张收据的合成照片，通过以下 4 个步骤完成数据数字化：

1. **多收据检测 & 透视矫正** — 使用 YOLOv8m-Detect 检测每张收据的边界框和 4 个角点，通过透视变换进行正面校正
2. **方向分类 & 矫正** — 使用 ResNet34 判断收据旋转角度（0°/90°/180°/270°），并校正为正向
3. **文字区域检测** — 使用 YOLOv5/YOLOv9 检测日期、店名、电话号码、合计金额 4 类文字区域
4. **OCR 识别 & CSV 输出** — 使用 PaddleOCR（PP-OCRv5）识别各区域文本，经后处理输出结构化 CSV

![OCR Pipeline Demo](receipt%20detection.gif)

### 系统架构

```
                    ┌─────────────────────────────────────┐
  合成照片           │  YOLOv8m-Detect                     │
  (4张收据)       ──▶  收据框 & 角点检测                   │
                    │  透视矫正 & 分割                      │
                    └──────────────┬──────────────────────┘
                                   │  4张独立收据
                    ┌──────────────▼──────────────────────┐
                    │  ResNet34 方向分类                   │
                    │  0°/90°/180°/270° → 0° 矫正          │
                    └──────────────┬──────────────────────┘
                                   │  4张正向收据
                    ┌──────────────▼──────────────────────┐
                    │  YOLOv5 / YOLOv9 文字区域检测        │
                    │  4类: 日期/店名/电话/合计金额          │
                    └──────────────┬──────────────────────┘
                                   │  检测框
                    ┌──────────────▼──────────────────────┐
                    │  PaddleOCR (PP-OCRv5)               │
                    │  分类别文本识别 + 后处理               │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                          CSV输出 (.csv)
```

### 处理流程

1. **Step 0 — 输入**: 上传包含 4 张收据的合成照片
2. **Step 1 — 分割 & 透视矫正**: YOLOv8m-Detect 检测每张收据位置及 4 个角点，通过透视变换生成独立矫正图像
3. **Step 2 — 方向矫正**: ResNet34 分类器判断收据方向，旋转至正向（0°）
4. **Step 3 — 文字区域检测**: YOLOv5/YOLOv9 检测日期、店名、电话、合计金额区域
5. **Step 4 — OCR 识别**: PaddleOCR 识别各区域文本，经后处理输出 CSV

---

## 项目结构

```
receipt-auto-digitizer/
├── Receipt_OCR_UI_System-main/       # [核心] OCR处理系统 + Streamlit UI
│   ├── app.py                        # Streamlit 主入口
│   ├── OCR.py                        # PaddleOCR 模块
│   ├── Classification.py             # ResNet34 方向分类器
│   ├── detect.py                     # YOLOv5 文字区域检测
│   ├── perspective_utils.py          # 透视矫正工具
│   └── README.md                     # 详细文档（日语）
│
├── Receipt_classificaion_model-main/ # 收据方向分类模型
│   ├── data_load_train.py            # 训练脚本
│   ├── prediction.py                 # 推理与矫正脚本
│   └── README.md                     # 详细文档（日语）
│
├── Receipt_detection_model-main/     # 收据文字区域检测模型
│   ├── detect.py                     # YOLOv9 推理脚本
│   ├── train.py                      # 训练脚本
│   ├── best_text.pt                  # 预训练权重
│   └── README.md                     # 详细文档（日语）
│
├── multi_receipts detection/         # 多票据角点检测系统
│   ├── train/generate_data.py        # 合成数据生成
│   ├── evaluate.py                   # 评估脚本
│   ├── tools/perspective_utils.py    # 透视矫正工具
│   └── README.md                     # 详细文档（日语）
│
├── back up/                          # 备份数据（VOC2007格式）
│
├── README.md                         # 日文主文档
├── README.zh.md                      # 本文件（中文）
└── README.en.md                      # 英文文档
```

---

## 技术栈

| 类别 | 技术 | 版本 | 用途 |
|------|------|------|------|
| **目标检测** | Ultralytics YOLOv8m-Detect | >=8.0 | 收据框 & 角点检测 |
| **目标检测** | YOLOv5 / YOLOv9 (GELAN-C) | - | 文字区域检测 |
| **图像分类** | ResNet34 | - | 方向分类（4类） |
| **OCR** | PaddleOCR (PP-OCRv5) | 3.6.0 | 文本识别 |
| **深度学习** | PaddlePaddle | 3.3.1 | OCR 后端 |
| **深度学习** | PyTorch | >=2.0 | YOLO / ResNet34 后端 |
| **Web UI** | Streamlit | >=1.28 | 交互界面 |
| **图像处理** | OpenCV | >=4.5 | 透视变换 & 图像处理 |
| **图像处理** | Pillow | >=9.0 | 图像操作 |
| **数据处理** | NumPy | - | 矩阵运算 |
| **数据处理** | Pandas | >=1.5 | CSV 处理 |

---

## 环境依赖

### 系统要求

- **OS**: macOS / Linux / Windows
- **Python**: >= 3.8
- **CUDA**: 可选（推荐使用 GPU 推理）

### 安装步骤

```bash
# 1. 克隆仓库
git clone <repository-url>
cd receipt-auto-digitizer

# 2. 安装各子项目依赖（建议每个子项目使用独立虚拟环境）

# --- OCR UI 系统 ---
cd Receipt_OCR_UI_System-main
pip install streamlit ultralytics paddleocr paddlepaddle opencv-python pillow pandas torch torchvision

# --- 收据方向分类模型 ---
cd ../Receipt_classificaion_model-main
pip install torch torchvision pillow

# --- 收据文字区域检测模型 ---
cd ../Receipt_detection_model-main
pip install torch torchvision pyyaml

# --- 多票据角点检测 ---
cd "../multi_receipts detection"
pip install ultralytics pillow numpy pyyaml opencv-python
```

---

## 部署指南

### 快速启动（OCR UI 系统）

```bash
cd Receipt_OCR_UI_System-main
python -m streamlit run app.py --server.port 8503
```

浏览器访问 `http://localhost:8503`，上传合成照片运行流水线。

### 各子项目独立运行

请参考各子项目中的 README 文档。

---

## 开发规范

### 编码规范

- **语言**: Python 3.8+
- **风格**: 遵循 PEP 8
- **命名**: 采用蛇形命名法（`snake_case`）
- **类型注解**: 尽可能添加类型注解

### Git 操作

- **分支策略**: `main`（稳定版）, `develop`（开发中）, `feature/*`（功能分支）
- **提交信息**: 使用中文或英文简洁描述变更内容
- **拉取请求**: 必须经过审查，每个功能一个 PR

### 文档

- 每个子项目包含 README.md（日语）、README.zh.md（中文）、README.en.md（英文）
- 关键函数需编写 docstring

---

## 贡献指南

1. Fork 本仓库
2. 创建功能分支（`git checkout -b feature/amazing-feature`）
3. 提交更改（`git commit -m 'feat: 添加新功能'`）
4. 推送分支（`git push origin feature/amazing-feature`）
5. 创建 Pull Request

### 报告与建议

- 欢迎通过 Issue 提交 Bug 报告和功能建议
- 安全问题请直接联系维护者

---

## 许可证

本项目仅供学习和研究使用。详细信息请参考各子项目的许可协议。

---

## 维护者

- **作者**: [baiwenbin](https://github.com/baiwenbin)
- **联系方式**: 通过 GitHub Issues 联系

---

## 参考链接

- [YOLOv9](https://github.com/WongKinYiu/yolov9)
- [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [Streamlit](https://streamlit.io/)
