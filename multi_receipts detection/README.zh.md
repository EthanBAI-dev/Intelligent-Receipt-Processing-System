# 多票据角点检测系统

[日本語](README.md) | [English](README.en.md)

---

## 1. 项目核心功能与定位

本项目是一个基于 **YOLOv8m-Detect** 的多票据角点检测系统，专门用于从**4拼合合成图片**中精确检测每张票据的位置及其四个角点，为后续的透视矫正和 OCR 识别提供高精度的定位基础。

**核心用途**：
- 从一张包含 4 张票据的合成图中，检测出每张票据的边界框
- 检测每张票据的 4 个角点（左上、右上、右下、左下）
- 为透视矫正提供角点坐标，实现票据的正面校正
- 本项目是一个**基础检测模型**，已被集成到完整的 OCR 识别系统中

---

## 2. 核心技术方法与实现逻辑

### 技术选型

| 技术 | 用途 |
|------|------|
| **YOLOv8m-Detect** | 目标检测模型，同时检测票据框和角点框 |
| **PIL / OpenCV** | 图像透视变换与处理 |
| **NumPy** | 矩阵运算（单应性矩阵计算） |
| **SAHI** | 切片辅助推理（v1/v2 阶段历史使用） |

### 模型输出（5 类）

| Class ID | 名称 | 说明 |
|----------|------|------|
| 0 | receipt | 票据整体框 |
| 1 | corner_tl | 左上角点 |
| 2 | corner_tr | 右上角点 |
| 3 | corner_br | 右下角点 |
| 4 | corner_bl | 左下角点 |

### 数据生成逻辑

`train/generate_data.py` 将多张单票据图片通过以下步骤合成为训练数据：

1. **源图片采集** — 从 `train/source_images/` 读取单张裁切好的票据图片
2. **透视变换** — 每张票据随机施加 pitch/yaw ±12° 的 3D 透视变换 + 局部微变形
3. **随机旋转** — 70% 概率 ±6° 旋转以增强数据多样性
4. **裁剪偏移** — 自动去除变换后的空白边缘
5. **水平拼接** — 4 张处理后的票据拼贴到灰色画布，间隔 120px
6. **标签生成** — 每张票据标注 5 个框（1 个 receipt + 4 个角点）
7. **自动压缩** — 所有图片最长边压缩至 1280px

### 三级透视矫正策略

`tools/perspective_utils.py` 中的 `rectify_receipt()` 提供三级回退：

1. **角点法**（优先）— 使用 4 个检测到的角点框中心做四点透视变换（含 5 项验证）
2. **OpenCV 轮廓法**（回退）— CLAHE + 自适应阈值 + 形态学操作 + 最大四边形拟合
3. **原图返回**（最终回退）

---

## 3. 迭代历程

### v1 — YOLOv8-Pose 关键点方案（已归档）

- **方案**：使用 YOLOv8-Pose 模型，角点作为关键点 (keypoints) 回归
- **数据**：~150 张合成训练图
- **问题**：Pose 关键点回归精度有限，且推理时无法输出置信度

### v2 — Pose 数据增强版（已归档）

- **方案**：与 v1 相同的 Pose 关键点方式
- **数据**：~500 张合成训练图
- **模型**：`best_2.pt`
- **问题**：Pose 方案的固有限制仍然存在

### v3 — YOLOv8-Detect 角点检测方案（当前生产版本）

- **关键变更**：将角点从 Pose 关键点改为**独立的检测目标（Detect 框）**
- **优势**：每个角点独立输出置信度，支持 NMS 过滤，精度更高
- **数据**：300 张合成训练图
- **模型**：`model/best_v8_300.pt`
- **状态**：生产使用，已集成到 OCR 系统

### 版本对比

| 版本 | 方案 | 数据量 | 角点方式 | 透视矫正成功率 | 状态 |
|------|------|--------|----------|---------------|------|
| v1 | YOLOv8-Pose | ~150 张 | 关键点 | 较低 | 已归档 |
| v2 | YOLOv8-Pose | ~500 张 | 关键点 | 较低 | 已归档 |
| **v3** | **YOLOv8-Detect** | **300 张** | **独立检测框** | **~70% 角点法 + OpenCV 回退** | **当前生产** |

> 历史版本已归档到 `back up/项目迭代文件/multi_receipt_detection/`

---

## 4. 安装部署指南

### 环境要求

- Python >= 3.8
- 操作系统：macOS / Linux / Windows

### 目录结构

```
multi_receipts detection/
├── train/                           # 训练流程
│   ├── generate_data.py             #     数据生成脚本
│   ├── source_images/               #     源票据图片（10 张示例）
│   └── yolo_dataset/                #     生成的合成数据集
│       ├── data.yaml                #     数据集配置文件
│       ├── images/train/            #     训练图片
│       ├── images/val/              #     验证图片
│       ├── labels/train/            #     训练标签
│       └── labels/val/              #     验证标签
├── model/                           # 模型权重
│   └── best_v8_300.pt               #     生产模型
├── tools/                           # 工具模块
│   └── perspective_utils.py         #     透视矫正工具函数
├── final_test/                      # 真实场景测试图片
│   ├── Image_1.jpg
│   ├── Image_2.jpg
│   └── Image_3.jpg
├── evaluate.py                      # 评估脚本
├── evaluate_output/                 # 评估输出结果
├── README.md                        # 日文主文档
├── README.zh.md                     # 本文件（中文）
├── README.en.md                     # 英文说明
└── REIMPLEMENTATION_PLAN.md         # 复现实施方案
```

### 安装步骤

```bash
# 1. 克隆或进入项目目录
cd multi_receipts detection/

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate   # macOS/Linux

# 3. 安装依赖
pip install ultralytics pillow numpy pyyaml opencv-python
```

### 快速开始

```bash
# 1. 数据生成（10 张示例）
cd train/
python generate_data.py

# 2. 运行推理评估
cd ..
python evaluate.py                     # 推理统计
python evaluate.py --save              # 推理 + 保存标注图

# 3. 模型训练（使用全部数据）
yolo detect train data=train/yolo_dataset/data.yaml \
      model=yolov8m.pt epochs=300 imgsz=1280 batch=8

# 4. 使用训练好的模型推理
yolo detect predict model=model/best_v8_300.pt \
      source=final_test/ save=True
```

### 测试图片

`final_test/` 目录包含 3 张真实拍摄的 4 拼合小票照片，可用于验证模型效果。

---

## 许可证

本项目仅供学习和研究使用。
