# 基于YOLOv9的收据文字区域检测模型

[【日本語】](README.md) | **中文** | [【English】](README.en.md)

---

## 摘要

<div style="max-width: 600px; word-wrap: break-word;">
本文记录了使用YOLO v9（GELAN-C）模型构建收据文字区域检测模型的方法。收集了约120张收据，标注了4个类别（标签）——日期（Date）、店铺名（Shopname）、电话号码（Telnum）、合计金额（Totalpay）——训练了文字区域检测模型。

结果表明，能够准确检测出指定的4个类别。同时支持在本地环境运行推理。
</div>

<div align="medium">
    <img src="images/result1.jpg" alt="检测结果1" width="100%">
</div>

本项目参考自 [YOLOv9](https://github.com/WongKinYiu/yolov9)。

---

## 简介

### 关于 YOLO v9
<div style="max-width: 600px; word-wrap: break-word;">
YOLO 是"You Only Look Once"的缩写，是以速度和精度著称的先进目标检测模型。第9版（v9）引入了"Programmable Gradient Information（PGI）"和"Generalized Efficient Layer Aggregation Network（GELAN）"等新架构，功能和性能得到了进一步增强。本项目使用的是 GELAN-C 架构。
</div>

---

## 准备工作

<div style="max-width: 600px; word-wrap: break-word;">

### 使用 Google Colab 进行训练

Google Colab 是一个提供免费 GPU 访问的云端平台，本文使用 Google Colab 的 T4 GPU 进行深度学习模型的训练。

### 数据准备

#### 1. 收集收据数据

自行收集了约120张以上购物收据，从中筛选出质量较好的图片使用。

#### 2. 使用 LabelImg 进行标注

标注工具使用 LabelImg，采用 VOC XML 格式进行标注，定义了以下4个类别：

| 类别ID | 标签名 | 说明 |
|:-----:|:------:|:----:|
| 0 | Date | 日期 |
| 1 | Shopname | 店铺名 |
| 2 | Telnum | 电话号码 |
| 3 | Totalpay | 合计金额 |

#### 3. 数据集构成

标注完成后，将数据集分为训练集、验证集和测试集。图像统一缩放至 640x640 尺寸用于训练。

</div>

---

## 训练流程

以下是在 Colab 上使用 YOLO v9 模型训练数据集的步骤。

### 1. 连接 Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. 克隆 YOLOv9 仓库并安装必要包

```python
!git clone https://github.com/WongKinYiu/yolov9
%cd yolov9
!pip install -r requirements.txt -q
```

### 3. 下载预训练模型

下载预训练的 GELAN-C 权重。

```python
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt
```

### 4. 准备数据集

将 LabelImg 创建的 VOC XML 格式标注转换为 YOLO 格式，按以下目录结构上传至 Colab。

```
dataset/
├── images/
│   ├── train/    (训练图像)
│   ├── val/      (验证图像)
│   └── test/     (测试图像)
└── labels/
    ├── train/    (训练标签)
    ├── val/      (验证标签)
    └── test/     (测试标签)
```

### 5. 开始训练

使用以下命令执行模型训练，共训练100个epoch，批次大小为8。

```python
%cd {HOME}/yolov9

!python train.py \
--batch 8 --epochs 100 --img 640 --device 0 --min-items 0 --close-mosaic 15 \
--data /content/dataset/data.yaml \
--weights {HOME}/weights/gelan-c.pt \
--cfg models/detect/gelan-c.yaml \
--hyp hyp.scratch-high.yaml
```

---

## 训练结果

经过100个epoch的训练，所有类别均达到了较高的平均精度（mAP）。

| 类别 | 精确率 (Precision) | 召回率 (Recall) | mAP@0.5 |
|:-----:|:------------------:|:---------------:|:-------:|
| Date | 0.995 | 1.000 | 0.995 |
| Shopname | 0.995 | 1.000 | 0.995 |
| Telnum | 0.995 | 1.000 | 0.995 |
| Totalpay | 0.995 | 1.000 | 0.995 |
| **整体** | **0.988** | **0.989** | **0.993** |

---

## 使用测试图片验证

使用训练好的模型对测试图片进行检测验证。

```python
!python detect.py \
--img 640 --conf 0.5 --device 0 \
--weights {HOME}/yolov9/runs/train/exp/weights/best.pt \
--source {HOME}/yolov9/dataset/test/images
```

### 本地环境推理

将训练好的权重文件（`best_text.pt`）放置在本地，通过以下命令执行推理：

```bash
python detect.py \
    --weights best_text.pt \
    --source test_images \
    --data data/text_data.yaml \
    --conf-thres 0.1 \
    --save-txt \
    --save-crop \
    --project runs/detect \
    --name text_result \
    --exist-ok
```

<div align="medium">
    <img src="images/result1.jpg" alt="检测结果1" width="48%">
    <img src="images/result2.jpg" alt="检测结果2" width="48%">
</div>

<div align="medium">
    <img src="images/result3.jpg" alt="检测结果3" width="48%">
    <img src="images/result4.jpg" alt="检测结果4" width="48%">
</div>

从上述结果可以看出，即使在较低的置信度阈值（conf-thres=0.1）下，也能完整检测出全部4个类别，无漏检。

---

## 项目结构

```
Receipt_detection_model-main/
├── data/
│   └── text_data.yaml        # 类别定义文件（4类）
├── images/                   # 检测结果图片
├── models/
│   ├── common.py             # 公共模块
│   ├── experimental.py       # 实验性模块
│   └── yolo.py               # YOLO模型定义
├── test_images/              # 推理用测试图片
├── utils/                    # 工具库
├── best_text.pt              # 训练好的权重文件
├── detect.py                 # 推理脚本
├── train.py                  # 训练脚本
└── val.py                    # 验证脚本
```

---

## 参考

<details><summary> <b>展开</b> </summary>

* [https://github.com/WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)
* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/VDIGPKU/DynamicDet](https://github.com/VDIGPKU/DynamicDet)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [LabelImg](https://github.com/tzutalin/labelImg)
</details>
