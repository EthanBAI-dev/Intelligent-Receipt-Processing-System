# 收据OCR处理系统

> **语言**: [日本語](README.md) | [English](README_EN.md)

基于YOLOv8、YOLOv5、ResNet34、PaddleOCR的多收据检测·分类·文字区域检测·OCR识别流水线。
使用Streamlit构建直观的Web界面。

---

## 系统架构

输入**4张收据拼接的合成照片**，经过分割、倾斜校正、文字区域检测、OCR识别，输出结构化CSV。

```
                    ┌─────────────────────────────────────┐
  合成照片            │  YOLOv8m-Detect (best_v8_300.pt)   │
  (4张收据)        ──▶  分割 & 透视校正                      │
                    └──────────────┬──────────────────────┘
                                   │  4张独立收据
                    ┌──────────────▼──────────────────────┐
                    │  ResNet34 分类                       │
                    │  方向校正 (0°/90°/180°/270°)          │
                    └──────────────┬──────────────────────┘
                                   │  4张正立收据
                    ┌──────────────▼──────────────────────┐
                    │  YOLOv5 检测 (runs/train/exp)        │
                    │  4类: 日期/店名/电话/合计金额           │
                    └──────────────┬──────────────────────┘
                                   │  检测框
                    ┌──────────────▼──────────────────────┐
                    │  PaddleOCR (PP-OCRv5, lang=japan)   │
                    │  分类别文本识别 + 后处理               │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                          CSV输出 (.csv)
```

## 项目结构

```
Receipt_OCR_UI_System-main/
├── app.py                     # Streamlit 主入口 (Full Pipeline + 各步骤)
├── OCR.py                     # PaddleOCR 模块 (全局单例, 4类后处理)
├── Classification.py          # ResNet34 方向分类器 (0°/90°/180°/270°)
├── detect.py                  # YOLOv5 推理脚本 (文字区域检测)
├── perspective_utils.py       # 透视校正共享模块 (角点法 + OpenCV回退)
├── best_v8_300.pt             # YOLOv8m-Detect: 收据 + 角点检测模型
│
├── logs/
│   └── Epoch82-Total_Loss0.0004.pth   # ResNet34 分类权重
│
├── runs/
│   ├── train/exp/weights/best.pt      # YOLOv5 文字检测权重
│   └── detect/exp*/                    # 检测输出 (自动递增)
│       ├── corrected_image_*.jpg       #   检测标注图
│       └── labels/
│           └── corrected_image_*.txt   #   YOLO格式标签
│
├── data/
│   ├── corrected_images/               # Step 2 输出: 方向校正后收据
│   ├── test/images/                    # 旧测试图片目录
│   ├── pipeline_temp/classified/       # 流水线临时分类结果
│   ├── prediction/                     # 分类输入目录
│   └── ocr_upload/                     # 历史上传目录
│
└── ocr_results*.csv                    # 最终OCR输出 (自动编号)
```

## 模型一览

### 1. 收据检测 & 分割

| 项目 | 值 |
|----------|-------|
| 模型 | YOLOv8m-Detect |
| 权重 | `best_v8_300.pt` |
| 类别数 | 5: `receipt(0)`, `corner_tl(1)`, `corner_tr(2)`, `corner_br(3)`, `corner_bl(4)` |
| 训练数据 | 300张合成照片, 每张4张收据 |
| 置信度 | 0.25 |
| NMS IoU | 0.45 |

**透视校正** (`perspective_utils.py`):

| 方式 | 触发条件 | 说明 |
|--------|---------|-------------|
| **角点法** | 4个角点 + 验证通过 | 直接使用class ID (TL/TR/BR/BL) 构建四边形，无需几何排序 |
| **OpenCV回退** | 角点不足 或 验证失败 | CLAHE + 自适应二值化 + 轮廓四边形检测 |

**验证项**: 边界检查、自交检测、面积 > 100px、宽高比 < 8:1。

### 2. 方向分类

| 项目 | 值 |
|----------|-------|
| 模型 | ResNet34 |
| 权重 | `logs/Epoch82-Total_Loss0.0004.pth` |
| 输入 | 224×224, [-1, 1] 归一化 |
| 类别 | 0=0°正立, 1=90°逆时针, 2=180°, 3=270°逆时针 |

### 3. 文字区域检测

| 项目 | 值 |
|----------|-------|
| 框架 | YOLOv5 |
| 权重 | `runs/train/exp/weights/best.pt` |
| 输入 | 640×640 |
| 置信度 | 0.5 |

| 类别ID | 项目 | CSV列名 | PaddleOCR 后处理 |
|:--------:|-------|:----------:|---------------------------|
| 0 | 日期 | `date` | 仅去空格 (PP-OCRv5 原生处理日期格式) |
| 1 | 店名 | `store` | 保留假名/汉字 + 字母数字 |
| 2 | 电话号码 | `phone` | 保留数字 + `- ( ) . :` + TEL |
| 3 | 合计金额 | `total` | 右50%裁剪 (¥数值部分), 保留数字 + ¥$ |

### 4. OCR识别

| 项目 | 值 |
|----------|-------|
| 引擎 | PaddleOCR 3.6.0 |
| 模型 | PP-OCRv5_server_det + PP-OCRv5_server_rec + PP-LCNet |
| 语言 | `japan` (日语 + 英语) |
| 实例 | 全局单例 (`get_ocr()`), 模块加载时初始化一次 |
| 缓存 | `~/.paddlex/` 自动下载并缓存 |
| 输出 | UTF-8 BOM CSV, 自动编号 (`ocr_results1.csv`, …) |

## CSV输出格式

| 列 | 类型 | 示例 |
|--------|------|---------|
| `image_name` | string | `corrected_image_1.jpg` |
| `date` | string | `2026年03月21日` |
| `store` | string | `Beisie 吉井店` |
| `phone` | string | `027-387-6511` |
| `total` | string | `￥1,529` |

## 使用方法

### 快速开始

```bash
# 安装依赖
pip install streamlit ultralytics paddleocr paddlepaddle pillow opencv-python pandas

# 启动服务
cd Receipt_OCR_UI_System-main
python -m streamlit run app.py --server.port 8503
```

### 侧边栏模式

| 模式 | 说明 |
|------|-------------|
| **Full Pipeline** | 上传合成照片 → 分割 → 分类 → 检测 → OCR → CSV (一键) |
| **Multi Receipts Detection** | 合成照片分割 → 裁剪 → 透视校正 |
| **Classification** | 单张图片方向校正 (0°/90°/180°/270°) |
| **Detection** | 对校正后图片运行YOLOv5文字区域检测 |
| **OCR** | 对已有检测结果运行PaddleOCR → CSV |

### Full Pipeline 显示布局

每个处理步骤以 **4列** 展示（每列一张收据），纵向对齐:

```
┌──────────┬──────────┬──────────┬──────────┐
│   Step 0: 原始输入图像                      │
├──────────┼──────────┼──────────┼──────────┤
│   Step 1: 裁剪 (4张独立收据)                │
├──────────┼──────────┼──────────┼──────────┤
│   Step 1: 透视校正 [方式]                   │
├──────────┼──────────┼──────────┼──────────┤
│   Step 2: 方向校正                          │
├──────────┼──────────┼──────────┼──────────┤
│   Step 3: 文字区域检测                      │
├──────────┼──────────┼──────────┼──────────┤
│   Step 4: OCR结果 CSV (可下载)              │
└──────────┴──────────┴──────────┴──────────┘
```

每个文件之间用 **红色水平分割线** 分隔。

## 文件存储规则

| 类别 | 路径 | 命名 | 保留周期 |
|----------|------|--------|-----------|
| 检测模型 | `./best_v8_300.pt` | 固定 | 永久 |
| 透视校正模块 | `./perspective_utils.py` | 固定 | 永久 |
| 分类权重 | `./logs/Epoch82-*.pth` | 固定 | 永久 |
| 检测权重 | `./runs/train/exp/weights/best.pt` | 固定 | 永久 |
| 校正后图片 | `./data/corrected_images/` | `corrected_image_{i}.jpg` | 每次运行覆盖 |
| 检测输出 | `./runs/detect/exp{n}/` | 自动递增 | 永久 |
| 检测标签 | `./runs/detect/exp{n}/labels/` | `corrected_image_{i}.txt` | 永久 |
| OCR结果 | `./` | `ocr_results{n}.csv` | 自动编号, 永久 |
| PaddleOCR模型 | `~/.paddlex/official_models/` | 自动下载 | 无限期缓存 |
| 上传原图 | 仅内存 | — | 会话期间 |

## 关键设计决策

1. **OCR全局单例** — `PaddleOCR` 通过 `get_ocr()` 仅初始化一次，避免重复加载模型开销。

2. **基于class ID的透视校正** — 直接使用角点class ID (1=TL, 2=TR, 3=BR, 4=BL) 构建四边形，避免几何排序在旋转/倾斜收据上失败的固有问题。

3. **自动OpenCV回退** — 角点不足或验证失败时，透明回退到OpenCV轮廓法实现透视校正。

4. **宽高比过滤** — 丢弃宽高比在 `[0.20, 0.95]` 之外的收据框，消除图像边缘窄条的误检。

5. **X范围约束的角点分组** — `group_corners_to_receipt()` 要求角点x坐标必须在收据框 `[x1 - 0.4*w, x2 + 0.4*w]` 范围内才进行匹配，防止跨收据的角点误分配。

6. **分类别OCR后处理** — 4类文字区域各有专属文本清洗规则（日期:最小处理、店名:日语字符、电话:数字+分隔符、合计:右半裁剪提取数值）。

## 依赖包

| 包 | 版本 | 用途 |
|---------|---------|---------|
| streamlit | >=1.28 | Web界面 |
| ultralytics | >=8.0 | YOLOv8推理 |
| paddleocr | 3.6.0 | OCR识别 |
| paddlepaddle | 3.3.1 | PaddlePaddle后端 |
| opencv-python | >=4.5 | 图像处理 & 透视变换 |
| pillow | >=9.0 | PIL图像操作 |
| pandas | >=1.5 | CSV处理 & DataFrame展示 |
| torch | >=2.0 | ResNet34分类 |
| torchvision | >=0.15 | 分类图像变换 |

## 参考

- [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics)
- [PaddleOCR / PP-OCRv5](https://github.com/PaddlePaddle/PaddleOCR)
- [Streamlit](https://streamlit.io/)
