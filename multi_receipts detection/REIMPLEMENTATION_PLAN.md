# 项目复现与模型重新训练实施方案

## 一、项目精简后目录结构（目标）

```
multi_receipts detection/
├── v3_yolov8_detect_corners/           # 主项目目录
│   ├── train/                          # [新增] 训练流程
│   │   ├── generate_data.py            #     数据生成脚本（从v3精简）
│   │   ├── yolo_dataset/               #     生成的训练/验证数据集
│   │   │   ├── images/train/
│   │   │   ├── images/val/
│   │   │   ├── labels/train/
│   │   │   └── labels/val/
│   │   └── data.yaml                   #     数据集配置文件
│   │
│   ├── evaluate/                       # [新增] 评估流程
│   │   ├── evaluate_model.py           #     mAP标准评估
│   │   ├── evaluate_crop_rectify.py    #     合成测试集评估
│   │   └── evaluate_real_test.py       #     真实图片对比测试
│   │
│   ├── final_test/                     # [保留] 真实测试图片
│   │   ├── Image_20260523093200_41_257.jpg
│   │   ├── Image_20260523093201_42_257.jpg
│   │   └── Image_20260523093202_43_257.jpg
│   │
│   ├── model/                          # [新增] 模型权重
│   │   └── best_v8_300.pt
│   │
│   ├── tools/                          # [新增] 工具模块
│   │   └── perspective_utils.py
│   │
│   └── README.md                       # 项目描述
│
└── REIMPLEMENTATION_PLAN.md            # 本文件
```

---

## 二、数据集构建模块

### 2.1 数据来源

源票据图片从备份的 v1/v2 数据中获取。`generate_data_corners.py` 当前的 `INPUT_DIR` 指向外部目录，需要改为指向 v1 数据集中的单张票据图片。

v1 备份数据中每个 `receipt_*.jpg` 本身就是单张裁切好的小票图片，适合作为合成素材。共约 **248 张** 可用。

### 2.2 生成配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `NUM_SAMPLES` | **10**（示例）→ **300**（正式训练） | 合成图数量 |
| `TRAIN_RATIO` | 0.8 | 训练/验证划分 |
| `NUM_PER_IMG` | 4 | 每张合成图拼贴 4 张票据 |
| 每张标签 | 5 类 × 4 票据 = 20 个框 | receipt(0) + corner_tl/tr/br/bl(1-4) |

### 2.3 实施步骤

```bash
# 1. 复制备用的单张票据图片到项目内
cp -r "/Users/...back up/项目迭代文件/multi_receipt_detection/v1_yolov8_pose/yolo_dataset_v2/images/train" \
      "./v3_yolov8_detect_corners/train/source_images/"

# 2. 修改 generate_data.py 中的 INPUT_DIR 指向 source_images/
#    → 同时删除外部路径依赖，改为相对路径

# 3. 运行数据生成（先用 10 张验证）
cd v3_yolov8_detect_corners/train/
python generate_data.py          # NUM_SAMPLES=10 快速验证
                                 # NUM_SAMPLES=300 正式训练
```

### 2.4 剔除的冗余文件

| 文件 | 处理 |
|------|------|
| `training_results_predict/` | 已在备份中，不保留 |
| `v1_yolov8_pose/` | 已在备份中，不保留 |
| `v2_yolov8_pose_upgrade/` | 已在备份中，不保留 |
| `crop_rectify_eval/` 评估结果图 | 运行时生成，不保留 |
| `real_test_compare/` 评估结果图 | 运行时生成，不保留 |
| `best_v8_100.pt` | 早期实验版本，不保留 |

保留 `final test/` 3 张真实测试图片用于评估。

---

## 三、配置文件准备（data.yaml）

`generate_data.py` 脚本末尾已自动生成 `data.yaml`，内容如下：

```yaml
path: ./v3_yolov8_detect_corners/train/yolo_dataset
train: images/train
val: images/val
nc: 5
names: ['receipt', 'corner_tl', 'corner_tr', 'corner_br', 'corner_bl']
```

训练时直接使用此配置文件：

```bash
yolo train model=yolov8m.pt data=./v3_yolov8_detect_corners/train/yolo_dataset/data.yaml \
      epochs=100 imgsz=1280 batch=8
```

---

## 四、测试集部署

### 4.1 测试集路径

`final_test/` 目录已包含 3 张真实拍摄的合成图片：

```
v3_yolov8_detect_corners/final_test/
├── Image_20260523093200_41_257.jpg
├── Image_20260523093201_42_257.jpg
└── Image_20260523093202_43_257.jpg
```

### 4.2 评估脚本调用

修改 `evaluate_real_test.py` 中的 `INPUT_DIR` 为相对路径：

```python
INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "final_test")
```

同样修改 `evaluate_crop_rectify.py` 中的 `REAL_TEST_DIR`。

---

## 五、流程优化（精简后的工作流）

### 5.1 完整工作流

```
① 数据生成                         ② 模型训练                   ③ 评估
──────                             ──────                        ──────
source_images/  ─▶ generate_data.py ─▶ yolo train         ─▶ evaluate_model.py
   (v1备份)         │                    │                      │
                    │                    ▼                      ▼
                    ▼                best.pt              evaluate_crop_rectify.py
               yolo_dataset/                               evaluate_real_test.py
               (images+labels+
                data.yaml)                                 final_test/
                                                              (测试图片)
```

### 5.2 目录结构规范（绝对路径 → 相对路径）

当前所有脚本使用硬编码绝对路径，需要改为相对路径：

```python
# 改为基于脚本所在目录的相对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "train", "source_images")
DATASET_ROOT = os.path.join(BASE_DIR, "train", "yolo_dataset")
```

### 5.3 保留/移除清单

| 类型 | 保留 | 移除（已备份） |
|------|------|---------------|
| 代码 | `generate_data.py`, `perspective_utils.py` | `generate_data_corners.py`（整合为 generate_data.py） |
| 评估 | `evaluate_model.py`, `evaluate_crop_rectify.py`, `evaluate_real_test.py` | - |
| 模型 | `best_v8_300.pt` | `best_v8_100.pt` |
| 数据 | `final_test/` 测试图片 | `crop_rectify_eval/`, `real_test_compare/` |
| 工具 | `README.md` | `perspective_utils.py` 保留在 tools/ |

### 5.4 快速复现命令

```bash
# 1. 数据生成
cd v3_yolov8_detect_corners/
python train/generate_data.py

# 2. 模型训练
yolo train model=yolov8m.pt \
      data=train/yolo_dataset/data.yaml \
      epochs=100 imgsz=1280 batch=8 \
      project=. name=trained_model

# 3. 裁剪矫正评估
python evaluate/evaluate_crop_rectify.py

# 4. 真实图片对比评估
python evaluate/evaluate_real_test.py

# 5. mAP 评估
python evaluate/evaluate_model.py
```

---

## 六、实施优先级

| 优先级 | 任务 | 工作量 |
|--------|------|--------|
| P0 | 修改 `generate_data.py` INPUT_DIR 为相对路径 + 从 v1 备份取素材 | 小 |
| P0 | 生成 10 张示例数据集验证流程 | 中 |
| P0 | 修改 3 个评估脚本的路径为相对路径 | 小 |
| P1 | 按新目录结构调整文件位置 | 中 |
| P1 | 生成 300 张完整训练集 | 大 |
| P2 | 实际运行训练并导出新模型 | 大 |
