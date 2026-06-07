# YOLOv9で構築した領収書の文字領域検出モデル

[【日本語】](README.md) | [【中文】](README.zh.md) | [【English】](README.en.md)

---

## Summary

<div style="max-width: 600px; word-wrap: break-word;">
本記事ではYOLO v9（GELAN-C）モデルを利用して、領収書の文字領域検出モデルの構築方法について記録します。約120枚の領収書を収集し、4つのラベル（クラス）——『日付（Date）』、『店舗名（Shopname）』、『電話番号（Telnum）』、『合計金額（Totalpay）』——を付けてデータをアノテーションし、文字領域の検出モデルを学習しました。

結果として、指定した4つのクラスを正確に検出することができました。ローカル環境でも推論を実行可能です。
</div>

<div align="medium">
    <img src="images/result1.jpg" alt="検出結果1" width="100%">
</div>

このプロジェクトは、[YOLOv9](https://github.com/WongKinYiu/yolov9) を参考にして作成されています。

---

## はじめに

### YOLO v9について
<div style="max-width: 600px; word-wrap: break-word;">
YOLOは「You Only Look Once」の略で、速度と精度の高さで知られる最先端のオブジェクト検出モデルです。このモデルの第9バージョン（v9）は、「Programmable Gradient Information（PGI）」や「Generalized Efficient Layer Aggregation Network（GELAN）」といった新しいアーキテクチャを導入し、機能と性能がさらに強化されています。本プロジェクトではGELAN-Cアーキテクチャを使用しています。
</div>

---

## 準備

<div style="max-width: 600px; word-wrap: break-word;">

### 学習はGoogle Colabを使用します。

Google Colabは、GPUへの無料アクセスを提供するクラウドベースのプラットフォームであり、この記事は、Google ColabのGPU（T4）を使って、ディープラーニングモデルの学習を行なっていました。

### データ準備

#### 1. 領収書データを収集

自分で買い物の領収書を約120枚以上収集し、その中から品質の良いものを選別して使用しました。

#### 2. LabelImgでアノテーション

アノテーションツールにはLabelImgを使用しました。VOC XML形式でアノテーションを行い、4つのクラスを定義しました。

| クラスID | ラベル名 | 説明 |
|:--------:|:---------:|:----:|
| 0 | Date | 日付 |
| 1 | Shopname | 店舗名 |
| 2 | Telnum | 電話番号 |
| 3 | Totalpay | 合計金額 |

#### 3. データセット構成

アノテーション完了後、データセットを学習用、検証用、テスト用に分割しました。画像は640x640サイズにリサイズして学習に使用しました。

</div>

---

## 学習の流れ

Colab上でYOLO v9モデルを使用してデータセットを学習する手順を説明します。

### 1. Google Driveを接続

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. YOLOv9のリポジトリをクローンと必要なパッケージをインストール

```python
!git clone https://github.com/WongKinYiu/yolov9
%cd yolov9
!pip install -r requirements.txt -q
```

### 3. モデルをダウンロード

事前学習済みのGELAN-C重みをダウンロードします。

```python
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt
```

### 4. データセットの準備

LabelImgで作成したVOC XML形式のアノテーションをYOLO形式に変換し、以下のようなディレクトリ構成でColabにアップロードします。

```
dataset/
├── images/
│   ├── train/    (学習用画像)
│   ├── val/      (検証用画像)
│   └── test/     (テスト用画像)
└── labels/
    ├── train/    (学習用ラベル)
    ├── val/      (検証用ラベル)
    └── test/     (テスト用ラベル)
```

### 5. 学習

以下のコマンドでモデルの学習を実行します。バッチサイズ8、エポック数100で学習を行いました。

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

## 学習の結果

100エポックの学習の結果、全クラスにおいて良好な平均適合率（mAP）を達成しました。

| クラス | 適合率 (Precision) | 再現率 (Recall) | mAP@0.5 |
|:-----:|:------------------:|:---------------:|:-------:|
| Date | 0.995 | 1.000 | 0.995 |
| Shopname | 0.995 | 1.000 | 0.995 |
| Telnum | 0.995 | 1.000 | 0.995 |
| Totalpay | 0.995 | 1.000 | 0.995 |
| **全体** | **0.988** | **0.989** | **0.993** |

---

## テストイメージを使って検証

学習したモデルを使って、テスト画像で検出を検証します。

```python
!python detect.py \
--img 640 --conf 0.5 --device 0 \
--weights {HOME}/yolov9/runs/train/exp/weights/best.pt \
--source {HOME}/yolov9/dataset/test/images
```

### ローカル環境での推論

学習済みの重みファイル（`best_text.pt`）をローカルに配置し、以下のコマンドで推論を実行できます。

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
    <img src="images/result1.jpg" alt="検出結果1" width="48%">
    <img src="images/result2.jpg" alt="検出結果2" width="48%">
</div>

<div align="medium">
    <img src="images/result3.jpg" alt="検出結果3" width="48%">
    <img src="images/result4.jpg" alt="検出結果4" width="48%">
</div>

上記の結果から、低い信頼度閾値（conf-thres=0.1）でも4つのクラス全てを漏れなく検出できていることが確認できます。

---

## プロジェクト構成

```
Receipt_detection_model-main/
├── data/
│   └── text_data.yaml        # クラス定義ファイル（4クラス）
├── images/                   # 検出結果画像
├── models/
│   ├── common.py             # 共通モジュール
│   ├── experimental.py       # 実験的モジュール
│   └── yolo.py               # YOLOモデル定義
├── test_images/              # 推論用テスト画像
├── utils/                    # ユーティリティ
├── best_text.pt              # 学習済み重みファイル
├── detect.py                 # 推論スクリプト
├── train.py                  # 学習スクリプト
└── val.py                    # 検証スクリプト
```

---

## Reference

<details><summary> <b>Expand</b> </summary>

* [https://github.com/WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)
* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/VDIGPKU/DynamicDet](https://github.com/VDIGPKU/DynamicDet)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [LabelImg](https://github.com/tzutalin/labelImg)
</details>
