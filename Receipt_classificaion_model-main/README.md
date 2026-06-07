[English](README.en.md) | [中文](README.zh.md) | **日本語**

# **領収書画像回転補正モデル（分類モデル）**

## **概要**
本プロジェクトでは、**ResNet34** を基盤とした分類モデルを使用し、領収書画像の回転角度（0°, 90°, 180°, 270°）を判別します。その後、画像の角度を自動的に 0° に補正します。

下記の例では、270°回転した画像をモデルで検出し、0° に自動補正しています。

<div align="medium">
  <img src="figure/prediction2.png", width="100%"> 
</div>

## **プロジェクト構成**

```
├── data/
│   ├── train/images/      # 訓練用画像
│   ├── test/images/       # テスト用画像
│   └── valid/images/      # 検証用画像
├── datamodule/
│   └── dataloader.py      # データセットと前処理
├── nets/
│   ├── blocks.py          # 畳み込みブロック定義
│   ├── resnet.py          # ResNet34 モデル定義
│   └── README.md
├── plotmodule/
│   └── plot_utils.py      # 可視化ユーティリティ
├── debug_images/
│   └── debug_utils.py     # デバッグ用画像保存
├── figure/                # README 用画像
├── logs/                  # 訓練済みモデル（.pth）
├── data_load_train.py     # 訓練スクリプト
├── prediction.py          # 推論・補正スクリプト
├── README.md              # 本ファイル（日本語）
├── README.en.md           # 英語版
└── README.zh.md           # 中国語版
```

## **データセットの準備**

学習用画像を `data/train/images/`、テスト用画像を `data/test/images/` に配置します。各画像は正方向（0°）で用意し、コード内で自動的に 90°, 180°, 270° の回転バージョンが生成され、4クラスのデータセットとなります。

- **クラス 0**: 0°（正方向）
- **クラス 1**: 90° 左回転
- **クラス 2**: 180° 回転
- **クラス 3**: 270° 左回転（= 90° 右回転）

<img width="800" height="500" src=figure/3.png/> 

## **処理の流れ**

### 1. データ前処理 — [dataloader.py](datamodule/dataloader.py)

`RotatedReceiptDataset` クラスが画像を読み込み、各画像から4つの回転バージョンを生成します。前処理として：

- **リサイズ**: すべての画像を 224×224 ピクセルに統一
- **正規化**: 各 RGB チャンネルを `[0.5, 0.5, 0.5]` で正規化

```python
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
```

### 2. 学習と検証 — [data_load_train.py](data_load_train.py)

| 設定 | 値 |
|------|-----|
| モデル | ResNet34 |
| 損失関数 | CrossEntropyLoss |
| オプティマイザ | Adam |
| 学習率 | 0.001（初期） |
| スケジューラ | StepLR（gamma=0.92） |
| バッチサイズ | 8 |
| エポック数 | 10（カスタマイズ可） |

学習後、モデルは `logs/` ディレクトリに保存されます。

<div align="medium">
  <img src="figure/trainresult.png", width="100%"> 
</div>

### 3. 推論と角度補正 — [prediction.py](prediction.py)

訓練済みモデルを使用して画像の回転角度を予測し、0° に補正します。

```python
# 予測ラベルに基づいて画像を回転補正
def correct_image_orientation(image, predicted_label):
    if predicted_label == 1:
        return image.rotate(270, expand=True)  # 90° → 0°
    elif predicted_label == 2:
        return image.rotate(180, expand=True)  # 180° → 0°
    elif predicted_label == 3:
        return image.rotate(90, expand=True)   # 270° → 0°
    return image  # すでに 0°
```

## **実行方法**

```bash
# 訓練
python3 data_load_train.py

# 推論
python3 prediction.py
```

## **モデル重みファイル**

訓練済みモデル: `logs/Epoch82-Total_Loss0.0004.pth`（ResNet34, 4クラス分類, 損失 0.0004）

## **参考リンク**

- [ResNet50-MNIST-pytorch](https://github.com/wangyunjeff/ResNet50-MNIST-pytorch/tree/master)
