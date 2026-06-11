# レシート自動デジタル化システム（Receipt Auto Digitizer）

[**日本語**](README.md) | [中文](README.zh.md) | [English](README.en.md)

---

> 本プロジェクトは、**複数レシートの合成写真**を入力とし、**検出・分類・文字領域認識・OCR** を経て、構造化されたCSVデータを自動生成するエンドツーエンドのパイプラインです。

---

## システム概要

**Receipt Auto Digitizer** は、4枚のレシートが並んだ合成写真から、以下の4つのステップでデータをデジタル化します。

1. **マルチレシート検出 & 透視補正** — YOLOv8m-Detect で各レシートの境界枠と4隅の角点を検出し、透視変換で正面補正
2. **傾き分類 & 補正** — ResNet34 でレシートの回転角度（0°/90°/180°/270°）を判定し、正立補正
3. **文字領域検出** — YOLOv5/YOLOv9 で日付・店名・電話番号・合計金額の4クラスを検出
4. **OCR 認識 & CSV 出力** — PaddleOCR（PP-OCRv5）で各領域のテキストを認識し、構造化CSVに出力

![OCR Pipeline Demo](receipt%20detection.gif)

### システムアーキテクチャ

```
                    ┌─────────────────────────────────────┐
  合成写真           │  YOLOv8m-Detect                     │
  (4枚のレシート) ──▶  レシート枠 & 角点検出               │
                    │  透視補正 & 分割                      │
                    └──────────────┬──────────────────────┘
                                   │  4枚の個別レシート
                    ┌──────────────▼──────────────────────┐
                    │  ResNet34 傾き分類                   │
                    │  0°/90°/180°/270° → 0° に補正       │
                    └──────────────┬──────────────────────┘
                                   │  4枚の正立レシート
                    ┌──────────────▼──────────────────────┐
                    │  YOLOv5 / YOLOv9 文字領域検出        │
                    │  4クラス: 日付/店名/電話番号/合計金額  │
                    └──────────────┬──────────────────────┘
                                   │  バウンディングボックス
                    ┌──────────────▼──────────────────────┐
                    │  PaddleOCR (PP-OCRv5)               │
                    │  クラス別テキスト認識 + 後処理        │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                          CSV出力 (.csv)
```

### 処理フロー詳細

1. **Step 0 — 入力**: 4枚のレシートが敷き詰められた合成写真をアップロード
2. **Step 1 — 分割 & 透視補正**: YOLOv8m-Detect で各レシートの位置と4隅のコーナーを検出、透視変換で個別の補正済み画像を生成
3. **Step 2 — 傾き補正**: ResNet34 分類器でレシートの向きを判定し、正立状態（0°）に回転補正
4. **Step 3 — 文字領域検出**: YOLOv5/YOLOv9 で日付・店名・電話番号・合計金額の領域を検出
5. **Step 4 — OCR 認識**: PaddleOCR で各領域のテキストを認識し、後処理を経てCSVに出力

---

## プロジェクト構成

```
receipt-auto-digitizer/
├── Receipt_OCR_UI_System-main/       # [中核] OCR処理システム + Streamlit UI
│   ├── app.py                        # Streamlit メインエントリ
│   ├── OCR.py                        # PaddleOCR モジュール
│   ├── Classification.py             # ResNet34 傾き分類器
│   ├── detect.py                     # YOLOv5 文字領域検出
│   ├── perspective_utils.py          # 透視補正ユーティリティ
│   └── README.md                     # 詳細ドキュメント（日本語）
│
├── Receipt_classificaion_model-main/ # レシート傾き分類モデル
│   ├── data_load_train.py            # 訓練スクリプト
│   ├── prediction.py                 # 推論・補正スクリプト
│   └── README.md                     # 詳細ドキュメント（日本語）
│
├── Receipt_detection_model-main/     # レシート文字領域検出モデル
│   ├── detect.py                     # YOLOv9 推論スクリプト
│   ├── train.py                      # 訓練スクリプト
│   ├── best_text.pt                  # 学習済み重み
│   └── README.md                     # 詳細ドキュメント（日本語）
│
├── multi_receipts detection/         # マルチレシート角点検出システム
│   ├── train/generate_data.py        # 合成データ生成
│   ├── evaluate.py                   # 評価スクリプト
│   ├── tools/perspective_utils.py    # 透視補正ツール
│   └── README.md                     # 詳細ドキュメント（日本語）
│
├── back up/                          # バックアップデータ（VOC2007形式）
│
├── README.md                         # 本ファイル（日本語）
├── README.zh.md                      # 中文版
└── README.en.md                      # English Version
```

---

## 技術スタック

| カテゴリ | 技術 | バージョン | 用途 |
|----------|------|-----------|------|
| **物体検出** | Ultralytics YOLOv8m-Detect | >=8.0 | レシート枠 & 角点検出 |
| **物体検出** | YOLOv5 / YOLOv9 (GELAN-C) | - | 文字領域検出 |
| **画像分類** | ResNet34 | - | 傾き角度分類（4クラス） |
| **OCR** | PaddleOCR (PP-OCRv5) | 3.6.0 | テキスト認識 |
| **深層学習** | PaddlePaddle | 3.3.1 | OCR バックエンド |
| **深層学習** | PyTorch | >=2.0 | YOLO / ResNet34 バックエンド |
| **Web UI** | Streamlit | >=1.28 | インタラクティブUI |
| **画像処理** | OpenCV | >=4.5 | 透視変換・画像処理 |
| **画像処理** | Pillow | >=9.0 | 画像操作 |
| **データ処理** | NumPy | - | 行列演算 |
| **データ処理** | Pandas | >=1.5 | CSV 処理 |

---

## 環境依存関係

### システム要件

- **OS**: macOS / Linux / Windows
- **Python**: >= 3.8
- **CUDA**: オプション（GPU推論推奨）

### インストール手順

```bash
# 1. リポジトリをクローン
git clone <repository-url>
cd receipt-auto-digitizer

# 2. 各サブプロジェクトの依存関係をインストール
# サブプロジェクトごとに仮想環境を作成することを推奨

# --- OCR UI システム ---
cd Receipt_OCR_UI_System-main
pip install streamlit ultralytics paddleocr paddlepaddle opencv-python pillow pandas torch torchvision

# --- レシート傾き分類モデル ---
cd ../Receipt_classificaion_model-main
pip install torch torchvision pillow

# --- レシート文字領域検出モデル ---
cd ../Receipt_detection_model-main
pip install torch torchvision pyyaml

# --- マルチレシート角点検出 ---
cd "../multi_receipts detection"
pip install ultralytics pillow numpy pyyaml opencv-python
```

---

## デプロイ手順

### クイックスタート（OCR UI システム）

```bash
cd Receipt_OCR_UI_System-main
python -m streamlit run app.py --server.port 8503
```

ブラウザで `http://localhost:8503` にアクセスし、合成写真をアップロードしてパイプラインを実行します。

### 各サブプロジェクトの個別実行

各サブプロジェクトの README を参照してください。

---

## 開発規範

### コーディング規約

- **言語**: Python 3.8+
- **スタイル**: PEP 8 準拠
- **命名規則**: スネークケース（`snake_case`）を基本とする
- **型ヒント**: 可能な限り型アノテーションを付与する

### Git 運用

- **ブランチ戦略**: `main` （安定版）, `develop` （開発中）, `feature/*` （機能別）
- **コミットメッセージ**: 日本語または英語で、変更内容を簡潔に記述
- **プルリクエスト**: レビュー必須、1機能につき1PR

### ドキュメント

- 各サブプロジェクトに README.md（日本語）, README.zh.md（中文）, README.en.md（English）を配置
- コード内の主要関数には docstring を記述

---

## 貢献ガイド

1. このリポジトリをフォーク
2. 機能ブランチを作成（`git checkout -b feature/amazing-feature`）
3. 変更をコミット（`git commit -m 'feat: 素晴らしい機能を追加'`）
4. ブランチにプッシュ（`git push origin feature/amazing-feature`）
5. プルリクエストを作成

### 報告・提案

- バグ報告や機能提案は Issue にて受け付けています
- セキュリティに関する問題は、直接メンテナーにご連絡ください

---

## ライセンス

本プロジェクトは学習および研究目的での使用を想定しています。詳細は各サブプロジェクトのライセンスを参照してください。

---

## メンテナー

- **Author**: [baiwenbin](https://github.com/baiwenbin)
- **お問い合わせ**: GitHub Issues にて受け付けています

---

## 参考リンク

- [YOLOv9](https://github.com/WongKinYiu/yolov9)
- [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [Streamlit](https://streamlit.io/)
