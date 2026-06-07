# 領収書OCR処理システム

> **言語**: [English](README.en.md) | [中文](README.zh.md)

YOLOv8、YOLOv5、ResNet34、PaddleOCR によるマルチレシート検出・分類・文字領域検出・OCR認識パイプライン。
Streamlit による直感的なWeb UI を提供します。

---

## システムアーキテクチャ

**4枚の領収書が並んだ合成写真**を入力とし、分割・傾き補正・文字領域検出・OCR認識を経て、構造化されたCSVを出力します。

```
                    ┌─────────────────────────────────────┐
  合成写真           │  YOLOv8m-Detect (best_v8_300.pt)   │
  (4枚の領収書)   ──▶  分割 & 台形補正                      │
                    └──────────────┬──────────────────────┘
                                   │  4枚の個別領収書
                    ┌──────────────▼──────────────────────┐
                    │  ResNet34 分類                       │
                    │  傾き補正 (0°/90°/180°/270°)          │
                    └──────────────┬──────────────────────┘
                                   │  4枚の正立領収書
                    ┌──────────────▼──────────────────────┐
                    │  YOLOv5 検出 (runs/train/exp)        │
                    │  4クラス: 日付/店名/電話番号/合計金額   │
                    └──────────────┬──────────────────────┘
                                   │  バウンディングボックス
                    ┌──────────────▼──────────────────────┐
                    │  PaddleOCR (PP-OCRv5, lang=japan)   │
                    │  クラス別テキスト認識 + 後処理          │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                          CSV出力 (.csv)
```

## プロジェクト構成

```
Receipt_OCR_UI_System-main/
├── app.py                     # Streamlit メインエントリ (Full Pipeline + 各ステップ)
├── OCR.py                     # PaddleOCR モジュール (グローバルシングルトン, 4クラス後処理)
├── Classification.py          # ResNet34 傾き分類器 (0°/90°/180°/270°)
├── detect.py                  # YOLOv5 推論スクリプト (文字領域検出)
├── perspective_utils.py       # 台形補正共通モジュール (コーナー法 + OpenCVフォールバック)
├── best_v8_300.pt             # YOLOv8m-Detect: 領収書 + コーナー検出モデル
│
├── logs/
│   └── Epoch82-Total_Loss0.0004.pth   # ResNet34 分類重み
│
├── runs/
│   ├── train/exp/weights/best.pt      # YOLOv5 文字検出重み
│   └── detect/exp*/                    # 検出出力 (自動インクリメント)
│       ├── corrected_image_*.jpg       #   検出結果アノテーション画像
│       └── labels/
│           └── corrected_image_*.txt   #   YOLO形式ラベル
│
├── data/
│   ├── corrected_images/               # Step 2 出力: 傾き補正済み領収書
│   ├── test/images/                    # 旧テスト画像ディレクトリ
│   ├── pipeline_temp/classified/       # パイプライン一時分類結果
│   ├── prediction/                     # 分類入力ディレクトリ
│   └── ocr_upload/                     # 過去アップロード
│
└── ocr_results*.csv                    # 最終OCR出力 (自動連番)
```

## モデル一覧

### 1. 領収書検出 & 分割

| 項目 | 値 |
|----------|-------|
| モデル | YOLOv8m-Detect |
| 重み | `best_v8_300.pt` |
| クラス数 | 5: `receipt(0)`, `corner_tl(1)`, `corner_tr(2)`, `corner_br(3)`, `corner_bl(4)` |
| 学習データ | 合成写真300枚, 1枚あたり領収書4枚 |
| 信頼度閾値 | 0.25 |
| NMS IoU | 0.45 |

**台形補正** (`perspective_utils.py`):

| 方式 | トリガー | 説明 |
|--------|---------|-------------|
| **コーナー法** | 4点検出 + バリデーション通過 | クラスID (TL/TR/BR/BL) で直接マッピング、幾何ソート不要 |
| **OpenCVフォールバック** | コーナー不足 または バリデーション失敗 | CLAHE + 適応二値化 + 輪郭四角形検出 |

**バリデーション**: 境界チェック、自己交差検出、面積 > 100px、アスペクト比 < 8:1。

### 2. 傾き分類

| 項目 | 値 |
|----------|-------|
| モデル | ResNet34 |
| 重み | `logs/Epoch82-Total_Loss0.0004.pth` |
| 入力 | 224×224, [-1, 1] 正規化 |
| クラス | 0=0°正立, 1=90°反時計回り, 2=180°, 3=270°反時計回り |

### 3. 文字領域検出

| 項目 | 値 |
|----------|-------|
| フレームワーク | YOLOv5 |
| 重み | `runs/train/exp/weights/best.pt` |
| 入力 | 640×640 |
| 信頼度 | 0.5 |

| クラスID | 項目 | CSV列名 | PaddleOCR 後処理 |
|:--------:|-------|:----------:|---------------------------|
| 0 | 日付 | `date` | トリムのみ (PP-OCRv5 が日付形式をネイティブ処理) |
| 1 | 店名 | `store` | かな/漢字 + 英数字を保持 |
| 2 | 電話番号 | `phone` | 数字 + `- ( ) . :` + TEL を保持 |
| 3 | 合計金額 | `total` | 右50%クロップ (¥付き数値), 数字 + ¥$ を保持 |

### 4. OCR認識

| 項目 | 値 |
|----------|-------|
| エンジン | PaddleOCR 3.6.0 |
| モデル | PP-OCRv5_server_det + PP-OCRv5_server_rec + PP-LCNet |
| 言語 | `japan` (日本語 + 英語) |
| インスタンス | グローバルシングルトン (`get_ocr()`), モジュールロード時1回初期化 |
| キャッシュ | `~/.paddlex/` に自動ダウンロード & キャッシュ |
| 出力 | UTF-8 BOM CSV, 自動連番 (`ocr_results1.csv`, `ocr_results2.csv`, …) |

## CSV出力形式

| 列 | 型 | 例 |
|--------|------|---------|
| `image_name` | string | `corrected_image_1.jpg` |
| `date` | string | `2026年03月21日` |
| `store` | string | `Beisie 吉井店` |
| `phone` | string | `027-387-6511` |
| `total` | string | `￥1,529` |

## 使用方法

### クイックスタート

```bash
# 依存パッケージのインストール
pip install streamlit ultralytics paddleocr paddlepaddle pillow opencv-python pandas

# サーバー起動
cd Receipt_OCR_UI_System-main
python -m streamlit run app.py --server.port 8503
```

### サイドバーモード

| モード | 説明 |
|------|-------------|
| **Full Pipeline** | 合成写真アップロード → 分割 → 分類 → 検出 → OCR → CSV (ワンクリック) |
| **Multi Receipts Detection** | 合成写真分割 → クロップ → 台形補正 |
| **Classification** | 単一画像傾き補正 (0°/90°/180°/270°) |
| **Detection** | 補正済み画像に対しYOLOv5文字領域検出を実行 |
| **OCR** | 既存の検出結果に対しPaddleOCR → CSV |

### Full Pipeline 表示レイアウト

各処理ステップは **4列** で表示（領収書1枚ずつ）、ステップ間で縦方向に整列:

```
┌──────────┬──────────┬──────────┬──────────┐
│   Step 0: 元画像 (合成写真)                │
├──────────┼──────────┼──────────┼──────────┤
│   Step 1: クロップ (4枚の個別領収書)        │
├──────────┼──────────┼──────────┼──────────┤
│   Step 1: 台形補正 [方式]                  │
├──────────┼──────────┼──────────┼──────────┤
│   Step 2: 傾き補正                          │
├──────────┼──────────┼──────────┼──────────┤
│   Step 3: 文字領域検出                      │
├──────────┼──────────┼──────────┼──────────┤
│   Step 4: OCR結果 CSV (ダウンロード可能)     │
└──────────┴──────────┴──────────┴──────────┘
```

各ファイル間は **赤い水平線** で区切られます。

## ファイル保存規則

| カテゴリ | パス | 命名規則 | 保持期間 |
|----------|------|--------|-----------|
| 検出モデル | `./best_v8_300.pt` | 固定 | 永続 |
| 台形補正モジュール | `./perspective_utils.py` | 固定 | 永続 |
| 分類重み | `./logs/Epoch82-*.pth` | 固定 | 永続 |
| 検出重み | `./runs/train/exp/weights/best.pt` | 固定 | 永続 |
| 補正済み画像 | `./data/corrected_images/` | `corrected_image_{i}.jpg` | 実行毎に上書き |
| 検出出力 | `./runs/detect/exp{n}/` | 自動インクリメント | 永続 |
| 検出ラベル | `./runs/detect/exp{n}/labels/` | `corrected_image_{i}.txt` | 永続 |
| OCR結果 | `./` | `ocr_results{n}.csv` | 自動連番, 永続 |
| PaddleOCRモデル | `~/.paddlex/official_models/` | 自動ダウンロード | 無期限キャッシュ |
| アップロード元画像 | メモリのみ | — | セッション終了まで |

## 主要設計判断

1. **OCRグローバルシングルトン** — `PaddleOCR` は `get_ocr()` で1回のみ初期化。画像毎の再ロードを回避。

2. **クラスIDベースの台形補正** — コーナークラスID (1=TL, 2=TR, 3=BR, 4=BL) を直接用いて四角形を構築。`order_corners` の幾何ソートが回転・傾斜で失敗する問題を回避。

3. **自動OpenCVフォールバック** — 4点未満またはバリデーション失敗時、OpenCV輪郭ベースの台形補正に透過的にフォールバック。

4. **アスペクト比フィルタリング** — アスペクト比が `[0.20, 0.95]` 外の領収書枠を破棄。画像端の細長い誤検出を除去。

5. **X範囲制約付きコーナーグルーピング** — `group_corners_to_receipt()` はコーナーx座標が領収書枠の `[x1 - 0.4*w, x2 + 0.4*w]` 範囲内にある場合のみマッチング。領収書間のコーナー誤割当を防止。

6. **クラス別OCR後処理** — 4つの文字領域クラスそれぞれに専用テキストクレンジング（日付:最小限、店名:日本語文字、電話:数字+区切り、合計:右半分クロップで数値抽出）。

## 依存パッケージ

| パッケージ | バージョン | 用途 |
|---------|---------|---------|
| streamlit | >=1.28 | Web UI |
| ultralytics | >=8.0 | YOLOv8 推論 |
| paddleocr | 3.6.0 | OCR 認識 |
| paddlepaddle | 3.3.1 | PaddlePaddle バックエンド |
| opencv-python | >=4.5 | 画像処理 & 台形補正 |
| pillow | >=9.0 | PIL 画像操作 |
| pandas | >=1.5 | CSV 処理 & DataFrame 表示 |
| torch | >=2.0 | ResNet34 分類 |
| torchvision | >=0.15 | 分類変換 |

## 参考

- [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics)
- [PaddleOCR / PP-OCRv5](https://github.com/PaddlePaddle/PaddleOCR)
- [Streamlit](https://streamlit.io/)
