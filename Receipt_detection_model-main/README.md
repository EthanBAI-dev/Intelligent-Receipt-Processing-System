# YOLOv9で構築した領収書の検出モデル

## Summary

<div style="max-width: 600px; word-wrap: break-word;">
本記事ではYOLO v9モデルを利用して、自分でスーパー買い物の領収書を収集して物体の検出モデルの構築方法について記録します。その中に、125枚の領収書を収集し、3つのラベル（クラス）（『合計』、『日付』、『電話番号』）を付けてデータをアノテーションし、領収書の検出モデルを学習しました。

結果として、指定した3つのクラスを正確に検出することができました。
</div>

 <div align="medium">
    <img src="images/testresult.png" alt="YOLO" width="100%">
</div>

このプロジェクトは、[YOLOv9](https://github.com/WongKinYiu/yolov9) を参考にして作成されています。

## はじめに

### YOLO v9について
<div style="max-width: 600px; word-wrap: break-word;">
YOLOは「You Only Look Once」の略で、速度と精度の高さで知られる最先端のオブジェクト検出モデルです。このモデルの第9バージョン（v9）は、「Programmable Gradient Information（PGI）」や「Generalized Efficient Layer Aggregation Network（GELAN）」といった新しいアーキテクチャを導入し、機能と性能がさらに強化されています。
</div>


## 準備

<div style="max-width: 600px; word-wrap: break-word;">

### 学習はGoogle Colabを使用します。

Google Colabは、GPUへの無料アクセスを提供するクラウドベースのプラットフォームであり、この記事は、Google ColabのGPUを使って、ディープラーニングモデルの学習を行なっていました。

### データ準備

#### 1. 領収書データを収集

自分で買い物の領収書が200枚以上の領収書を収集して、その中は、比較的綺麗なと折り目ないの125枚を選び、使います。

#### 2. Roboflowでアノテーションに

アノテーションツールにはいくつかの選択肢がありますが、最初にLabelImgとLabelmeを試しました。しかし、比較した結果、ウェブベースのRoboflowが非常に使いやすいことが分かりました。今回はRoboflowを使用してアノテーションを行いました。

Roboflowについて
[Roboflow](https://app.roboflow.com/)はデータセットのアノテーションを簡単かつ迅速に行い、さまざまなモデルの学習に適した形式に変換できるソリューションを提供しています。

#### 3. データアップロード
ここでは、125枚の領収書写真のデータをRoboflowにアップロードしました。


<div align="medium">
    <img src="images/dataset1.png" alt="YOLO" width="100%">
</div>


#### 4. アノテーションに行きます
このプロジェクトでは、領収書から以下の3つのラベルを作ります：『合計』、『日付』、『電話番号』。そのため、それぞれのラベル名を入力し、各領収書サンプルのアノテーションを作ります。


<div align="medium">
    <img src="images/dataset2.png" alt="YOLO" width="100%">
</div>


#### 5. 前処理

アノテーションが完了したら、いくつかの前処理を行います。例えば、データセットを学習用、推論用、テスト用に分割りします。また、画像を640x640サイズにリサイズします（現在のYOLOv9は640x640サイズの画像しか処理できないため、このステップは必須です）。

これらの前処理を行った結果、データセットの87枚が学習用、26枚を推論用、12枚はテスト用として準備されました。

<div align="medium">
    <img src="images/dataset3.png" alt="YOLO" width="100%">

</div>


データセットの作成が完了したら、Roboflowの「Export Dataset」オプションを使用して、YOLO v9モデル用のデータセットをエクスポートできます。エクスポートしたデータセットコードを取得し、このコードをColab上で使用してモデルの学習を行います。


## 学習の流れ

それでは、Colab上でYOLO v9モデルを使用してデータセットをどのように学習するかを見ていきましょう。
まず、Colabの「ランタイムのタイプを変更」オプションからハードウェアアクセラレータをT4 GPUに変更します。その後、YOLOv9のリポジトリをGithubからGoogleドライブにクローンする必要があります。そのために、Googleドライブをマウントしてリポジトリをクローンし、以下のコードを使用して必要なファイルやパッケージをすべてインストールします。

###  Google Driveを接続に

```python
from google.colab import drive
drive.mount('/content/drive')
```

### YOLOv9のリポジトリをクローンと必要なパッケージをインストール

Clone YOLO v9 repository into your Google Drive.

```python
!git clone https://github.com/WongKinYiu/yolov9
%cd yolov9
!pip install -r requirements.txt -q
```

## モデルをダウンロード

リポジトリをクローンして必要なファイルをインストールした後、ディレクトリを作成し、以下のコードを実行してすべての重み（事前学習したモデル）をダウンロードして保存することができます。

```python
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-e.pt
```

## Roboflowからデータセットをインポート
次に、Roboflowで作成したデータセットをインポートします，その前に、Roboflowパッケージをインストールする必要があります。

```python
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="xxxxxxx")
project = rf.workspace().project("projectname")
version = project.version(version number)
dataset = version.download("yolov9")
```

## 学習

その後、データセットを使ってYOLOモデルを学習するための学習コードを実行します。ここでは、gelan-c weight.ptを使用してデータセットを学習します。学習を始める前に、モデルの設定ファイル（ここでは gelan-c.yaml）内のアンカー数を、データセットでアノテーションしたラベルの数に変更してください（この記事は３と使います）。モデルをより正確にするために、100エポックで学習を行いました。ただし、必要に応じてエポック数を変更することも可能です。

```python
%cd {HOME}/yolov9

!python train.py \
--batch 8 --epochs 100 --img 640 --device 0 --min-items 0 --close-mosaic 15 \
--data invoice_extraction-3/data.yaml \
--weights {HOME}/weights/gelan-c.pt \
--cfg models/detect/gelan-c.yaml \
--hyp hyp.scratch-high.yaml
```

## 学習の結果

これでモデルの学習が完了しました。モデルを100エポックで学習した結果、全クラスにおいて平均適合率（mean average precision）が0.96という良好な精度を達成しました。

 <div align="medium">
    <img src="images/results1.png" alt="YOLO" width="100%">
</div>
Accuracy and Precision of the trained YOLO-v9 model

 <div align="medium">
    <img src="images/results2.png" alt="YOLO" width="100%">
</div>
Graph showing the model performance


## テストイメージを使って検証
   
```python
!python detect.py \
--img 640 --conf 0.5 --device 0 \
--weights {HOME}/yolov9/runs/train/exp2/weights/best.pt \
--source {HOME}/yolov9/test/images

import glob
from IPython.display import Image, display

for image_path in glob.glob(f'{HOME}/yolov9/runs/detect/exp2/*.jpg')[:3]:
      display(Image(filename=image_path, width=600))
      print("\n")
```
 <div align="medium">
    <img src="images/testresult.png" alt="YOLO" width="100%">
</div>

## Reference

<details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)
* [https://github.com/VDIGPKU/DynamicDet](https://github.com/VDIGPKU/DynamicDet)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
