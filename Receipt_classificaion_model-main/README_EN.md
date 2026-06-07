**日本語** | [中文](README_ZH.md) | [English](README_EN.md)

# **Receipt Image Rotation Correction Model (Classification Model)**

## **Overview**
This project uses a **ResNet34**-based classification model to determine the rotation angle (0°, 90°, 180°, 270°) of receipt images and automatically correct them to 0°.

Below is an example of a 270° rotated image being detected and corrected to 0° by the model.

<div align="medium">
  <img src="figure/prediction2.png", width="100%"> 
</div>

## **Project Structure**

```
├── data/
│   ├── train/images/      # Training images
│   ├── test/images/       # Test images
│   └── valid/images/      # Validation images
├── datamodule/
│   └── dataloader.py      # Dataset & preprocessing
├── nets/
│   ├── blocks.py          # Convolution block definitions
│   ├── resnet.py          # ResNet34 model definition
│   └── README.md
├── plotmodule/
│   └── plot_utils.py      # Visualization utilities
├── debug_images/
│   └── debug_utils.py     # Debug image saving
├── figure/                # README images
├── logs/                  # Trained models (.pth)
├── data_load_train.py     # Training script
├── prediction.py          # Inference & correction script
├── README.md              # Japanese version
├── README_EN.md           # This file (English)
└── README_ZH.md           # Chinese version
```

## **Dataset Preparation**

Place training images in `data/train/images/` and test images in `data/test/images/`. All images should be in their upright orientation (0°). The code will automatically generate 90°, 180°, and 270° rotated versions, creating a 4-class dataset.

- **Class 0**: 0° (upright)
- **Class 1**: 90° rotated left
- **Class 2**: 180° rotated
- **Class 3**: 270° rotated left (= 90° rotated right)

<img width="800" height="500" src=figure/3.png/> 

## **Pipeline**

### 1. Data Preprocessing — [dataloader.py](datamodule/dataloader.py)

The `RotatedReceiptDataset` class loads images and generates 4 rotated versions from each. Preprocessing includes:

- **Resize**: All images resized to 224×224 pixels
- **Normalization**: Each RGB channel normalized to `[0.5, 0.5, 0.5]`

```python
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
```

### 2. Training & Validation — [data_load_train.py](data_load_train.py)

| Setting | Value |
|---------|-------|
| Model | ResNet34 |
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam |
| Learning Rate | 0.001 (initial) |
| Scheduler | StepLR (gamma=0.92) |
| Batch Size | 8 |
| Epochs | 10 (configurable) |

After training, models are saved to the `logs/` directory.

<div align="medium">
  <img src="figure/trainresult.png", width="100%"> 
</div>

### 3. Inference & Angle Correction — [prediction.py](prediction.py)

Uses a trained model to predict the rotation angle and correct the image to 0°.

```python
# Rotate image based on predicted label
def correct_image_orientation(image, predicted_label):
    if predicted_label == 1:
        return image.rotate(270, expand=True)  # 90° → 0°
    elif predicted_label == 2:
        return image.rotate(180, expand=True)  # 180° → 0°
    elif predicted_label == 3:
        return image.rotate(90, expand=True)   # 270° → 0°
    return image  # already 0°
```

## **Usage**

```bash
# Train
python3 data_load_train.py

# Inference
python3 prediction.py
```

## **Trained Model**

Pre-trained model: `logs/Epoch82-Total_Loss0.0004.pth` (ResNet34, 4-class classification, loss 0.0004)

## **Reference**

- [ResNet50-MNIST-pytorch](https://github.com/wangyunjeff/ResNet50-MNIST-pytorch/tree/master)
