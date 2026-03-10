import os
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from datamodule.dataloader import RotatedReceiptDataset
from nets.resnet import resnet34
from PIL import Image

# Set model path and batch size
MODEL_PATH = './logs/Epoch10-Total_Loss0.0067.pth'  # Modify to your trained model path
Batch_Size = 1  # Number of images to predict at a time

# Define data transformations (consistent with training)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize
])

# Define the prediction folder
prediction_folder = './data/Receipt_data/prediction'

# Load the model
model = resnet34(num_classes=4)  # Modify according to the number of classes
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# Get all image files from the folder
image_files = [f for f in os.listdir(prediction_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

# Function to validate and correct image orientation
def correct_image_orientation(image, predicted_label):
    """
    Rotate the image to 0 degrees based on the predicted label.
    """
    if predicted_label == 1:
        return image.rotate(270, expand=True)  # 90° counterclockwise, becomes 0°
    elif predicted_label == 2:
        return image.rotate(180, expand=True)  # 180° counterclockwise, becomes 0°
    elif predicted_label == 3:
        return image.rotate(90, expand=True)  # 270° counterclockwise, becomes 0°
    return image  # If already 0°, no rotation

# Function to predict and correct image orientation
def predict_and_correct_images(image_files, model, transform):
    for file_name in image_files:
        file_path = os.path.join(prediction_folder, file_name)

        # Read the image and convert to PIL format
        image = Image.open(file_path).convert("RGB")
        original_image = image.copy()  # Save the original image for display

        # Apply data transformations
        transformed_image = transform(image).unsqueeze(0).to(device)

        # Perform prediction
        with torch.no_grad():
            outputs = model(transformed_image)

            # If outputs is a tuple, take the first element
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities, 1)

        # Print prediction results
        print(f"File: {file_name}, Predicted Label: {predicted.item()}")
        print(f"Probabilities: {probabilities.cpu().numpy()}")

        # Rotate the image if the predicted label is not 0
        corrected_image = correct_image_orientation(original_image, predicted.item())

        # Display the original and corrected images
        plot_original_and_corrected_image(original_image, corrected_image, file_name, predicted.item())

# Visualization function: Display original and corrected images
def plot_original_and_corrected_image(original_image, corrected_image, file_name, predicted_label):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Display the original image
    axs[0].imshow(original_image)
    axs[0].axis('off')
    axs[0].set_title(f"Original: {file_name}\nPred: {predicted_label}")

    # Display the corrected image
    axs[1].imshow(corrected_image)
    axs[1].axis('off')
    axs[1].set_title("Corrected to 0°")

    plt.tight_layout()
    plt.show()

# Execute the prediction and correction
predict_and_correct_images(image_files, model, data_transforms)