import os
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from nets.resnet import resnet34
from PIL import Image

# Set model path and batch size
MODEL_PATH = './logs/Epoch100-Total_Loss0.0031.pth'  # Modify to your trained model path
Batch_Size = 1  # Number of images to predict at a time

# Define data transformations (consistent with training)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize
])

# Define the prediction folder
prediction_folder = './data/prediction'

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

import os

def predict_and_correct_images(images, model, transform, save_folder="./data/corrected_images"):
    """
    Predict and correct orientation for a list of images and save them.
    :param images: List of PIL image objects to process.
    :param model: Trained PyTorch model.
    :param transform: Transformation pipeline for the input images.
    :param save_folder: Folder to save corrected images.
    :return: Tuple of (corrected_images, predicted_labels)
    """
    corrected_images = []
    predicted_labels = []

    # Ensure the save folder exists
    os.makedirs(save_folder, exist_ok=True)

    for idx, image in enumerate(images):
        original_image = image.copy()  # Save the original image for correction

        # Apply data transformations
        transformed_image = transform(image).unsqueeze(0).to(device)

        # Perform prediction
        with torch.no_grad():
            outputs = model(transformed_image)

            # If outputs is a tuple, take the first element
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted_label = torch.max(probabilities, 1)

        # Print prediction results
        print(f"Predicted Label: {predicted_label.item()}")
        print(f"Probabilities: {probabilities.cpu().numpy()}")

        # Rotate the image if the predicted label is not 0
        corrected_image = correct_image_orientation(original_image, predicted_label.item())
        corrected_images.append(corrected_image)
        predicted_labels.append(predicted_label.item())

        # Save the corrected image
        corrected_image_path = os.path.join(save_folder, f"corrected_image_{idx + 1}.jpg")
        corrected_image.save(corrected_image_path)
        print(f"Saved corrected image to {corrected_image_path}")

    return corrected_images, predicted_labels

# # Function to predict and correct image orientation, and save corrected images
# def predict_and_correct_images(image_files, model, transform, save_folder="./data/corrected_images/"):
#     # Create the folder to save corrected images if it does not exist
#     os.makedirs(save_folder, exist_ok=True)

#     for file_name in image_files:
#         file_path = os.path.join(prediction_folder, file_name)

#         # Read the image and convert to PIL format
#         image = Image.open(file_path).convert("RGB")
#         original_image = image.copy()  # Save the original image for display

#         # Apply data transformations
#         transformed_image = transform(image).unsqueeze(0).to(device)

#         # Perform prediction
#         with torch.no_grad():
#             outputs = model(transformed_image)

#             # If outputs is a tuple, take the first element
#             if isinstance(outputs, tuple):
#                 outputs = outputs[0]

#             probabilities = torch.nn.functional.softmax(outputs, dim=1)
#             _, predicted = torch.max(probabilities, 1)

#         # Print prediction results
#         print(f"File: {file_name}, Predicted Label: {predicted.item()}")
#         print(f"Probabilities: {probabilities.cpu().numpy()}")

#         # Rotate the image if the predicted label is not 0
#         corrected_image = correct_image_orientation(original_image, predicted.item())

#         # Save the corrected image
#         corrected_image_path = os.path.join(save_folder, file_name)
#         corrected_image.save(corrected_image_path)
#         print(f"Saved corrected image to {corrected_image_path}")

#         # Optionally, display the original and corrected images
#         plot_original_and_corrected_image(original_image, corrected_image, file_name, predicted.item())
        


# # Visualization function: Display original and corrected images
# def plot_original_and_corrected_image(original_image, corrected_image, file_name, predicted_label):
#     fig, axs = plt.subplots(1, 2, figsize=(12, 6))

#     # Display the original image
#     axs[0].imshow(original_image)
#     axs[0].axis('off')
#     axs[0].set_title(f"Original: {file_name}\nPred: {predicted_label}")

#     # Display the corrected image
#     axs[1].imshow(corrected_image)
#     axs[1].axis('off')
#     axs[1].set_title("Corrected to 0°")

#     plt.tight_layout()
#     plt.show()

# # Execute the prediction and correction
# predict_and_correct_images(image_files, model, data_transforms)

# def predict_and_correct_images(images, model, transform):
#     """
#     Predict and correct orientation for a list of images.
#     :param images: List of PIL image objects to process.
#     :param model: Trained PyTorch model.
#     :param transform: Transformation pipeline for the input images.
#     :return: Tuple of (corrected_images, predicted_labels)
#     """
#     corrected_images = []
#     predicted_labels = []

#     for image in images:
#         original_image = image.copy()  # Save the original image for correction

#         # Apply data transformations
#         transformed_image = transform(image).unsqueeze(0).to(device)

#         # Perform prediction
#         with torch.no_grad():
#             outputs = model(transformed_image)

#             # If outputs is a tuple, take the first element
#             if isinstance(outputs, tuple):
#                 outputs = outputs[0]

#             probabilities = torch.nn.functional.softmax(outputs, dim=1)
#             _, predicted_label = torch.max(probabilities, 1)

#         # Print prediction results
#         print(f"Predicted Label: {predicted_label.item()}")
#         print(f"Probabilities: {probabilities.cpu().numpy()}")

#         # Rotate the image if the predicted label is not 0
#         corrected_image = correct_image_orientation(original_image, predicted_label.item())
#         corrected_images.append(corrected_image)
#         predicted_labels.append(predicted_label.item())

#     return corrected_images, predicted_labels