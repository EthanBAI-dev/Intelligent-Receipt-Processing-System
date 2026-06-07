import os
import cv2
import torch

def save_first_batch_images(epoch, iteration, images, targets, save_dir="debug_images/"):
    """
    Save images with corresponding labels during the first batch of training for debugging purposes.
    
    Args:
        epoch (int): Current epoch number.
        iteration (int): Current batch iteration number.
        images (torch.Tensor): Batch of images.
        targets (torch.Tensor): Corresponding labels for the images.
        save_dir (str): Directory to save the images.
    """
    if iteration == 0:  # Only save images for the first batch
        # Create a directory for the current epoch
        epoch_save_dir = os.path.join(save_dir, f"epoch_{epoch + 1}/")
        os.makedirs(epoch_save_dir, exist_ok=True)

        for i in range(images.size(0)):  # Loop through the batch
            img_tensor = images[i].cpu()  # Move image to CPU for saving
            target = targets[i].item()    # Get the target label
            
            # Convert tensor to NumPy array for saving with OpenCV
            img_np = img_tensor.permute(1, 2, 0).numpy() * 255  # Convert from (C, H, W) to (H, W, C)
            img_np = img_np.astype("uint8")

            # Save the image with the target label as the file name
            img_file_name = os.path.join(epoch_save_dir, f"image_{iteration}_{i}_label_{target}.png")
            cv2.imwrite(img_file_name, img_np)