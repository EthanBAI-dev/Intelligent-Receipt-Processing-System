# plot_utils.py
import matplotlib.pyplot as plt
import random
import numpy as np 



# Function to display a tensor as an image
def imshow(image, ax, title=None):
    # Convert the tensor to a NumPy array and unnormalize it
    image = image.numpy().transpose((1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    image = std * image + mean  # Unnormalize the image
    image = np.clip(image, 0, 1)  # Clip values to make sure they are between [0, 1]

    # Plot the image
    ax.imshow(image)
    if title:
        ax.set_title(title)
    ax.axis('off')  # Hide axes for better visualization

# Function to randomly plot 5 images from each class
def plot_samples_from_each_class(dataset, num_samples=5):
    # Dictionary to store image indices for each class
    class_to_indices = {i: [] for i in range(4)}  # Assuming you have 4 classes (0, 1, 2, 3)

    # Group indices by class label
    for i in range(len(dataset)):
        _, label = dataset[i]
        class_to_indices[label].append(i)  # Store the index, not the image, for sampling

    # Set up a figure with subplots (4 rows for classes, num_samples columns)
    fig, axs = plt.subplots(4, num_samples, figsize=(15, 12))

    # Loop through each class and randomly sample images
    for class_idx in range(4):
        selected_indices = random.sample(class_to_indices[class_idx], num_samples)  # Randomly select indices
        for i, idx in enumerate(selected_indices):
            image, label = dataset[idx]  # Get the image and label using the index
            ax = axs[class_idx, i]  # Get the subplot for this image
            imshow(image, ax, title=f"Class {label} Image {i+1}")

    plt.tight_layout()
    plt.show()
    
    
# code with axis 
# # Function to display a tensor as an image
# def imshow(image, ax, title=None):
#     # Convert the tensor to a NumPy array and unnormalize it
#     image = image.numpy().transpose((1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
#     mean = [0.5, 0.5, 0.5]
#     std = [0.5, 0.5, 0.5]
#     image = std * image + mean  # Unnormalize the image
#     image = np.clip(image, 0, 1)  # Clip values to make sure they are between [0, 1]

#     # Plot the image
#     ax.imshow(image)
#     if title:
#         ax.set_title(title)
#     ax.set_xticks([])  # Hide x ticks
#     ax.set_yticks([])  # Hide y ticks

# # Function to randomly plot 5 images from each class, adding axis labels
# def plot_samples_from_each_class(dataset, num_samples=5):
#     # Dictionary to store image indices for each class
#     class_to_indices = {i: [] for i in range(4)}  # Assuming you have 4 classes (0, 1, 2, 3)

#     # Group indices by class label
#     for i in range(len(dataset)):
#         _, label = dataset[i]
#         class_to_indices[label].append(i)  # Store the index, not the image, for sampling

#     # Set up a figure with subplots (4 rows for classes, num_samples columns)
#     fig, axs = plt.subplots(4, num_samples, figsize=(15, 12))

#     # Loop through each class and randomly sample images
#     for class_idx in range(4):
#         selected_indices = random.sample(class_to_indices[class_idx], num_samples)  # Randomly select indices
#         for i, idx in enumerate(selected_indices):
#             image, label = dataset[idx]  # Get the image and label using the index
#             ax = axs[class_idx, i]  # Get the subplot for this image
#             imshow(image, ax, title=f"Class {label} Image {i+1}")
    
#     # Add x-axis label for image position (1 to num_samples)
#     for i in range(num_samples):
#         axs[3, i].set_xlabel(f"Image {i+1}", fontsize=12)  # Set xlabel on the last row
    
#     # Add y-axis label for class (0 to 3)
#     for class_idx in range(4):
#         axs[class_idx, 0].set_ylabel(f"Class {class_idx}", fontsize=12, rotation=90, labelpad=20)  # Set ylabel for each row

#     plt.tight_layout()
#     plt.show()
