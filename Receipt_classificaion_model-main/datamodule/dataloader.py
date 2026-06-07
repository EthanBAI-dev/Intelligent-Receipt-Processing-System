import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class RotatedReceiptDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the raw images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # self.image_filenames = os.listdir(self.root_dir)
        
        # Filter out non-image files using _is_image_file method
        self.image_filenames = [f for f in os.listdir(self.root_dir) if self._is_image_file(f)]


    def __len__(self):
        # Each image has 4 versions: original, 90°, 180°, and 270° rotations
        return len(self.image_filenames) * 4

    def __getitem__(self, idx):
        # Determine the base image and its rotation type
        image_idx = idx // 4  # Determine which image
        rotation_type = idx % 4  # Determine which rotation
        
        # Load the image
        img_name = os.path.join(self.root_dir, self.image_filenames[image_idx])
        image = Image.open(img_name)

        # Apply the rotation based on the rotation_type
        if rotation_type == 1:
            image = image.rotate(90, expand=True)
        elif rotation_type == 2:
            image = image.rotate(180, expand=True)
        elif rotation_type == 3:
            image = image.rotate(270, expand=True)

        # Apply any transformations (like resizing or normalization)
        if self.transform:
            image = self.transform(image)

        return image, rotation_type  # Return the image and its rotation type
    
    def _is_image_file(self, filename):
        # Check if the file is an image by its extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        return any(filename.lower().endswith(ext) for ext in valid_extensions)

# Add the get_data_transforms() function here
def get_data_transforms(image_size=224):
    """
    Returns the transformations for training and validation data.
    
    Args:
        image_size (int): The size to which the images will be resized. Default is 224.
    
    Returns:
        dict: A dictionary with 'train' and 'val' transformations.
    """
    return {
        'train': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'val': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
         'pre': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        
    }