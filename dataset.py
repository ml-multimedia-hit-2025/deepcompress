import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random # For random cropping

class ImageCompressionDataset(Dataset):
    """
    PyTorch Dataset for loading and preprocessing images for compression autoencoder training.

    Args:
        root_dir (str): Directory containing image files.
        train_crop_size (int, optional): Size for random crop during training. Default is 256.
        is_train (bool, optional): Whether the dataset is for training (enables random cropping). Default is True.
    """
    def __init__(self, root_dir, train_crop_size=256, is_train=True):
        super().__init__()
        self.root_dir = root_dir
        self.train_crop_size = train_crop_size
        self.is_train = is_train # To decide whether to crop or not
        self.image_paths = []

        if not os.path.isdir(root_dir):
            print(f"Warning: Root directory '{root_dir}' does not exist.")
        else:
            self.image_paths = [
                os.path.join(root_dir, f)
                for f in os.listdir(root_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
        
        if not self.image_paths:
            print(f"Warning: No images found in '{root_dir}'.")

    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Loads and preprocesses an image at the given index.

        Args:
            idx (int): Index of the image to load.

        Returns:
            tuple: (input_tensor, target_tensor) for autoencoder training.
        """
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            # Fallback for bad image: a tensor of zeros matching crop size or a default small size
            if self.is_train:
                dummy_size = (self.train_crop_size, self.train_crop_size)
            else:
                dummy_size = (64,64) # Default for eval/test
            img_array = np.zeros((dummy_size[1], dummy_size[0], 3), dtype=np.float32)
            tensor = torch.FloatTensor(img_array).permute(2, 0, 1)
            return tensor, tensor

        if self.is_train:
            # Random crop for training
            w, h = img.size
            if w < self.train_crop_size or h < self.train_crop_size:
                # Resize if image is smaller than crop size
                img = img.resize((self.train_crop_size, self.train_crop_size), Image.LANCZOS)
                x1, y1 = 0, 0 # Crop from top-left
            else:
                x1 = random.randint(0, w - self.train_crop_size)
                y1 = random.randint(0, h - self.train_crop_size)
            img_cropped = img.crop((x1, y1, x1 + self.train_crop_size, y1 + self.train_crop_size))
            img_to_process = img_cropped
        else:
            # For validation/testing, use the whole image
            img_to_process = img

        img_array = np.array(img_to_process, dtype=np.float32) / 255.0
        tensor = torch.FloatTensor(img_array).permute(2, 0, 1)

        return tensor, tensor