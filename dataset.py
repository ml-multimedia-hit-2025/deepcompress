import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random # For random cropping

class ImageCompressionDataset(Dataset):
    def __init__(self, root_dir, train_crop_size=256, is_train=True): # Added train_crop_size and is_train
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
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            # Fallback for bad image: a tensor of zeros matching crop size or a default small size
            # This needs careful handling if such errors are common.
            # For simplicity, let's assume images are mostly fine or create a dummy.
            if self.is_train:
                dummy_size = (self.train_crop_size, self.train_crop_size)
            else: # For eval/test, we might not have a crop_size defined, use a small default
                dummy_size = (64,64) # Or handle appropriately
            img_array = np.zeros((dummy_size[1], dummy_size[0], 3), dtype=np.float32)
            tensor = torch.FloatTensor(img_array).permute(2, 0, 1)
            return tensor, tensor

        if self.is_train:
            # Random crop for training
            w, h = img.size
            if w < self.train_crop_size or h < self.train_crop_size:
                # If image is smaller than crop size, resize it up, then crop
                # Or, just resize to crop_size directly if always smaller
                # For simplicity, let's resize to crop_size if too small
                img = img.resize((self.train_crop_size, self.train_crop_size), Image.LANCZOS)
                x1, y1 = 0, 0 # Crop from top-left
            else:
                x1 = random.randint(0, w - self.train_crop_size)
                y1 = random.randint(0, h - self.train_crop_size)
            
            img_cropped = img.crop((x1, y1, x1 + self.train_crop_size, y1 + self.train_crop_size))
            img_to_process = img_cropped
        else:
            # For validation/testing, use the whole image (or center crop if preferred)
            # The FCN model can handle variable sizes at inference.
            img_to_process = img

        img_array = np.array(img_to_process, dtype=np.float32) / 255.0
        tensor = torch.FloatTensor(img_array).permute(2, 0, 1)

        return tensor, tensor