# File 1: dataset.py (Custom Dataset Loader)
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class ImageCompressionDataset(Dataset):
    def __init__(self, root_dir, img_size=256):
        self.root_dir = root_dir
        self.img_size = img_size
        self.image_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')

        # Manual resizing and normalization
        img = img.resize((self.img_size, self.img_size))
        img_array = np.array(img) / 255.0
        tensor = torch.FloatTensor(img_array).permute(2, 0, 1)

        return tensor, tensor  # Return same tensor for input/target