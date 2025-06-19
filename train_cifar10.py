import os
import tarfile
import urllib.request
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

"""
train_cifar10.py

Script to train a Fully Convolutional Autoencoder (FCN) on the CIFAR-10 dataset for image compression.
Handles dataset download, preprocessing, model definition, and training loop.
"""

# ---------- Dataset ----------
class CIFAR10Dataset(Dataset):
    """
    PyTorch Dataset for loading CIFAR-10 data for autoencoder training.

    Args:
        data (dict): Dictionary with 'data' and 'labels' loaded from CIFAR-10 batches.
        transform (callable, optional): Optional transform to apply to images.
    """
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        return len(self.data["data"])

    def __getitem__(self, idx):
        """
        Loads and preprocesses an image at the given index.

        Args:
            idx (int): Index of the image to load.

        Returns:
            tuple: (input_tensor, target_tensor) for autoencoder training.
        """
        img = self.data["data"][idx].reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, img  # autoencoder: input == target


# ---------- Transform ----------
class ToTensor:
    """
    Transform class to convert a PIL image to a normalized torch tensor.
    """
    def __call__(self, img):
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        return torch.tensor(arr, dtype=torch.float32)

class Resize:
    """
    Transform class to resize a PIL image to a given size using LANCZOS filter.

    Args:
        size (int): Target size for both width and height.
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return img.resize((self.size, self.size), Image.LANCZOS)

transform = lambda img: ToTensor()(Resize(256)(img))


# ---------- Download CIFAR-10 ----------
def download_and_extract_cifar10():
    """
    Downloads and extracts the CIFAR-10 dataset if not already present in the working directory.
    """
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    dest = "cifar-10-python.tar.gz"
    if not os.path.exists(dest):
        print("[↓] Downloading CIFAR-10...")
        urllib.request.urlretrieve(url, dest)

    if not os.path.exists("cifar-10-batches-py"):
        print("[✱] Extracting...")
        with tarfile.open(dest) as tar:
            tar.extractall()
    print("[✓] CIFAR-10 ready.")


# ---------- Load CIFAR-10 Batches ----------
def load_cifar10_data():
    """
    Loads all CIFAR-10 data batches into a single dictionary.

    Returns:
        dict: Dictionary with 'data' and 'labels' keys.
    """
    data = {"data": [], "labels": []}
    for i in range(1, 6):
        with open(f"cifar-10-batches-py/data_batch_{i}", "rb") as f:
            batch = pickle.load(f, encoding="bytes")
            data["data"].extend(batch[b"data"])
            data["labels"].extend(batch[b"labels"])
    data["data"] = np.array(data["data"])
    return data


# ---------- Model ----------
class FullyConvolutionalAE(nn.Module):
    """
    Fully Convolutional Autoencoder for image compression.

    Args:
        latent_channels (int, optional): Number of channels in the bottleneck latent space. Default is 64.
    """
    def __init__(self, latent_channels=64):
        super().__init__()
        # Encoder: Downsamples input image to latent representation
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, latent_channels, 3, 1, 1),
            nn.ReLU()
        )
        # Decoder: Upsamples latent representation back to image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Reconstructed image tensor of shape (B, 3, H, W).
        """
        return self.decoder(self.encoder(x))


# ---------- Training ----------
def train():
    """
    Trains the Fully Convolutional Autoencoder on the CIFAR-10 dataset.
    Saves the trained model to disk after training.
    """
    download_and_extract_cifar10()
    data_dict = load_cifar10_data()
    dataset = CIFAR10Dataset(data_dict, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = FullyConvolutionalAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(5):
        total_loss = 0
        for x, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            x = x.to(device)
            y = model(x)
            loss = criterion(y, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "fcn_compression_ae3.pth")
    print("[✓] Model saved to fcn_compression_ae.pth")


# ---------- Run ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    train()
