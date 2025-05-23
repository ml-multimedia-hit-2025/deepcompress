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

# ---------- Dataset ----------
class CIFAR10Dataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data["data"])

    def __getitem__(self, idx):
        img = self.data["data"][idx].reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, img  # autoencoder: input == target


# ---------- Transform ----------
class ToTensor:
    def __call__(self, img):
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        return torch.tensor(arr, dtype=torch.float32)

class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return img.resize((self.size, self.size), Image.LANCZOS)

transform = lambda img: ToTensor()(Resize(256)(img))


# ---------- Download CIFAR-10 ----------
def download_and_extract_cifar10():
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
    def __init__(self, latent_channels=64):
        super().__init__()
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
        return self.decoder(self.encoder(x))


# ---------- Training ----------
def train():
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
