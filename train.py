"""
train.py

Script for training a compression autoencoder on a custom image dataset using PyTorch.
Handles dataset loading, model training, and saving the trained model.
"""
import os
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import ImageCompressionDataset
from model import CompressionAE

def train_model():
    """
    Trains the CompressionAE model on the ImageCompressionDataset.
    Loads data, runs the training loop, and saves the trained model to disk.
    """
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001

    # Initialize dataset and loader
    dataset = ImageCompressionDataset("./data/train")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model and optimizer
    model = CompressionAE().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Print progress
        avg_loss = epoch_loss / len(dataloader)
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} Time: {epoch_time:.2f}s")

    # Save model
    torch.save(model.state_dict(), "compression_ae.pth")

if __name__ == "__main__":
    train_model()