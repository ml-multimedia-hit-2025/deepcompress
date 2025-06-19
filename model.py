import torch
import torch.nn as nn

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
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # (B, 32, H/2, W/2)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # (B, 64, H/4, W/4)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # (B, 128, H/8, W/8)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_channels, kernel_size=3, stride=1, padding=1), # (B, latent_channels, H/8, W/8)
            nn.ReLU(inplace=True)
        )
        # Decoder: Upsamples latent representation back to image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, kernel_size=3, stride=1, padding=1), # (B, 128, H/8, W/8)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # (B, 64, H/4, W/4)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # (B, 32, H/2, W/2)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),   # (B, 3, H, W)
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Reconstructed image tensor of shape (B, 3, H, W).
        """
        # Encode input to latent space
        latent = self.encoder(x)
        # Decode latent to reconstruct image
        reconstructed = self.decoder(latent)
        return reconstructed
