import torch
import torch.nn as nn

class FullyConvolutionalAE(nn.Module):
    def __init__(self, latent_channels=64): # Number of channels in the bottleneck
        super().__init__()

        # Encoder
        # Input: (B, 3, H, W)
        self.encoder = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # (B, 32, H/2, W/2)
            nn.ReLU(inplace=True),
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # (B, 64, H/4, W/4)
            nn.ReLU(inplace=True),
            # Layer 3
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # (B, 128, H/8, W/8)
            nn.ReLU(inplace=True),
            # Bottleneck Layer
            nn.Conv2d(128, latent_channels, kernel_size=3, stride=1, padding=1), # (B, latent_channels, H/8, W/8)
            nn.ReLU(inplace=True) # Or Tanh, or nothing, depending on latent space design
        )

        # Decoder
        # Input: (B, latent_channels, H_latent, W_latent) where H_latent=H/8, W_latent=W/8
        self.decoder = nn.Sequential(
            # Layer 1 (from bottleneck)
            nn.ConvTranspose2d(latent_channels, 128, kernel_size=3, stride=1, padding=1), # (B, 128, H/8, W/8)
            nn.ReLU(inplace=True),
            # Layer 2
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # (B, 64, H/4, W/4)
            nn.ReLU(inplace=True),
            # Layer 3
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # (B, 32, H/2, W/2)
            nn.ReLU(inplace=True),
            # Output Layer
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),   # (B, 3, H, W)
            nn.Sigmoid()  # To ensure output is in [0, 1]
        )

    def forward(self, x):
        # Note: Input x can be of any H, W (ideally divisible by 8 for this arch)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# Rename CompressionAE to FullyConvolutionalAE where it's used,
# or just replace the content of CompressionAE if you prefer.
# For this example, I'll assume you'll use FullyConvolutionalAE
# but you can rename it back to CompressionAE if you update all files.