"""
Frequency Stream — FFT-based spectral analysis.

Converts face crops to frequency domain via 2D FFT, then processes
the log-magnitude spectrum through a CNN to detect:
- GAN checkerboard artifacts
- Diffusion model spectral signatures
- Upsampling/interpolation patterns

These artifacts are invisible in the spatial domain but clearly visible
in the frequency domain.
"""

import torch
import torch.nn as nn


class FrequencyStream(nn.Module):
    """
    FFT → log-magnitude → CNN → embedding.

    Input:  (B, 3, 256, 256) face crops
    Output: h_freq (B, embed_dim)
    """

    def __init__(self, embed_dim=512):
        super().__init__()

        self.cnn = nn.Sequential(
            # Block 1: 256x256 → 128x128
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),

            # Block 2: 128x128 → 64x64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),

            # Block 3: 64x64 → 16x16
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(16),

            # Block 4: 16x16 → 1x1
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),
        )

        self.proj = nn.Linear(256, embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, 3, 256, 256) face crops (RGB)

        Returns:
            h_freq: (B, embed_dim) frequency embedding
        """
        # Convert to grayscale
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]  # (B, H, W)

        # 2D FFT → shift DC to center → log-magnitude
        freq = torch.fft.fftshift(torch.fft.fft2(gray))
        magnitude = torch.log(torch.abs(freq) + 1e-8)  # (B, H, W)
        magnitude = magnitude.unsqueeze(1)               # (B, 1, H, W)

        # CNN feature extraction
        features = self.cnn(magnitude)    # (B, 256)
        h_freq = self.proj(features)      # (B, embed_dim)

        return h_freq
