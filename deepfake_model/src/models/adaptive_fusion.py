"""
Adaptive Fusion Engine — Quality-aware stream combination.

Uses a Quality Estimator module to predict input degradation levels
(compression, blur, motion), then dynamically weights the three
streams based on predicted quality.

Example: If image is heavily compressed, reduce frequency stream weight
(frequency artifacts get destroyed by compression) and increase
spatial stream weight.
"""

import torch
import torch.nn as nn


class QualityEstimator(nn.Module):
    """
    Predicts input quality factors from stream embeddings.

    Output: (B, 3) — [compression_level, blur_level, motion_level] ∈ [0, 1]

    Ground truth: pseudo-labels from known augmentation parameters
    (e.g., JPEG quality, blur sigma applied during training).
    """

    def __init__(self, dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 3, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3),
            nn.Sigmoid(),
        )

    def forward(self, h_spatial, h_freq, h_attn):
        """
        Args:
            h_spatial, h_freq, h_attn: (B, dim) stream embeddings

        Returns:
            quality: (B, 3) predicted quality factors
        """
        combined = torch.cat([h_spatial, h_freq, h_attn], dim=-1)
        return self.net(combined)


class AdaptiveFusionEngine(nn.Module):
    """
    Dynamically weights stream contributions based on predicted input quality.

    Input:  Three stream embeddings (B, dim) each
    Output: fused (B, dim), quality (B, 3)
    """

    def __init__(self, dim=512):
        super().__init__()

        self.quality_est = QualityEstimator(dim)

        # Learnable weight network: quality + embeddings → stream weights
        self.weight_net = nn.Sequential(
            nn.Linear(3 + dim * 3, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3),  # one weight per stream
        )

        self.out_proj = nn.Linear(dim, dim)

    def forward(self, h_spatial, h_freq, h_attn):
        """
        Args:
            h_spatial: (B, dim) spatial stream embedding
            h_freq:    (B, dim) frequency stream embedding
            h_attn:    (B, dim) attention stream embedding

        Returns:
            fused:   (B, dim) quality-adaptive fused embedding
            quality: (B, 3) predicted quality factors
        """
        # Predict quality
        quality = self.quality_est(h_spatial, h_freq, h_attn)  # (B, 3)

        # Compute adaptive weights
        weight_input = torch.cat([quality, h_spatial, h_freq, h_attn], dim=-1)
        weights = torch.softmax(self.weight_net(weight_input), dim=-1)  # (B, 3)

        # Weighted fusion
        fused = (
            weights[:, 0:1] * h_spatial +
            weights[:, 1:2] * h_freq +
            weights[:, 2:3] * h_attn
        )

        fused = self.out_proj(fused)  # (B, dim)
        return fused, quality
