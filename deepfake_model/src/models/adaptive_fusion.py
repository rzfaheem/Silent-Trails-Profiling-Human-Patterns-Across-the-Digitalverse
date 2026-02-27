"""
Fusion Engines for combining multi-stream embeddings.

Two options:
- SimpleFusion (Phase 1): Concat + gating MLP. Easy to train, easy to debug.
- AdaptiveFusionEngine (Phase 3): Quality-aware dynamic weighting.

Phase 1 uses SimpleFusion by default. Upgrade to AdaptiveFusionEngine
once the core model is validated and stable.
"""

import torch
import torch.nn as nn


class SimpleFusion(nn.Module):
    """
    Phase 1 fusion: Concatenate stream embeddings → gating MLP → output.

    Simple and effective. The MLP implicitly learns which streams to trust
    without needing explicit quality estimation or pseudo-labels.

    Input:  Three stream embeddings (B, dim) each
    Output: fused (B, dim), quality placeholder (B, 3)
    """

    def __init__(self, dim=512, dropout=0.2):
        super().__init__()

        # Gating MLP: 1536 (3×512 concat) → 512
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, h_spatial, h_freq, h_attn):
        """
        Args:
            h_spatial: (B, dim) spatial stream embedding
            h_freq:    (B, dim) frequency stream embedding
            h_attn:    (B, dim) attention stream embedding

        Returns:
            fused:   (B, dim) fused embedding
            quality: (B, 3) placeholder zeros (no quality estimation in Phase 1)
        """
        combined = torch.cat([h_spatial, h_freq, h_attn], dim=-1)  # (B, dim*3)
        fused = self.gate(combined)  # (B, dim)

        # Return dummy quality to keep API compatible with AdaptiveFusionEngine
        quality = torch.zeros(h_spatial.size(0), 3, device=h_spatial.device)

        return fused, quality


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
