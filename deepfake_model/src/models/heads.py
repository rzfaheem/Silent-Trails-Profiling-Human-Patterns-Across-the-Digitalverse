"""
Classification + Metric Learning Heads.

ClassificationHead: Binary output (real/fake) via BCE + Focal loss.
MetricHead: Projects embeddings to unit hypersphere for triplet loss —
            pushes real/fake embeddings apart in metric space,
            improving generalization to unseen forgery methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """
    Binary classification: real vs. fake.

    Input:  (B, dim) fused embedding
    Output: (B, 1) raw logits (apply sigmoid for probability)
    """

    def __init__(self, dim=512, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.head(x)


class MetricHead(nn.Module):
    """
    Metric learning projection for triplet loss.

    Projects embeddings to a lower-dimensional unit hypersphere.
    Real images cluster together, fake images are pushed apart.
    This embedding space generalizes better to unseen generators.

    Input:  (B, dim) fused embedding
    Output: (B, embed_dim) L2-normalized embedding
    """

    def __init__(self, dim=512, embed_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=-1)
