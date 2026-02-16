"""
Attention Forgery Stream — Region-based attention for localization.

Uses learnable region queries (eyes, mouth, jaw, hair) to attend to
specific facial regions in the DINOv2 patch tokens. This provides:
1. Region-aware forgery features
2. Attention weight maps → Grad-CAM heatmaps showing WHERE manipulation was detected
"""

import torch
import torch.nn as nn


class AttentionForgeryStream(nn.Module):
    """
    Cross-attention from learnable region queries to backbone patch tokens.

    Input:  patch_tokens (B, num_patches, backbone_dim) from SpatialStream
    Output: h_attn (B, embed_dim), attn_weights (B, num_regions, num_patches)
    """

    def __init__(self, backbone_dim=768, num_regions=4, num_heads=8, embed_dim=512):
        super().__init__()

        # Learnable region queries: eyes, mouth, jaw, hair
        self.region_queries = nn.Parameter(torch.randn(num_regions, backbone_dim))

        # Multi-head cross-attention
        self.region_attn = nn.MultiheadAttention(
            embed_dim=backbone_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )

        # Layer norm for stable training
        self.norm = nn.LayerNorm(backbone_dim)

        # Project to common embedding dimension
        self.proj = nn.Linear(backbone_dim, embed_dim)

    def forward(self, patch_tokens):
        """
        Args:
            patch_tokens: (B, num_patches, 768) from DINOv2 backbone

        Returns:
            h_attn: (B, embed_dim) attention-based forgery embedding
            attn_weights: (B, num_regions, num_patches) — reshape to heatmap
        """
        B = patch_tokens.size(0)

        # Expand queries for batch
        queries = self.region_queries.unsqueeze(0).expand(B, -1, -1)  # (B, 4, 768)

        # Cross-attention: queries attend to patch tokens
        attn_out, attn_weights = self.region_attn(
            query=queries,
            key=patch_tokens,
            value=patch_tokens,
        )
        # attn_out: (B, 4, 768), attn_weights: (B, 4, num_patches)

        # Pool across regions and project
        pooled = self.norm(attn_out.mean(dim=1))  # (B, 768)
        h_attn = self.proj(pooled)                 # (B, embed_dim)

        return h_attn, attn_weights

    def get_heatmap(self, attn_weights, patch_size=16, face_size=256):
        """
        Convert attention weights to a spatial heatmap.

        Args:
            attn_weights: (B, num_regions, num_patches)
            patch_size: size of each DINOv2 patch (16px for dinov2-base)
            face_size: original face crop size

        Returns:
            heatmap: (B, 1, face_size, face_size) attention heatmap
        """
        B = attn_weights.size(0)
        grid_size = face_size // patch_size  # 256/16 = 16

        # Average across regions → (B, num_patches)
        avg_attn = attn_weights.mean(dim=1)

        # Reshape to spatial grid → (B, 1, 16, 16)
        heatmap = avg_attn.view(B, 1, grid_size, grid_size)

        # Upscale to face size → (B, 1, 256, 256)
        heatmap = nn.functional.interpolate(
            heatmap, size=(face_size, face_size),
            mode='bilinear', align_corners=False
        )

        # Normalize to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return heatmap
