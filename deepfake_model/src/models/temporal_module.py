"""
Temporal Transformer — Video-level temporal reasoning.

Processes sequences of per-frame embeddings to detect temporal
inconsistencies: blinking anomalies, identity drift, flicker artifacts,
and inter-frame manipulation patterns.

Only used for video inputs. Skipped for single images.
"""

import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):
    """
    Transformer encoder over frame-level embeddings for video analysis.

    Input:  (B, T, dim) sequence of per-frame fused embeddings
    Output: (B, dim) video-level embedding (CLS token)
    """

    def __init__(self, dim=512, num_layers=4, num_heads=8, max_frames=32, dropout=0.1):
        super().__init__()

        # Learnable positional embeddings for frame positions
        self.pos_embed = nn.Parameter(torch.randn(1, max_frames + 1, dim))

        # Learnable CLS token for video-level representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(dim)

    def forward(self, frame_embeddings, mask=None):
        """
        Args:
            frame_embeddings: (B, T, dim) per-frame fused embeddings
            mask: (B, T+1) optional padding mask for variable-length videos

        Returns:
            video_embedding: (B, dim) video-level representation
        """
        B, T, D = frame_embeddings.shape

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)                    # (B, 1, dim)
        x = torch.cat([cls, frame_embeddings], dim=1)             # (B, T+1, dim)

        # Add positional embeddings
        x = x + self.pos_embed[:, :T + 1]

        # Transformer encoding
        x = self.encoder(x, src_key_padding_mask=mask)

        # CLS token output → video-level embedding
        video_embedding = self.norm(x[:, 0])  # (B, dim)

        return video_embedding
