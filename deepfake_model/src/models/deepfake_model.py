"""
Deepfake Forensics — Full Model Assembly.

Combines all streams (spatial, frequency, attention), adaptive fusion,
optional temporal transformer (video), and classification + metric heads
into a single end-to-end model.
"""

import torch
import torch.nn as nn

from .spatial_stream import SpatialStream
from .frequency_stream import FrequencyStream
from .attention_stream import AttentionForgeryStream
from .adaptive_fusion import AdaptiveFusionEngine
from .temporal_module import TemporalTransformer
from .heads import ClassificationHead, MetricHead


class DeepfakeForensicsModel(nn.Module):
    """
    Full deepfake forensics model.

    Image mode:  (B, 3, 256, 256) → prediction
    Video mode:  (B, T, 3, 256, 256) → prediction

    Returns dict with: logits, probability, embedding, attention maps, quality estimates.
    """

    def __init__(self, video_mode=False, lora_r=8, lora_alpha=16, embed_dim=512):
        super().__init__()

        self.video_mode = video_mode

        # Three analysis streams
        self.spatial = SpatialStream(lora_r=lora_r, lora_alpha=lora_alpha, embed_dim=embed_dim)
        self.frequency = FrequencyStream(embed_dim=embed_dim)
        self.attention = AttentionForgeryStream(embed_dim=embed_dim)

        # Quality-adaptive fusion
        self.fusion = AdaptiveFusionEngine(dim=embed_dim)

        # Temporal module (video only)
        self.temporal = TemporalTransformer(dim=embed_dim) if video_mode else None

        # Output heads
        self.cls_head = ClassificationHead(dim=embed_dim)
        self.metric_head = MetricHead(dim=embed_dim)

    def forward_frame(self, face_crop):
        """
        Process a single face crop through all three streams + fusion.

        Args:
            face_crop: (B, 3, 256, 256) aligned face crop

        Returns:
            fused:    (B, embed_dim) fused embedding
            attn_map: (B, num_regions, num_patches) attention weights
            quality:  (B, 3) predicted quality factors
        """
        # Spatial stream: DINOv2 + LoRA
        h_spatial, patch_tokens = self.spatial(face_crop)

        # Frequency stream: FFT + CNN
        h_freq = self.frequency(face_crop)

        # Attention stream: region queries on patch tokens
        h_attn, attn_map = self.attention(patch_tokens)

        # Adaptive fusion
        fused, quality = self.fusion(h_spatial, h_freq, h_attn)

        return fused, attn_map, quality

    def forward(self, x, mask=None):
        """
        Full forward pass.

        Args:
            x: Image (B, 3, 256, 256) or Video (B, T, 3, 256, 256)
            mask: Optional padding mask for video (B, T+1)

        Returns:
            dict with keys: logits, probability, embedding, attn_maps, quality
        """
        if self.video_mode and x.dim() == 5:
            # Video mode: process each frame, then temporal reasoning
            B, T = x.shape[:2]
            frames_flat = x.view(B * T, *x.shape[2:])  # (B*T, 3, 256, 256)

            fused, attn_maps, quality = self.forward_frame(frames_flat)
            fused = fused.view(B, T, -1)  # (B, T, embed_dim)

            # Temporal transformer
            fused = self.temporal(fused, mask)  # (B, embed_dim)

            # Reshape attn_maps for per-frame output
            attn_maps = attn_maps.view(B, T, *attn_maps.shape[1:])
            quality = quality.view(B, T, -1)
        else:
            # Image mode
            fused, attn_maps, quality = self.forward_frame(x)

        # Classification + metric heads
        logits = self.cls_head(fused)       # (B, 1)
        embedding = self.metric_head(fused)  # (B, 128)

        return {
            "logits": logits,
            "probability": torch.sigmoid(logits),
            "embedding": embedding,
            "attn_maps": attn_maps,
            "quality": quality,
        }

    def get_param_groups(self, lr_lora=5e-4, lr_new=1e-3):
        """
        Get parameter groups with different learning rates.

        - LoRA params: lower LR (adapting pretrained features)
        - New modules: higher LR (learning from scratch)
        """
        lora_params = []
        new_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_" in name:
                lora_params.append(param)
            else:
                new_params.append(param)

        return [
            {"params": lora_params, "lr": lr_lora},
            {"params": new_params, "lr": lr_new},
        ]

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "trainable_pct": round(trainable / total * 100, 2),
        }
