"""
Spatial Stream — DINOv2 backbone with LoRA adapters.

Extracts spatial features from face crops using a frozen DINOv2 model
with lightweight trainable LoRA adapters for domain adaptation.
Only ~0.5M trainable params vs. 19M for full fine-tuning.
"""

import torch
import torch.nn as nn
from transformers import Dinov2Model
from peft import LoraConfig, get_peft_model


class SpatialStream(nn.Module):
    """
    DINOv2-Base backbone with LoRA adapters for deepfake spatial feature extraction.

    Input:  (B, 3, 256, 256) face crops
    Output: h_spatial (B, 512), patch_tokens (B, 256, 768)
    """

    def __init__(self, model_name="facebook/dinov2-base", lora_r=8, lora_alpha=16,
                 lora_dropout=0.1, embed_dim=512):
        super().__init__()

        # Load frozen DINOv2 backbone
        self.backbone = Dinov2Model.from_pretrained(model_name)

        # Apply LoRA adapters to attention layers
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["query", "value"],
            lora_dropout=lora_dropout,
        )
        self.backbone = get_peft_model(self.backbone, lora_config)

        # Projection to common embedding dimension
        self.proj = nn.Linear(768, embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, 3, 256, 256) face crops

        Returns:
            h_spatial: (B, embed_dim) spatial embedding
            patch_tokens: (B, num_patches, 768) for attention stream
        """
        features = self.backbone(x).last_hidden_state  # (B, 257, 768) — 1 CLS + 256 patches
        cls_token = features[:, 0]                      # (B, 768)
        patch_tokens = features[:, 1:]                  # (B, 256, 768)

        h_spatial = self.proj(cls_token)                # (B, embed_dim)
        return h_spatial, patch_tokens

    def get_trainable_params(self):
        """Return count of trainable parameters (LoRA only)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
