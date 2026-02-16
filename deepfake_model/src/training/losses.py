"""
Loss Functions — Compound loss for deepfake detection.

Total Loss = BCE + Focal + Triplet + Quality MSE

Each loss targets a different aspect:
- BCE: basic binary classification
- Focal: focus on hard-to-classify samples
- Triplet: push real/fake apart in embedding space
- Quality MSE: train quality estimator with pseudo-labels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss — downweights easy examples, focuses on hard ones.
    Very helpful for imbalanced or curriculum training.
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1) raw model output
            targets: (B, 1) binary labels (0=real, 1=fake)
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)  # probability of correct class
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class TripletLoss(nn.Module):
    """
    Simplified triplet loss for metric learning.

    Pushes real embeddings close together and fake embeddings away.
    Uses batch-level anchor (mean of real embeddings).
    """

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, embed_dim) L2-normalized embeddings
            labels: (B,) binary labels (0=real, 1=fake)
        """
        real_mask = labels == 0
        fake_mask = labels == 1

        real_emb = embeddings[real_mask]
        fake_emb = embeddings[fake_mask]

        # Need at least one of each
        if len(real_emb) < 1 or len(fake_emb) < 1:
            return torch.tensor(0.0, device=embeddings.device)

        # Anchor = mean of real embeddings
        anchor = real_emb.mean(0, keepdim=True)

        # Positive distance: anchor ↔ real
        d_pos = F.pairwise_distance(anchor.expand_as(real_emb), real_emb).mean()

        # Negative distance: anchor ↔ fake
        d_neg = F.pairwise_distance(anchor.expand_as(fake_emb), fake_emb).mean()

        # Triplet loss: want d_pos < d_neg by at least margin
        loss = F.relu(d_pos - d_neg + self.margin)
        return loss


class CompoundLoss(nn.Module):
    """
    Combined loss for deepfake detection training.

    Total = bce_w * BCE + focal_w * Focal + triplet_w * Triplet + quality_w * QualityMSE
    """

    def __init__(self, bce_weight=1.0, focal_weight=0.5, focal_gamma=2.0,
                 triplet_weight=0.5, triplet_margin=0.3, quality_weight=0.1):
        super().__init__()

        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.triplet_weight = triplet_weight
        self.quality_weight = quality_weight

        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss(gamma=focal_gamma)
        self.triplet = TripletLoss(margin=triplet_margin)
        self.quality_mse = nn.MSELoss()

    def forward(self, outputs, targets, quality_labels=None):
        """
        Args:
            outputs: dict from DeepfakeForensicsModel.forward()
                - logits: (B, 1)
                - embedding: (B, 128)
                - quality: (B, 3)
            targets: (B,) binary labels (0=real, 1=fake)
            quality_labels: (B, 3) optional pseudo ground-truth quality

        Returns:
            total_loss, loss_dict with individual loss components
        """
        logits = outputs["logits"]
        targets_float = targets.float().unsqueeze(1)  # (B, 1)

        # BCE loss
        loss_bce = self.bce(logits, targets_float)

        # Focal loss
        loss_focal = self.focal(logits, targets_float)

        # Triplet loss
        loss_triplet = self.triplet(outputs["embedding"], targets)

        # Quality estimation loss
        loss_quality = torch.tensor(0.0, device=logits.device)
        if quality_labels is not None and self.quality_weight > 0:
            loss_quality = self.quality_mse(outputs["quality"], quality_labels)

        # Total
        total = (
            self.bce_weight * loss_bce +
            self.focal_weight * loss_focal +
            self.triplet_weight * loss_triplet +
            self.quality_weight * loss_quality
        )

        loss_dict = {
            "total": total.item(),
            "bce": loss_bce.item(),
            "focal": loss_focal.item(),
            "triplet": loss_triplet.item(),
            "quality": loss_quality.item(),
        }

        return total, loss_dict
