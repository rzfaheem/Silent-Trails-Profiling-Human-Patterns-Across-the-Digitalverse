"""
Multi-Dataset Loader for deepfake detection training.

Handles loading face crops from multiple datasets (FF++, CelebDF, DFDC)
with unified label format, balanced sampling, and curriculum support.
"""

import os
import random
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class DeepfakeDataset(Dataset):
    """
    Unified dataset loader for face crops across multiple deepfake datasets.

    Expected directory structure (after face extraction):
        data_root/
        ├── ff++/
        │   ├── real/         # original face crops
        │   └── fake/         # manipulated face crops
        ├── celebdf/
        │   ├── real/
        │   └── fake/
        └── dfdc/
            ├── real/
            └── fake/
    """

    def __init__(self, data_root, datasets=None, split="train",
                 transform=None, max_samples=None, difficulty=None):
        """
        Args:
            data_root: root directory containing dataset folders
            datasets: list of dataset names to include (e.g., ["ff++", "celebdf"])
            split: "train", "val", or "test"
            transform: albumentations transform pipeline
            max_samples: cap total samples (for debugging)
            difficulty: "easy", "medium", "hard", or None for all
        """
        self.data_root = Path(data_root)
        self.transform = transform
        self.samples = []  # list of (image_path, label, dataset_name)

        if datasets is None:
            datasets = ["ff++", "celebdf", "dfdc"]

        for ds_name in datasets:
            ds_path = self.data_root / ds_name
            if not ds_path.exists():
                print(f"  [WARN] Dataset not found: {ds_path}")
                continue

            # Load real samples
            real_dir = ds_path / "real"
            if real_dir.exists():
                for img_path in real_dir.rglob("*.jpg"):
                    self.samples.append((str(img_path), 0, ds_name))

            # Load fake samples
            fake_dir = ds_path / "fake"
            if fake_dir.exists():
                for img_path in fake_dir.rglob("*.jpg"):
                    self.samples.append((str(img_path), 1, ds_name))

        # Shuffle
        random.shuffle(self.samples)

        # Cap if needed
        if max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

        print(f"  [{split}] Loaded {len(self.samples)} samples from {datasets}")
        self._print_stats()

    def _print_stats(self):
        """Print dataset composition."""
        real = sum(1 for _, l, _ in self.samples if l == 0)
        fake = sum(1 for _, l, _ in self.samples if l == 1)
        print(f"    Real: {real} | Fake: {fake} | Ratio: {real/(real+fake):.2%} real")

        # Per-dataset breakdown
        ds_counts = {}
        for _, _, ds in self.samples:
            ds_counts[ds] = ds_counts.get(ds, 0) + 1
        for ds, count in sorted(ds_counts.items()):
            print(f"    {ds}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, ds_name = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        # Apply augmentation
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]  # albumentations returns dict
        else:
            # Default: resize + normalize
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Quality pseudo-labels placeholder (filled by augmentation pipeline)
        quality_labels = torch.zeros(3)  # [compression, blur, motion]

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "quality": quality_labels,
            "dataset": ds_name,
        }

    def get_balanced_sampler(self):
        """Get a WeightedRandomSampler that balances real/fake classes."""
        labels = [s[1] for s in self.samples]
        class_counts = [labels.count(0), labels.count(1)]
        weights = [1.0 / class_counts[l] for l in labels]
        return WeightedRandomSampler(weights, len(weights))


def create_dataloaders(data_root, batch_size=32, num_workers=4,
                       train_transform=None, val_transform=None,
                       datasets=None, val_ratio=0.15, test_ratio=0.1):
    """
    Create train/val/test dataloaders.

    Args:
        data_root: root data directory
        batch_size: batch size
        num_workers: dataloader workers
        train_transform: albumentations transform for training
        val_transform: albumentations transform for validation
        datasets: which datasets to include
        val_ratio: validation split ratio
        test_ratio: test split ratio

    Returns:
        train_loader, val_loader, test_loader
    """
    # For now, create full dataset and split
    full_dataset = DeepfakeDataset(
        data_root, datasets=datasets, split="full",
        transform=None, max_samples=None,
    )

    # Split indices
    n = len(full_dataset)
    indices = list(range(n))
    random.shuffle(indices)

    test_end = int(n * test_ratio)
    val_end = test_end + int(n * val_ratio)

    test_indices = indices[:test_end]
    val_indices = indices[test_end:val_end]
    train_indices = indices[val_end:]

    # Create subset datasets with appropriate transforms
    train_dataset = DeepfakeDataset(
        data_root, datasets=datasets, split="train",
        transform=train_transform,
    )
    val_dataset = DeepfakeDataset(
        data_root, datasets=datasets, split="val",
        transform=val_transform,
    )

    # Balanced sampling for training
    sampler = train_dataset.get_balanced_sampler()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=sampler, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
