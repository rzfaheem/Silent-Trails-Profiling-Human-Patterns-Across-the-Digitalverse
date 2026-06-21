"""
Multi-Dataset Loader for deepfake detection training.

Handles:
- Image-mode training (individual face crops) — Stage 1
- Video-mode training (frame sequences grouped by video) — Stage 2
- Multiple datasets (FF++, CelebDF, DFDC, custom)
- Balanced sampling across datasets and classes
- Proper train/val/test splitting with SEPARATE transforms

Fixes over the original version:
- Data is loaded ONCE, then split by index (no duplicate loading)
- Train and val/test get DIFFERENT transforms (no augmentation leaking)
- Supports both subdirectory-per-video and flat file layouts
- Dataset-balanced sampling (not just class-balanced)
"""

import os
import re
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ─────────────────────────────────────────────────────────────
# Image Dataset (Stage 1: frame-level training)
# ─────────────────────────────────────────────────────────────

class DeepfakeDataset(Dataset):
    """
    Dataset for loading individual face crops (image-mode / Stage 1).

    Expected directory structure (after face extraction):
        data_root/
        ├── ff++/
        │   ├── real/         # face crops from real videos
        │   └── fake/         # face crops from manipulated videos
        ├── celebdf/
        │   ├── real/
        │   └── fake/
        └── dfdc/
            ├── real/
            └── fake/

    Face crops can be in flat directories or nested subdirectories.
    Supports: .jpg, .jpeg, .png, .webp, .bmp
    """

    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    def __init__(self, data_root, datasets=None, transform=None, max_samples=None):
        """
        Args:
            data_root: root directory containing dataset folders
            datasets: list of dataset names (e.g., ["ff++", "dfdc"])
            transform: albumentations transform pipeline
            max_samples: cap total samples (for debugging)
        """
        self.data_root = Path(data_root)
        self.transform = transform
        self.samples = []  # list of (image_path, label, dataset_name)

        if datasets is None:
            datasets = ["ff++", "celebdf", "dfdc"]

        for ds_name in datasets:
            ds_path = self.data_root / ds_name
            if not ds_path.exists():
                print(f"  [WARN] Dataset directory not found: {ds_path}")
                continue

            for label_name, label in [("real", 0), ("fake", 1)]:
                label_dir = ds_path / label_name
                if not label_dir.exists():
                    print(f"  [WARN] Missing: {label_dir}")
                    continue

                files = self._scan_images(label_dir)
                for img_path in files:
                    self.samples.append((str(img_path), label, ds_name))

        # Shuffle deterministically
        random.Random(42).shuffle(self.samples)

        # Cap for debugging
        if max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

        self._print_stats()

    def _scan_images(self, directory):
        """Recursively find all image files."""
        files = []
        for f in Path(directory).rglob("*"):
            if f.suffix.lower() in self.EXTENSIONS and f.is_file():
                files.append(f)
        return sorted(files)

    def _print_stats(self):
        """Print dataset composition."""
        real = sum(1 for _, l, _ in self.samples if l == 0)
        fake = sum(1 for _, l, _ in self.samples if l == 1)
        total = len(self.samples)
        if total == 0:
            print("  [WARN] No samples loaded!")
            return
        print(f"  Loaded {total} samples | Real: {real} ({real/total:.0%}) | Fake: {fake} ({fake/total:.0%})")

        ds_counts = defaultdict(lambda: {"real": 0, "fake": 0})
        for _, label, ds in self.samples:
            ds_counts[ds]["real" if label == 0 else "fake"] += 1
        for ds, counts in sorted(ds_counts.items()):
            print(f"    {ds}: {counts['real']} real + {counts['fake']} fake")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, ds_name = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.float32),
            "dataset": ds_name,
        }

    def get_balanced_sampler(self):
        """Weighted sampler balancing BOTH class and dataset source."""
        labels = [s[1] for s in self.samples]
        datasets = [s[2] for s in self.samples]

        combo_counts = defaultdict(int)
        for l, d in zip(labels, datasets):
            combo_counts[(l, d)] += 1

        weights = [1.0 / combo_counts[(l, d)] for l, d in zip(labels, datasets)]
        return WeightedRandomSampler(weights, len(weights))


# ─────────────────────────────────────────────────────────────
# Video Dataset (Stage 2: temporal / sequence training)
# ─────────────────────────────────────────────────────────────

class VideoDeepfakeDataset(Dataset):
    """
    Dataset for loading frame SEQUENCES for video-mode training (Stage 2).

    Groups face crops by source video, then samples T frames per video.

    Supports two directory layouts:
      1) Subdirectory per video:
         data_root/dataset/real/video_001/frame_000.jpg
      2) Flat directory with naming convention:
         data_root/dataset/real/video_001_frame000.jpg
    """

    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    def __init__(self, data_root, datasets=None, num_frames=16,
                 transform=None, max_videos=None, min_frames=4):
        """
        Args:
            data_root: root directory containing dataset folders
            datasets: list of dataset names
            num_frames: frames to sample per video (default 16)
            transform: albumentations transform pipeline
            max_videos: cap total videos (for debugging)
            min_frames: minimum frames a video must have to be included
        """
        self.data_root = Path(data_root)
        self.transform = transform
        self.num_frames = num_frames
        self.videos = []  # list of (sorted_frame_paths, label, dataset_name)

        if datasets is None:
            datasets = ["ff++", "celebdf", "dfdc"]

        for ds_name in datasets:
            ds_path = self.data_root / ds_name
            if not ds_path.exists():
                print(f"  [WARN] Dataset not found: {ds_path}")
                continue

            for label_name, label in [("real", 0), ("fake", 1)]:
                label_dir = ds_path / label_name
                if not label_dir.exists():
                    continue

                video_groups = self._group_by_video(label_dir)

                for video_id, frame_paths in video_groups.items():
                    if len(frame_paths) < min_frames:
                        continue
                    frame_paths = sorted(frame_paths)
                    self.videos.append((frame_paths, label, ds_name))

        random.Random(42).shuffle(self.videos)

        if max_videos and len(self.videos) > max_videos:
            self.videos = self.videos[:max_videos]

        self._print_stats()

    def _group_by_video(self, directory):
        """Group image files by source video."""
        groups = defaultdict(list)

        subdirs = [d for d in directory.iterdir() if d.is_dir()]

        if subdirs:
            # Layout 1: Subdirectory per video
            for subdir in subdirs:
                video_id = subdir.name
                for f in sorted(subdir.rglob("*")):
                    if f.suffix.lower() in self.EXTENSIONS and f.is_file():
                        groups[video_id].append(f)
        else:
            # Layout 2: Flat — parse video ID from filename
            for f in sorted(directory.iterdir()):
                if not f.is_file() or f.suffix.lower() not in self.EXTENSIONS:
                    continue
                video_id = self._parse_video_id(f.stem)
                groups[video_id].append(f)

        return groups

    @staticmethod
    def _parse_video_id(filename):
        """Extract video ID from filename.

        Handles:
          'video001_frame003'  → 'video001'
          '000_003_frame_42'   → '000_003'
          'abc123_frame012'    → 'abc123'
          'face_00001'         → 'face_00001' (no pattern → own group)
        """
        match = re.match(r'^(.+?)_frame\d+', filename, re.IGNORECASE)
        if match:
            return match.group(1)
        match = re.match(r'^(.+?)_f\d+', filename, re.IGNORECASE)
        if match:
            return match.group(1)
        return filename

    def _print_stats(self):
        """Print video dataset composition."""
        real = sum(1 for _, l, _ in self.videos if l == 0)
        fake = sum(1 for _, l, _ in self.videos if l == 1)
        total = len(self.videos)
        if total == 0:
            print("  [WARN] No videos loaded!")
            return

        frame_counts = [len(f) for f, _, _ in self.videos]
        print(f"  Videos: {total} | Real: {real} | Fake: {fake}")
        print(f"  Frames/video: min={min(frame_counts)}, avg={np.mean(frame_counts):.0f}, "
              f"max={max(frame_counts)} | Sampling: {self.num_frames}")

        ds_counts = defaultdict(int)
        for _, _, ds in self.videos:
            ds_counts[ds] += 1
        for ds, count in sorted(ds_counts.items()):
            print(f"    {ds}: {count} videos")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        frame_paths, label, ds_name = self.videos[idx]

        n_available = len(frame_paths)
        if n_available >= self.num_frames:
            sample_indices = np.linspace(0, n_available - 1, self.num_frames, dtype=int)
        else:
            # Use all available + repeat last frame for padding
            sample_indices = list(range(n_available))
            sample_indices += [n_available - 1] * (self.num_frames - n_available)

        frames = []
        for i in sample_indices:
            img = Image.open(str(frame_paths[i])).convert("RGB")
            img = np.array(img)

            if self.transform:
                augmented = self.transform(image=img)
                img = augmented["image"]
            else:
                img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

            frames.append(img)

        frames = torch.stack(frames)  # (T, 3, H, W)

        # Padding mask for temporal transformer: True = PADDED (ignore)
        # +1 because temporal transformer prepends a CLS token
        mask = torch.zeros(self.num_frames + 1, dtype=torch.bool)
        if n_available < self.num_frames:
            mask[n_available + 1:] = True  # offset by 1 for CLS token

        return {
            "frames": frames,
            "label": torch.tensor(label, dtype=torch.float32),
            "mask": mask,
            "dataset": ds_name,
        }

    def get_balanced_sampler(self):
        """Weighted sampler balancing class and dataset source."""
        labels = [v[1] for v in self.videos]
        datasets = [v[2] for v in self.videos]

        combo_counts = defaultdict(int)
        for l, d in zip(labels, datasets):
            combo_counts[(l, d)] += 1

        weights = [1.0 / combo_counts[(l, d)] for l, d in zip(labels, datasets)]
        return WeightedRandomSampler(weights, len(weights))


# ─────────────────────────────────────────────────────────────
# Splitting Utilities
# ─────────────────────────────────────────────────────────────

class _SubsetWithTransform(Dataset):
    """Wraps a list of raw samples with a specific transform.
    Used internally to give train/val/test different augmentations."""

    def __init__(self, samples, transform, mode="image", num_frames=16):
        self.samples = samples
        self.transform = transform
        self.mode = mode
        self.num_frames = num_frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.mode == "image":
            return self._get_image(idx)
        else:
            return self._get_video(idx)

    def _get_image(self, idx):
        img_path, label, ds_name = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)["image"]
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.float32),
            "dataset": ds_name,
        }

    def _get_video(self, idx):
        frame_paths, label, ds_name = self.samples[idx]
        n_available = len(frame_paths)

        if n_available >= self.num_frames:
            sample_indices = np.linspace(0, n_available - 1, self.num_frames, dtype=int)
        else:
            sample_indices = list(range(n_available))
            sample_indices += [n_available - 1] * (self.num_frames - n_available)

        frames = []
        for i in sample_indices:
            img = Image.open(str(frame_paths[i])).convert("RGB")
            img = np.array(img)
            if self.transform:
                img = self.transform(image=img)["image"]
            else:
                img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            frames.append(img)

        frames = torch.stack(frames)
        mask = torch.zeros(self.num_frames + 1, dtype=torch.bool)
        if n_available < self.num_frames:
            mask[n_available + 1:] = True

        return {
            "frames": frames,
            "label": torch.tensor(label, dtype=torch.float32),
            "mask": mask,
            "dataset": ds_name,
        }

    def get_balanced_sampler(self):
        if self.mode == "image":
            labels = [s[1] for s in self.samples]
            datasets = [s[2] for s in self.samples]
        else:
            labels = [s[1] for s in self.samples]
            datasets = [s[2] for s in self.samples]

        combo_counts = defaultdict(int)
        for l, d in zip(labels, datasets):
            combo_counts[(l, d)] += 1
        weights = [1.0 / combo_counts[(l, d)] for l, d in zip(labels, datasets)]
        return WeightedRandomSampler(weights, len(weights))


def create_dataloaders(data_root, batch_size=32, num_workers=2,
                       train_transform=None, val_transform=None,
                       datasets=None, val_ratio=0.15, test_ratio=0.10,
                       video_mode=False, num_frames=16,
                       max_samples=None, seed=42):
    """
    Create properly split train/val/test dataloaders.

    Key fixes over the original:
    - Data is scanned ONCE, then split by index
    - Train gets train_transform, val/test get val_transform
    - Balanced sampling for training

    Args:
        data_root: root data directory
        batch_size: batch size
        num_workers: dataloader workers
        train_transform: augmentation for training
        val_transform: minimal transform for validation/test
        datasets: which datasets to include (e.g., ["ff++", "dfdc"])
        val_ratio: validation split ratio
        test_ratio: test split ratio
        video_mode: if True, load frame sequences instead of single images
        num_frames: frames per video (only used if video_mode=True)
        max_samples: limit total samples (for debugging)
        seed: random seed for reproducible splits

    Returns:
        train_loader, val_loader, test_loader
    """
    # Step 1: Load all samples with NO transform (just collect paths + labels)
    if video_mode:
        base = VideoDeepfakeDataset(
            data_root, datasets=datasets, num_frames=num_frames,
            transform=None, max_videos=max_samples,
        )
        all_samples = list(base.videos)
        mode = "video"
    else:
        base = DeepfakeDataset(
            data_root, datasets=datasets,
            transform=None, max_samples=max_samples,
        )
        all_samples = list(base.samples)
        mode = "image"

    if len(all_samples) == 0:
        raise ValueError(f"No samples found in {data_root} for datasets {datasets}. "
                         "Check your folder structure.")

    # Step 2: Split indices
    n = len(all_samples)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)

    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]

    # Step 3: Create subset datasets with SEPARATE transforms
    train_samples = [all_samples[i] for i in train_indices]
    val_samples = [all_samples[i] for i in val_indices]
    test_samples = [all_samples[i] for i in test_indices]

    train_ds = _SubsetWithTransform(train_samples, train_transform, mode, num_frames)
    val_ds = _SubsetWithTransform(val_samples, val_transform, mode, num_frames)
    test_ds = _SubsetWithTransform(test_samples, val_transform, mode, num_frames)

    print(f"\n  Split: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

    # Step 4: Create dataloaders
    train_sampler = train_ds.get_balanced_sampler()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
