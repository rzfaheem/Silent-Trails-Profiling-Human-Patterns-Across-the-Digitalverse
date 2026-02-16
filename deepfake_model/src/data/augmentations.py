"""
Augmentation Pipeline for deepfake detection training.

Designed to simulate real-world degradation (social media compression,
screenshots, resizing) to build robustness AND prevent shortcut learning.

Quality pseudo-labels are generated from augmentation parameters for
the Quality Estimator auxiliary loss.
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(face_size=256):
    """
    Heavy augmentation pipeline for training.

    Simulates: JPEG compression, blur, noise, resize, color jitter,
    horizontal flip, and pixel dropout.
    """
    return A.Compose([
        A.Resize(face_size, face_size),

        # Spatial transforms
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=10, p=0.3),

        # Color transforms
        A.ColorJitter(brightness=0.3, contrast=0.3,
                      saturation=0.3, hue=0.1, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.3),

        # Degradation (simulating social media)
        A.OneOf([
            A.ImageCompression(quality_lower=30, quality_upper=95, p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.GaussNoise(var_limit=(10, 50), p=1.0),
        ], p=0.6),

        # Resize down and back up (simulating screenshots/reposts)
        A.OneOf([
            A.Downscale(scale_min=0.5, scale_max=0.9, p=1.0),
        ], p=0.3),

        # Pixel-level
        A.CoarseDropout(max_holes=6, max_height=16, max_width=16,
                        fill_value=0, p=0.2),

        # Normalize and convert
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(face_size=256):
    """Minimal transforms for validation — just resize + normalize."""
    return A.Compose([
        A.Resize(face_size, face_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_chaos_transforms(face_size=256):
    """
    Extreme augmentation for chaos training phase (epochs 19-25).

    Applies multiple degradations simultaneously to build maximum robustness.
    """
    return A.Compose([
        A.Resize(face_size, face_size),

        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                           rotate_limit=15, p=0.5),

        # Aggressive color transforms
        A.ColorJitter(brightness=0.4, contrast=0.4,
                      saturation=0.4, hue=0.15, p=0.7),

        # Stack multiple degradations
        A.ImageCompression(quality_lower=20, quality_upper=80, p=0.5),
        A.GaussianBlur(blur_limit=(3, 9), p=0.4),
        A.GaussNoise(var_limit=(15, 60), p=0.4),

        # Heavy resize simulation
        A.Downscale(scale_min=0.3, scale_max=0.8, p=0.4),

        # Pixel noise
        A.CoarseDropout(max_holes=10, max_height=20, max_width=20,
                        fill_value=0, p=0.3),

        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_anti_shortcut_transforms(face_size=256):
    """
    Transforms specifically designed to prevent shortcut learning.

    Applied BEFORE regular augmentation to equalize inputs:
    - Fixed output resolution (no resolution-based shortcuts)
    - Random recompression (no compression-level shortcuts)
    - Color normalization (no color-space shortcuts)
    """
    return A.Compose([
        # Equalize resolution
        A.Resize(face_size, face_size),

        # Random recompression — destroys dataset-specific JPEG artifacts
        A.ImageCompression(quality_lower=60, quality_upper=95, p=0.8),

        # Color normalization
        A.CLAHE(clip_limit=2.0, p=0.3),

        # No normalize/tensor here — applied after main augmentation
    ])
