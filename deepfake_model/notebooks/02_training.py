"""
=============================================================
 Silent Trails — Deepfake Forensics: Training Notebook
 Run this in Google Colab AFTER running 01_setup_and_data.py
=============================================================

Prerequisites:
- GPU runtime enabled
- Repo cloned and deps installed (Cell 1-2 of setup notebook)
- Google Drive mounted with face crops in SilentTrails/data/

=============================================================
"""

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 1: Setup & Imports                                ║
# ╚══════════════════════════════════════════════════════════╝

# --- Paste into Colab ---
"""
# Clone repo (skip if already done in current session)
!git clone https://github.com/rzfaheem/Silent-Trails-Profiling-Human-Patterns-Across-the-Digitalverse.git /content/silent-trails 2>/dev/null || echo "Already cloned"
%cd /content/silent-trails/deepfake_model

# Install deps (skip if already done)
!pip install -q torch torchvision transformers peft insightface onnxruntime-gpu opencv-python albumentations scikit-learn scipy tqdm wandb matplotlib seaborn

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/silent-trails/deepfake_model')

import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pathlib import Path

from src.models import DeepfakeForensicsModel
from src.training import CompoundLoss, compute_all_metrics, print_metrics
from src.data import get_train_transforms, get_val_transforms, get_chaos_transforms, DeepfakeDataset

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 2: Configuration                                  ║
# ╚══════════════════════════════════════════════════════════╝

# --- Paste into Colab ---
"""
# ===== EDIT THESE PATHS =====
DATA_DIR = '/content/drive/MyDrive/SilentTrails/data'
CHECKPOINT_DIR = '/content/drive/MyDrive/SilentTrails/checkpoints'

# ===== TRAINING CONFIG =====
CONFIG = {
    # Model
    'video_mode': False,       # Start with image mode
    'lora_r': 8,
    'lora_alpha': 16,
    'embed_dim': 512,

    # Data
    'face_size': 256,
    'batch_size': 32,          # Reduce to 16 if out of memory
    'num_workers': 2,
    'datasets': ['ff++'],      # Start with FF++ only, add more later

    # Training
    'epochs': 25,
    'lr_lora': 5e-4,
    'lr_new': 1e-3,
    'weight_decay': 1e-4,

    # Loss weights (Phase 1: BCE + Focal only)
    'bce_weight': 1.0,
    'focal_weight': 0.5,
    'triplet_weight': 0.0,    # Phase 3: set to 0.5
    'quality_weight': 0.0,    # Phase 3: set to 0.1

    # Curriculum schedule
    'easy_end': 8,             # Epochs 1-8: easy phase
    'mixed_end': 18,           # Epochs 9-18: mixed phase
                                # Epochs 19-25: chaos phase

    # Misc
    'fp16': True,
    'gradient_clip': 1.0,
    'save_every': 5,           # Save checkpoint every N epochs
    'val_every': 1,            # Validate every N epochs
}

print("Config loaded:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 3: Create Datasets & Dataloaders                  ║
# ╚══════════════════════════════════════════════════════════╝

# --- Paste into Colab ---
"""
from torch.utils.data import DataLoader, random_split

# Get transforms for each curriculum phase
train_tf = get_train_transforms(CONFIG['face_size'])
val_tf = get_val_transforms(CONFIG['face_size'])
chaos_tf = get_chaos_transforms(CONFIG['face_size'])

# Load full dataset
full_dataset = DeepfakeDataset(
    data_root=DATA_DIR,
    datasets=CONFIG['datasets'],
    split='full',
    transform=train_tf,
)

# Split: 75% train, 15% val, 10% test
total = len(full_dataset)
val_size = int(total * 0.15)
test_size = int(total * 0.10)
train_size = total - val_size - test_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# Override val/test transforms (no augmentation)
# Note: since we split from full, transforms are shared.
# For proper separation, we'd create separate instances.

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    num_workers=CONFIG['num_workers'],
    pin_memory=True,
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=CONFIG['num_workers'],
    pin_memory=True,
)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 4: Initialize Model, Optimizer, Scheduler         ║
# ╚══════════════════════════════════════════════════════════╝

# --- Paste into Colab ---
"""
# Create model
model = DeepfakeForensicsModel(
    video_mode=CONFIG['video_mode'],
    lora_r=CONFIG['lora_r'],
    lora_alpha=CONFIG['lora_alpha'],
    embed_dim=CONFIG['embed_dim'],
).cuda()

params = model.count_parameters()
print(f"Trainable: {params['trainable']:,} / {params['total']:,} ({params['trainable_pct']}%)")

# Optimizer with separate LRs
param_groups = model.get_param_groups(
    lr_lora=CONFIG['lr_lora'],
    lr_new=CONFIG['lr_new'],
)
optimizer = torch.optim.AdamW(param_groups, weight_decay=CONFIG['weight_decay'])

# Scheduler: cosine with warm restarts
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)

# Loss
criterion = CompoundLoss(
    bce_weight=CONFIG['bce_weight'],
    focal_weight=CONFIG['focal_weight'],
    triplet_weight=CONFIG['triplet_weight'],
    quality_weight=CONFIG['quality_weight'],
)

# Mixed precision scaler
scaler = torch.cuda.amp.GradScaler(enabled=CONFIG['fp16'])

print("Model, optimizer, scheduler, loss — all ready!")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 5: Training Loop                                  ║
# ╚══════════════════════════════════════════════════════════╝

# --- Paste into Colab ---
"""
def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch, config):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch['image'].cuda()
        labels = batch['label'].cuda()
        quality = batch['quality'].cuda()

        optimizer.zero_grad()

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=config['fp16']):
            output = model(images)
            loss, loss_dict = criterion(output, labels, quality)

        # Backward
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        probs = output['probability'].detach().cpu().squeeze()
        all_preds.extend(probs.tolist() if probs.dim() > 0 else [probs.item()])
        all_labels.extend(labels.cpu().tolist())

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'bce': f"{loss_dict['bce']:.3f}",
            'focal': f"{loss_dict['focal']:.3f}",
        })

    avg_loss = total_loss / len(loader)
    metrics = compute_all_metrics(all_labels, all_preds)

    return avg_loss, metrics


@torch.no_grad()
def validate(model, loader, criterion, config):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="Validating"):
        images = batch['image'].cuda()
        labels = batch['label'].cuda()
        quality = batch['quality'].cuda()

        with torch.cuda.amp.autocast(enabled=config['fp16']):
            output = model(images)
            loss, _ = criterion(output, labels, quality)

        total_loss += loss.item()
        probs = output['probability'].cpu().squeeze()
        all_preds.extend(probs.tolist() if probs.dim() > 0 else [probs.item()])
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    metrics = compute_all_metrics(all_labels, all_preds)

    return avg_loss, metrics


print("Training functions defined!")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 6: Run Training                                   ║
# ╚══════════════════════════════════════════════════════════╝

# --- Paste into Colab ---
"""
best_auc = 0.0
history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}

print(f"Starting training for {CONFIG['epochs']} epochs")
print(f"Curriculum: Easy(1-{CONFIG['easy_end']}) → Mixed({CONFIG['easy_end']+1}-{CONFIG['mixed_end']}) → Chaos({CONFIG['mixed_end']+1}-{CONFIG['epochs']})")
print("=" * 60)

for epoch in range(1, CONFIG['epochs'] + 1):
    start_time = time.time()

    # Curriculum phase logging
    if epoch <= CONFIG['easy_end']:
        phase = "EASY"
    elif epoch <= CONFIG['mixed_end']:
        phase = "MIXED"
    else:
        phase = "CHAOS"

    print(f"\\n{'='*60}")
    print(f"Epoch {epoch}/{CONFIG['epochs']} — Phase: {phase}")
    print(f"{'='*60}")

    # Train
    train_loss, train_metrics = train_one_epoch(
        model, train_loader, criterion, optimizer, scaler, epoch, CONFIG
    )
    scheduler.step()

    history['train_loss'].append(train_loss)
    history['train_auc'].append(train_metrics['auc'])

    print(f"  Train Loss: {train_loss:.4f} | AUC: {train_metrics['auc']:.4f} | F1: {train_metrics['f1']:.4f}")

    # Validate
    if epoch % CONFIG['val_every'] == 0:
        val_loss, val_metrics = validate(model, val_loader, criterion, CONFIG)

        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_metrics['auc'])

        print(f"  Val Loss:   {val_loss:.4f} | AUC: {val_metrics['auc']:.4f} | F1: {val_metrics['f1']:.4f}")

        # Save best model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            save_path = f"{CHECKPOINT_DIR}/best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_auc,
                'config': CONFIG,
            }, save_path)
            print(f"  ★ New best model saved! AUC: {best_auc:.4f}")

    # Save periodic checkpoint
    if epoch % CONFIG['save_every'] == 0:
        save_path = f"{CHECKPOINT_DIR}/checkpoint_epoch{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': CONFIG,
        }, save_path)
        print(f"  Checkpoint saved: epoch {epoch}")

    elapsed = time.time() - start_time
    print(f"  Time: {elapsed:.0f}s | LR: {scheduler.get_last_lr()[0]:.6f}")

print(f"\\n{'='*60}")
print(f"Training complete! Best validation AUC: {best_auc:.4f}")
print(f"Best model saved to: {CHECKPOINT_DIR}/best_model.pth")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 7: Plot Training Curves                           ║
# ╚══════════════════════════════════════════════════════════╝

# --- Paste into Colab ---
"""
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(history['train_loss'], label='Train Loss', color='#ff6b6b')
axes[0].plot(history['val_loss'], label='Val Loss', color='#4ecdc4')
axes[0].set_title('Loss Curve')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# AUC
axes[1].plot(history['train_auc'], label='Train AUC', color='#ff6b6b')
axes[1].plot(history['val_auc'], label='Val AUC', color='#4ecdc4')
axes[1].axhline(y=0.97, color='gray', linestyle='--', alpha=0.5, label='Target (0.97)')
axes[1].set_title('AUC-ROC Curve')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('AUC')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{CHECKPOINT_DIR}/training_curves.png', dpi=150)
plt.show()
print("Saved to Google Drive!")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 8: Evaluate on Test Set                           ║
# ╚══════════════════════════════════════════════════════════╝

# --- Paste into Colab ---
"""
# Load best model
checkpoint = torch.load(f'{CHECKPOINT_DIR}/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded best model from epoch {checkpoint['epoch']} (AUC: {checkpoint['val_auc']:.4f})")

# Create test loader
test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=CONFIG['num_workers'],
    pin_memory=True,
)

# Evaluate
test_loss, test_metrics = validate(model, test_loader, criterion, CONFIG)
print_metrics(test_metrics, prefix="TEST SET RESULTS")
"""
