"""
Silent Trails — Deepfake Forensics Training Script

Two-stage training:
  Stage 1 (Image):  Train frame-level model on face crops
  Stage 2 (Video):  Load Stage 1 weights, fine-tune with temporal module

Usage (run from deepfake_model/ directory):
  # Stage 1 — Image training
  python train.py --stage 1 --data_root /path/to/data --datasets ff++ dfdc --epochs 15

  # Stage 2 — Video training (loads Stage 1 checkpoint)
  python train.py --stage 2 --data_root /path/to/data --datasets dfdc --epochs 10 \
                   --checkpoint /path/to/stage1_best.pth

  # Resume interrupted training
  python train.py --stage 1 --data_root /path/to/data --resume /path/to/last_checkpoint.pth

Features:
  - Automatic checkpoint saving every epoch (survives Colab disconnects)
  - Best model saved based on validation AUC
  - Curriculum training (easy → mixed → chaos augmentation)
  - Mixed precision (FP16) for speed
  - Gradient clipping for stability
  - Balanced sampling across datasets and classes
"""

import os
import sys
import time
import json
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import DeepfakeForensicsModel
from src.training import CompoundLoss, compute_all_metrics, print_metrics
from src.data import (
    DeepfakeDataset, VideoDeepfakeDataset,
    create_dataloaders,
    get_val_transforms,
    compose_training_transforms, compose_chaos_transforms,
)


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_curriculum_transform(epoch, config):
    """Get augmentation transform based on curriculum phase.

    All phases include anti-shortcut preprocessing (JPEG recompression,
    CLAHE color normalization) applied BEFORE the regular augmentation.
    This prevents the model from learning dataset-specific shortcuts.
    """
    face_size = config['face_size']
    if epoch <= config['easy_end']:
        # Easy + Mixed: anti-shortcut → standard augmentation
        return compose_training_transforms(face_size), "EASY"
    elif epoch <= config['mixed_end']:
        return compose_training_transforms(face_size), "MIXED"
    else:
        # Chaos: anti-shortcut → extreme augmentation
        return compose_chaos_transforms(face_size), "CHAOS"


def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch, config, video_mode=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    batch_count = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=True)
    for batch in pbar:
        if video_mode:
            inputs = batch['frames'].cuda()          # (B, T, 3, H, W)
            mask = batch['mask'].cuda()               # (B, T+1)
        else:
            inputs = batch['image'].cuda()            # (B, 3, H, W)
            mask = None

        labels = batch['label'].cuda()                # (B,)

        optimizer.zero_grad()

        # Forward with mixed precision
        with torch.cuda.amp.autocast(enabled=config['fp16']):
            output = model(inputs, mask=mask)
            loss, loss_dict = criterion(output, labels.long())

        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        batch_count += 1

        probs = output['probability'].detach().cpu().squeeze()
        if probs.dim() == 0:
            all_preds.append(probs.item())
        else:
            all_preds.extend(probs.tolist())
        all_labels.extend(labels.cpu().tolist())

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'bce': f"{loss_dict['bce']:.3f}",
        })

    avg_loss = total_loss / max(batch_count, 1)
    metrics = compute_all_metrics(all_labels, all_preds)
    return avg_loss, metrics


@torch.no_grad()
def validate(model, loader, criterion, config, video_mode=False):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    batch_count = 0

    for batch in tqdm(loader, desc="[Validate]", leave=False):
        if video_mode:
            inputs = batch['frames'].cuda()
            mask = batch['mask'].cuda()
        else:
            inputs = batch['image'].cuda()
            mask = None

        labels = batch['label'].cuda()

        with torch.cuda.amp.autocast(enabled=config['fp16']):
            output = model(inputs, mask=mask)
            loss, _ = criterion(output, labels.long())

        total_loss += loss.item()
        batch_count += 1

        probs = output['probability'].cpu().squeeze()
        if probs.dim() == 0:
            all_preds.append(probs.item())
        else:
            all_preds.extend(probs.tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(batch_count, 1)
    metrics = compute_all_metrics(all_labels, all_preds)
    return avg_loss, metrics


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, config,
                    val_auc, history, save_path):
    """Save a training checkpoint (for resume on disconnect)."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'val_auc': val_auc,
        'config': config,
        'history': history,
    }, save_path)


def load_stage1_weights(model, checkpoint_path):
    """Load Stage 1 (image) weights into a Stage 2 (video) model.

    Loads all weights except the temporal transformer (which is new in Stage 2).
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']

    # Filter out temporal module weights (they don't exist in Stage 1)
    model_dict = model.state_dict()
    pretrained = {k: v for k, v in state_dict.items()
                  if k in model_dict and 'temporal' not in k}

    missing = set(model_dict.keys()) - set(pretrained.keys())
    print(f"\n  Loaded {len(pretrained)} parameter groups from Stage 1")
    print(f"  New parameters (Stage 2): {len(missing)}")
    for key in sorted(missing):
        print(f"    + {key}")

    model_dict.update(pretrained)
    model.load_state_dict(model_dict)

    print(f"  Stage 1 val AUC was: {checkpoint.get('val_auc', 'unknown')}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Deepfake Forensics Training")
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2],
                        help='Stage 1 (image) or Stage 2 (video)')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing dataset folders')
    parser.add_argument('--datasets', nargs='+', default=['ff++'],
                        help='Datasets to train on (e.g., ff++ dfdc)')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: 16 for image, 4 for video)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Stage 1 checkpoint to load for Stage 2')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from this checkpoint')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Frames per video (Stage 2 only)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit samples for debugging')
    parser.add_argument('--lr_lora', type=float, default=5e-4)
    parser.add_argument('--lr_new', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # ── Configuration ──────────────────────────────────────
    video_mode = (args.stage == 2)
    default_batch = 4 if video_mode else 16
    batch_size = args.batch_size or default_batch

    config = {
        'stage': args.stage,
        'video_mode': video_mode,
        'face_size': 256,
        'batch_size': batch_size,
        'num_frames': args.num_frames,
        'epochs': args.epochs,
        'lr_lora': args.lr_lora if args.stage == 1 else args.lr_lora * 0.1,
        'lr_new': args.lr_new if args.stage == 1 else args.lr_new,
        'lr_temporal': args.lr_new,  # Temporal module gets full LR in Stage 2
        'weight_decay': 1e-4,
        'fp16': True,
        'gradient_clip': 1.0,
        'easy_end': min(8, args.epochs // 3),
        'mixed_end': min(18, args.epochs * 2 // 3),
        'bce_weight': 1.0,
        'focal_weight': 0.5,
        'triplet_weight': 0.0,
        'quality_weight': 0.0,
    }

    os.makedirs(args.save_dir, exist_ok=True)
    stage_name = "image" if args.stage == 1 else "video"

    print(f"\n{'='*60}")
    print(f"  DEEPFAKE FORENSICS — Stage {args.stage} ({stage_name.upper()}) Training")
    print(f"{'='*60}")
    print(f"  Data root:  {args.data_root}")
    print(f"  Datasets:   {args.datasets}")
    print(f"  Epochs:     {config['epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Video mode: {config['video_mode']}")
    if video_mode:
        print(f"  Frames/vid: {config['num_frames']}")
    print(f"  Save dir:   {args.save_dir}")
    print(f"{'='*60}\n")

    # ── Data ───────────────────────────────────────────────
    # Training uses anti-shortcut preprocessing → augmentation
    # (JPEG recompression + CLAHE applied BEFORE augmentation
    #  to destroy dataset-specific compression/color fingerprints)
    train_tf = compose_training_transforms(config['face_size'])
    val_tf = get_val_transforms(config['face_size'])

    print("Loading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=config['batch_size'],
        num_workers=args.num_workers,
        train_transform=train_tf,
        val_transform=val_tf,
        datasets=args.datasets,
        video_mode=config['video_mode'],
        num_frames=config['num_frames'],
        max_samples=args.max_samples,
    )

    # ── Model ──────────────────────────────────────────────
    print("\nCreating model...")
    model = DeepfakeForensicsModel(
        video_mode=config['video_mode'],
        lora_r=8,
        lora_alpha=16,
        embed_dim=512,
    )

    # Load Stage 1 weights for Stage 2
    if args.stage == 2 and args.checkpoint:
        print(f"\nLoading Stage 1 weights from: {args.checkpoint}")
        model = load_stage1_weights(model, args.checkpoint)

    model = model.cuda()
    params = model.count_parameters()
    print(f"  Total: {params['total']:,} | Trainable: {params['trainable']:,} "
          f"({params['trainable_pct']}%)")

    # ── Optimizer ──────────────────────────────────────────
    if args.stage == 2:
        # Stage 2: separate LR groups
        # - Frame-level modules: LOW lr (already trained in Stage 1)
        # - Temporal module: FULL lr (learning from scratch)
        param_groups = []
        temporal_params = []
        lora_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'temporal' in name:
                temporal_params.append(param)
            elif 'lora_' in name:
                lora_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {"params": lora_params, "lr": config['lr_lora']},
            {"params": other_params, "lr": config['lr_new'] * 0.1},  # Low LR for pretrained
            {"params": temporal_params, "lr": config['lr_temporal']},  # Full LR for new temporal
        ]
        print(f"  Param groups: {len(lora_params)} LoRA, {len(other_params)} pretrained, "
              f"{len(temporal_params)} temporal")
    else:
        param_groups = model.get_param_groups(
            lr_lora=config['lr_lora'],
            lr_new=config['lr_new'],
        )

    optimizer = torch.optim.AdamW(param_groups, weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(config['epochs'] // 3, 3), T_mult=2
    )
    criterion = CompoundLoss(
        bce_weight=config['bce_weight'],
        focal_weight=config['focal_weight'],
        triplet_weight=config['triplet_weight'],
        quality_weight=config['quality_weight'],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config['fp16'])

    # ── Resume ─────────────────────────────────────────────
    start_epoch = 1
    best_auc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}

    if args.resume:
        print(f"\nResuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location='cuda', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_auc = ckpt.get('val_auc', 0.0)
        history = ckpt.get('history', history)
        print(f"  Resuming from epoch {start_epoch}, best AUC: {best_auc:.4f}")

    # ── Training Loop ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Starting training: epochs {start_epoch} → {config['epochs']}")
    print(f"  Curriculum: Easy(1-{config['easy_end']}) → Mixed({config['easy_end']+1}-"
          f"{config['mixed_end']}) → Chaos({config['mixed_end']+1}-{config['epochs']})")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, config['epochs'] + 1):
        epoch_start = time.time()

        # Curriculum phase
        _, phase = get_curriculum_transform(epoch, config)

        print(f"\n{'─'*50}")
        print(f"  Epoch {epoch}/{config['epochs']}  |  Phase: {phase}  |  "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'─'*50}")

        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            epoch, config, video_mode=config['video_mode']
        )
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_metrics['auc'])

        print(f"  Train — Loss: {train_loss:.4f} | AUC: {train_metrics['auc']:.4f} | "
              f"F1: {train_metrics['f1']:.4f} | Acc: {train_metrics['accuracy']:.4f}")

        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, config,
            video_mode=config['video_mode']
        )
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_metrics['auc'])

        print(f"  Val   — Loss: {val_loss:.4f} | AUC: {val_metrics['auc']:.4f} | "
              f"F1: {val_metrics['f1']:.4f} | Acc: {val_metrics['accuracy']:.4f}")

        # Save best model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_path = os.path.join(args.save_dir, f'stage{args.stage}_best.pth')
            save_checkpoint(model, optimizer, scheduler, scaler,
                            epoch, config, best_auc, history, best_path)
            print(f"  ★ New best! AUC: {best_auc:.4f} → saved to {best_path}")

        # Save latest checkpoint (for resume)
        latest_path = os.path.join(args.save_dir, f'stage{args.stage}_latest.pth')
        save_checkpoint(model, optimizer, scheduler, scaler,
                        epoch, config, val_metrics['auc'], history, latest_path)

        elapsed = time.time() - epoch_start
        print(f"  Time: {elapsed:.0f}s")

    # ── Final Evaluation ───────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training Complete! Best Val AUC: {best_auc:.4f}")
    print(f"{'='*60}")

    # Load best model and evaluate on test set
    best_ckpt = torch.load(os.path.join(args.save_dir, f'stage{args.stage}_best.pth'), weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])

    print("\n  Evaluating on test set...")
    test_loss, test_metrics = validate(
        model, test_loader, criterion, config,
        video_mode=config['video_mode']
    )
    print_metrics(test_metrics, prefix=f"STAGE {args.stage} TEST RESULTS")

    # Save history
    history_path = os.path.join(args.save_dir, f'stage{args.stage}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n  History saved: {history_path}")
    print(f"  Best model:    {os.path.join(args.save_dir, f'stage{args.stage}_best.pth')}")


if __name__ == '__main__':
    main()
