"""
================================================================
 Silent Trails — Complete Training Notebook
 Datasets: FF++ (Kaggle) + HiDF (Drive) + DeepDetect-2025 (Kaggle)
 Cross-eval: CelebDF-v2 (Drive) — NEVER trained on
================================================================

HOW TO USE:
1. Open https://colab.research.google.com → New notebook
2. Runtime → Change runtime type → A100 GPU (Colab Pro)
3. Copy each CELL into a separate Colab cell
4. Run cells TOP TO BOTTOM, one at a time
5. Wait for each cell to finish before running the next
================================================================
"""


# ════════════════════════════════════════════════════════════════
# CELL 1 — CHECK GPU
# Expected output: Should say "A100" or "T4"
# ════════════════════════════════════════════════════════════════
"""
!nvidia-smi
import torch
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
"""


# ════════════════════════════════════════════════════════════════
# CELL 2 — CLONE YOUR CODE FROM GITHUB
# This downloads all your model code (train.py, src/, etc.)
# ════════════════════════════════════════════════════════════════
"""
!git clone https://github.com/rzfaheem/Silent-Trails-Profiling-Human-Patterns-Across-the-Digitalverse.git /content/silent-trails 2>/dev/null || (cd /content/silent-trails && git pull)
%cd /content/silent-trails/deepfake_model
!ls
"""


# ════════════════════════════════════════════════════════════════
# CELL 3 — INSTALL LIBRARIES
# Takes 2-3 minutes. Wait until it finishes fully.
# ════════════════════════════════════════════════════════════════
"""
!pip install -q torch torchvision transformers peft
!pip install -q insightface onnxruntime-gpu
!pip install -q opencv-python albumentations
!pip install -q scikit-learn tqdm matplotlib seaborn
!pip install -q kaggle

import torch
print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
"""


# ════════════════════════════════════════════════════════════════
# CELL 4 — MOUNT GOOGLE DRIVE + CREATE FOLDERS
#
# SPACE STRATEGY (works with 20GB Google Drive):
#   - FF++ and DeepDetect → /content/ (Colab local, 100GB free, temporary)
#   - HiDF and CelebDF   → Google Drive (you upload these manually)
#   - Checkpoints        → Google Drive (CRITICAL — must survive disconnects)
#
# This way your Drive only needs ~5-8 GB, not 20+ GB.
# ════════════════════════════════════════════════════════════════
"""
from google.colab import drive
drive.mount('/content/drive')

import os

# Google Drive — only for things that MUST persist
DRIVE_BASE = '/content/drive/MyDrive/SilentTrails'
CKPTS = f'{DRIVE_BASE}/checkpoints'
DRIVE_DATA = f'{DRIVE_BASE}/data'

# Create Drive folders for HiDF, CelebDF, and checkpoints
for ds in ['hidf', 'celebdf']:
    os.makedirs(f'{DRIVE_DATA}/{ds}/real', exist_ok=True)
    os.makedirs(f'{DRIVE_DATA}/{ds}/fake', exist_ok=True)
os.makedirs(CKPTS, exist_ok=True)

# Colab local storage — for large datasets (FF++ and DeepDetect)
# These are temporary (lost on disconnect) but we re-download from Kaggle
LOCAL_DATA = '/content/data'
for ds in ['ff++', 'deepdetect']:
    os.makedirs(f'{LOCAL_DATA}/{ds}/real', exist_ok=True)
    os.makedirs(f'{LOCAL_DATA}/{ds}/fake', exist_ok=True)

# Combined data root that train.py will use
# We create symlinks so train.py sees everything under one /content/data/ folder
os.symlink(f'{DRIVE_DATA}/hidf',   f'{LOCAL_DATA}/hidf',   target_is_directory=True)
os.symlink(f'{DRIVE_DATA}/celebdf', f'{LOCAL_DATA}/celebdf', target_is_directory=True)

DATA = LOCAL_DATA  # This is what we pass to train.py

print("Setup complete!")
print(f"Large datasets (FF++, DeepDetect) → {LOCAL_DATA}  (Colab local, free)")
print(f"HiDF + CelebDF                    → {DRIVE_DATA}  (Google Drive)")
print(f"Checkpoints                        → {CKPTS}  (Google Drive, persists)")
"""


# ════════════════════════════════════════════════════════════════
# CELL 5 — SETUP KAGGLE API (New Token Format)
#
# Kaggle now uses a token (KGAT_...) instead of the old JSON file.
#
# HOW TO GET YOUR TOKEN:
#   1. Go to https://www.kaggle.com/settings
#   2. Scroll to "API" section
#   3. Click "Create New Token"
#   4. Copy the token shown (starts with KGAT_...)
#   5. Paste it below where it says YOUR_TOKEN_HERE
# ════════════════════════════════════════════════════════════════
"""
import os, json

# ⚠️ PASTE YOUR KAGGLE TOKEN HERE (the KGAT_... value)
KAGGLE_TOKEN = "YOUR_TOKEN_HERE"   # e.g. "KGAT_a072bd400cecd5b7e8353bfa309526e6"

# --- DO NOT EDIT BELOW THIS LINE ---
if KAGGLE_TOKEN == "YOUR_TOKEN_HERE":
    print("ERROR: You forgot to paste your Kaggle token above!")
    print("Replace YOUR_TOKEN_HERE with your actual KGAT_... token")
else:
    os.makedirs('/root/.kaggle', exist_ok=True)
    # New Kaggle API uses token format
    kaggle_config = {"token": KAGGLE_TOKEN}
    with open('/root/.kaggle/kaggle.json', 'w') as f:
        json.dump(kaggle_config, f)
    os.chmod('/root/.kaggle/kaggle.json', 0o600)

    # Also set as environment variable (belt and suspenders)
    os.environ['KAGGLE_TOKEN'] = KAGGLE_TOKEN

    # Verify it works
    result = !kaggle datasets list --search "faceforensics" 2>&1
    print('\n'.join(result[:5]))
    print("\nKaggle API is ready!")
"""


# ════════════════════════════════════════════════════════════════
# CELL 6 — DOWNLOAD FF++ DATASET FROM KAGGLE
#
# This downloads 14.36 GB directly into Colab.
# Takes ~10-15 minutes depending on network speed.
# You do NOT need to manually download anything.
#
# Dataset: adhamelmy/faceforencis-extracted-frames
# Structure after download:
#   /content/ffpp/fake/Deepfakes/*.png
#   /content/ffpp/fake/Face2Face/*.png
#   /content/ffpp/fake/FaceSwap/*.png
#   /content/ffpp/fake/NeuralTextures/*.png
#   /content/ffpp/fake/FaceShifter/*.png
#   /content/ffpp/real/*.png
# ════════════════════════════════════════════════════════════════
"""
import os

# Downloads to Colab local storage (NOT Google Drive)
# Colab has ~100GB local free — no Drive space used
FFPP_DOWNLOAD = '/content/data/ff++_raw'
os.makedirs(FFPP_DOWNLOAD, exist_ok=True)

print("Downloading FF++ from Kaggle (14.36 GB) to Colab local storage...")
print("This takes 10-15 minutes. Do not interrupt.")
!kaggle datasets download -d adhamelmy/faceforencis-extracted-frames -p {FFPP_DOWNLOAD} --unzip

# Show what we got
!echo "Downloaded files:" && ls {FFPP_DOWNLOAD}
!echo "Fake subdirectories:" && ls {FFPP_DOWNLOAD}/fake/
!echo "Real count:" && ls {FFPP_DOWNLOAD}/real/ | wc -l
!echo "Fake count (all subdirs):" && find {FFPP_DOWNLOAD}/fake -name "*.png" | wc -l
"""


# ════════════════════════════════════════════════════════════════
# CELL 7 — ORGANIZE FF++ INTO TRAINING FOLDERS
#
# Our model expects: data/ff++/real/ and data/ff++/fake/
# The fake/ already has subdirectories (Deepfakes, Face2Face, etc.)
# Our dataset loader uses rglob() so it finds ALL images recursively.
# We just need to symlink or copy the root folders.
# ════════════════════════════════════════════════════════════════
"""
import os, shutil

# FF++ already downloaded to /content/data/ff++_raw
# We just move it into the right place under /content/data/ff++/
import os
FFPP_RAW  = '/content/data/ff++_raw'
FFPP_REAL = '/content/data/ff++/real'
FFPP_FAKE = '/content/data/ff++/fake'
os.makedirs(FFPP_REAL, exist_ok=True)
os.makedirs(FFPP_FAKE, exist_ok=True)

# Move real frames (no copy — same disk, instant)
print("Moving real frames...")
!mv {FFPP_RAW}/real/* {FFPP_REAL}/ 2>/dev/null || cp -r {FFPP_RAW}/real/. {FFPP_REAL}/

# Move fake frames (keeps subdirectory structure)
print("Moving fake frames (Deepfakes, Face2Face, FaceSwap, etc.)...")
!mv {FFPP_RAW}/fake/* {FFPP_FAKE}/ 2>/dev/null || cp -r {FFPP_RAW}/fake/. {FFPP_FAKE}/

from pathlib import Path
real_count = sum(1 for f in Path(FFPP_REAL).rglob('*') if f.is_file())
fake_count = sum(1 for f in Path(FFPP_FAKE).rglob('*') if f.is_file())
print(f"FF++ ready: {real_count:,} real + {fake_count:,} fake (all in /content/, NO Drive space used)")
"""


# ════════════════════════════════════════════════════════════════
# CELL 8 — DOWNLOAD DEEPDETECT-2025 FROM KAGGLE
#
# Covers DALL-E 3, Stable Diffusion 3, Midjourney, StyleGAN3
# These are AI-generated FACE images (not face swaps)
# This teaches our frequency stream to detect diffusion artifacts
#
# NOTE: Search for the exact dataset slug on Kaggle first.
# Go to kaggle.com → search "deepdetect 2025 deepfake"
# Copy the dataset slug (username/dataset-name) and paste below.
# ════════════════════════════════════════════════════════════════
"""
import os

# Also goes to Colab local /content/ — no Drive space used
DD_DOWNLOAD = '/content/data/deepdetect_raw'
os.makedirs(DD_DOWNLOAD, exist_ok=True)

# ⚠️ REPLACE THIS with the exact Kaggle dataset slug you find
# Search on kaggle.com for "AI generated faces deepfake 2025"
DEEPDETECT_SLUG = 'YOUR_USERNAME/deepdetect-2025'

print(f"Downloading DeepDetect-2025: {DEEPDETECT_SLUG}")
!kaggle datasets download -d {DEEPDETECT_SLUG} -p {DD_DOWNLOAD} --unzip

# Inspect structure so you know what folders exist
!ls {DD_DOWNLOAD}
"""


# ════════════════════════════════════════════════════════════════
# CELL 9 — ORGANIZE DEEPDETECT INTO TRAINING FOLDERS
#
# After downloading, check the structure with the cell above.
# Adjust the paths below based on what you see.
# ════════════════════════════════════════════════════════════════
"""
import os
from pathlib import Path

# DeepDetect goes to /content/data/deepdetect/ (local, no Drive space)
DD_RAW  = '/content/data/deepdetect_raw'
DD_REAL = '/content/data/deepdetect/real'
DD_FAKE = '/content/data/deepdetect/fake'
os.makedirs(DD_REAL, exist_ok=True)
os.makedirs(DD_FAKE, exist_ok=True)

# Show structure so you know what folders exist
for p in sorted(Path(DD_RAW).iterdir()):
    print(p)

# ⚠️ Adjust these paths based on what you see above
DD_REAL_SRC = f'{DD_RAW}/real'   # Change if folder has different name
DD_FAKE_SRC = f'{DD_RAW}/fake'   # Change if folder has different name

!mv {DD_REAL_SRC}/* {DD_REAL}/ 2>/dev/null || cp -r {DD_REAL_SRC}/. {DD_REAL}/
!mv {DD_FAKE_SRC}/* {DD_FAKE}/ 2>/dev/null || cp -r {DD_FAKE_SRC}/. {DD_FAKE}/

real_count = sum(1 for f in Path(DD_REAL).rglob('*') if f.is_file())
fake_count = sum(1 for f in Path(DD_FAKE).rglob('*') if f.is_file())
print(f"DeepDetect ready: {real_count:,} real + {fake_count:,} fake (NO Drive space used)")
"""


# ════════════════════════════════════════════════════════════════
# CELL 10 — HIDF DATASET
#
# HiDF is already on your PC. You need to upload it to Google Drive.
# DO THIS MANUALLY before running this cell:
#   1. Open Google Drive in your browser
#   2. Navigate to: SilentTrails/data/
#   3. Upload your HiDF folder with this structure:
#      hidf/
#      ├── real/   ← real images
#      └── fake/   ← fake/AI-generated images
#
# Then run this cell to verify it's there:
# ════════════════════════════════════════════════════════════════
"""
from pathlib import Path

DATA = '/content/drive/MyDrive/SilentTrails/data'

real_count = sum(1 for f in Path(f'{DATA}/hidf/real').rglob('*') if f.is_file())
fake_count = sum(1 for f in Path(f'{DATA}/hidf/fake').rglob('*') if f.is_file())

if real_count + fake_count == 0:
    print("HiDF NOT FOUND. Please upload it to Google Drive first.")
    print(f"Expected path: {DATA}/hidf/")
else:
    print(f"HiDF ready: {real_count:,} real + {fake_count:,} fake")
"""


# ════════════════════════════════════════════════════════════════
# CELL 11 — CELEBDF-V2 (Cross-eval dataset)
#
# Upload CelebDF to Google Drive manually:
#   SilentTrails/data/celebdf/real/
#   SilentTrails/data/celebdf/fake/
#
# ⚠️ NEVER include 'celebdf' in TRAIN_DATASETS
# ════════════════════════════════════════════════════════════════
"""
from pathlib import Path

DATA = '/content/drive/MyDrive/SilentTrails/data'

real_count = sum(1 for f in Path(f'{DATA}/celebdf/real').rglob('*') if f.is_file())
fake_count = sum(1 for f in Path(f'{DATA}/celebdf/fake').rglob('*') if f.is_file())

if real_count + fake_count == 0:
    print("CelebDF NOT FOUND. Upload it to Google Drive.")
else:
    print(f"CelebDF ready: {real_count:,} real + {fake_count:,} fake (cross-eval ONLY)")
"""


# ════════════════════════════════════════════════════════════════
# CELL 12 — CHECK ALL DATA
# Run this to see a summary of everything before training
# ════════════════════════════════════════════════════════════════
"""
import os
from pathlib import Path

DATA = '/content/drive/MyDrive/SilentTrails/data'
CKPTS = '/content/drive/MyDrive/SilentTrails/checkpoints'

print("=" * 55)
print("  DATASET SUMMARY")
print("=" * 55)

total = 0
ready = []
for ds in ['ff++', 'hidf', 'deepdetect', 'celebdf']:
    r = sum(1 for f in Path(f'{DATA}/{ds}/real').rglob('*') if f.is_file()) if os.path.exists(f'{DATA}/{ds}/real') else 0
    fk = sum(1 for f in Path(f'{DATA}/{ds}/fake').rglob('*') if f.is_file()) if os.path.exists(f'{DATA}/{ds}/fake') else 0
    flag = '✅' if (r + fk) > 0 else '❌'
    note = '(cross-eval only)' if ds == 'celebdf' else ''
    print(f"  {flag} {ds:12s}: {r:6,} real + {fk:6,} fake  {note}")
    if r + fk > 0 and ds != 'celebdf':
        ready.append(ds)
    total += r + fk

print(f"\n  Total training images: {total:,}")
print(f"  Datasets ready to train: {ready}")
print("=" * 55)
"""


# ════════════════════════════════════════════════════════════════
# CELL 13 — VERIFY MODEL LOADS
# This downloads DINOv2 (~350MB) on first run. Wait for it.
# If output shows model parameter counts, you're good.
# ════════════════════════════════════════════════════════════════
"""
import sys
sys.path.insert(0, '/content/silent-trails/deepfake_model')

import torch
from src.models import DeepfakeForensicsModel

print("Loading model (downloads DINOv2 ~350MB on first run)...")
model = DeepfakeForensicsModel(video_mode=False).cuda()
p = model.count_parameters()
print(f"Total:     {p['total']:,}")
print(f"Trainable: {p['trainable']:,} ({p['trainable_pct']}%)")
print(f"Frozen:    {p['frozen']:,} (DINOv2 backbone)")

# Quick forward pass test
dummy = torch.randn(2, 3, 256, 256).cuda()
with torch.no_grad():
    out = model(dummy)
print(f"Output shapes: logits={out['logits'].shape}, prob={out['probability'].shape}")
print("Model OK!")

del model, dummy
torch.cuda.empty_cache()
"""


# ════════════════════════════════════════════════════════════════
# CELL 14 — SMOKE TEST (5 minutes)
#
# ALWAYS run this before the real training.
# It checks that data loads + model trains + checkpoint saves.
# If this fails, fix the error before doing the real 15-epoch run.
# ════════════════════════════════════════════════════════════════
"""
%cd /content/silent-trails/deepfake_model

DATA  = '/content/drive/MyDrive/SilentTrails/data'
CKPTS = '/content/drive/MyDrive/SilentTrails/checkpoints'

# Only include datasets that showed ✅ in Cell 12
# Remove any dataset that had 0 images
DATASETS = 'ff++'  # Start with just FF++ for smoke test

!python train.py \
    --stage 1 \
    --data_root "{DATA}" \
    --datasets {DATASETS} \
    --epochs 2 \
    --batch_size 8 \
    --max_samples 500 \
    --save_dir "{CKPTS}" \
    --num_workers 2

print("If you see 'Training Complete' above, smoke test PASSED. Proceed to Cell 15.")
print("If you see an error, DO NOT proceed. Share the error message.")
"""


# ════════════════════════════════════════════════════════════════
# CELL 15 — STAGE 1: IMAGE TRAINING (THE MAIN TRAINING)
#
# Expected time: 3-5 hours on A100
# Let this run. You can close the browser — it keeps running.
# Checkpoints save to Google Drive every epoch automatically.
# If Colab disconnects, use Cell 15b to resume.
#
# Expected val AUC progress:
#   Epoch 5:  ~0.70-0.80
#   Epoch 10: ~0.82-0.88
#   Epoch 15: ~0.85-0.92 ← target
# ════════════════════════════════════════════════════════════════
"""
%cd /content/silent-trails/deepfake_model

DATA  = '/content/drive/MyDrive/SilentTrails/data'
CKPTS = '/content/drive/MyDrive/SilentTrails/checkpoints'

# Include all datasets that have data (from Cell 12)
# Do NOT include 'celebdf'
TRAIN_DATASETS = 'ff++ hidf deepdetect'  # Remove any that have 0 images

EPOCHS     = 15
BATCH_SIZE = 16  # Reduce to 8 if you get "CUDA out of memory" error

!python train.py \
    --stage 1 \
    --data_root "{DATA}" \
    --datasets {TRAIN_DATASETS} \
    --epochs {EPOCHS} \
    --batch_size {BATCH_SIZE} \
    --save_dir "{CKPTS}" \
    --num_workers 2
"""


# ════════════════════════════════════════════════════════════════
# CELL 15b — RESUME STAGE 1 (only if Colab disconnected)
# Run this INSTEAD of Cell 15 if training was interrupted
# ════════════════════════════════════════════════════════════════
"""
%cd /content/silent-trails/deepfake_model

DATA  = '/content/drive/MyDrive/SilentTrails/data'
CKPTS = '/content/drive/MyDrive/SilentTrails/checkpoints'
TRAIN_DATASETS = 'ff++ hidf deepdetect'
EPOCHS     = 15
BATCH_SIZE = 16

!python train.py \
    --stage 1 \
    --data_root "{DATA}" \
    --datasets {TRAIN_DATASETS} \
    --epochs {EPOCHS} \
    --batch_size {BATCH_SIZE} \
    --save_dir "{CKPTS}" \
    --resume "{CKPTS}/stage1_latest.pth" \
    --num_workers 2
"""


# ════════════════════════════════════════════════════════════════
# CELL 16 — CROSS-DATASET EVALUATION ON CELEBDF-V2
#
# THE MOST IMPORTANT CELL FOR YOUR FYP DEFENSE.
# Tests your trained model on data it has NEVER seen.
# Run immediately after Stage 1 finishes.
#
# The AUC number from this cell is what you present.
# ════════════════════════════════════════════════════════════════
"""
import sys, os, torch, numpy as np
from tqdm import tqdm
sys.path.insert(0, '/content/silent-trails/deepfake_model')

from src.models import DeepfakeForensicsModel
from src.data import create_dataloaders, get_val_transforms
from src.training import compute_all_metrics, print_metrics

DATA  = '/content/drive/MyDrive/SilentTrails/data'
CKPTS = '/content/drive/MyDrive/SilentTrails/checkpoints'

# Load best trained model
CKPT = f'{CKPTS}/stage1_best.pth'
if not os.path.exists(CKPT):
    print("ERROR: No checkpoint found. Run Cell 15 first.")
else:
    checkpoint = torch.load(CKPT, map_location='cuda', weights_only=False)
    model = DeepfakeForensicsModel(video_mode=False).cuda()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, val AUC: {checkpoint['val_auc']:.4f}")

    # Load ALL of CelebDF as test (never trained on it)
    val_tf = get_val_transforms(256)
    _, _, test_loader = create_dataloaders(
        data_root=DATA,
        batch_size=32,
        train_transform=val_tf,
        val_transform=val_tf,
        datasets=['celebdf'],
        val_ratio=0.0,
        test_ratio=1.0,
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on CelebDF-v2"):
            out = model(batch['image'].cuda())
            probs = out['probability'].cpu().squeeze()
            all_preds.extend(probs.tolist() if probs.dim() > 0 else [probs.item()])
            all_labels.extend(batch['label'].tolist())

    metrics = compute_all_metrics(all_labels, all_preds)
    print_metrics(metrics, prefix="CROSS-DATASET: Trained on FF++/HiDF/DeepDetect → Tested on CelebDF-v2")
    print(f"\n  YOUR CROSS-DATASET AUC: {metrics['auc']:.4f}")
    print(f"  (This is the number to present at your defense)")
"""


# ════════════════════════════════════════════════════════════════
# CELL 17 — STAGE 2: VIDEO TRAINING (optional, if time allows)
#
# Run this ONLY after Stage 1 is complete and cross-eval is done.
# Adds the Temporal Transformer for video-level reasoning.
# Expected time: 3-4 hours on A100.
# ════════════════════════════════════════════════════════════════
"""
%cd /content/silent-trails/deepfake_model

DATA  = '/content/drive/MyDrive/SilentTrails/data'
CKPTS = '/content/drive/MyDrive/SilentTrails/checkpoints'

# Only datasets with frame sequences (grouped by video)
VIDEO_DATASETS = 'ff++'

!python train.py \
    --stage 2 \
    --data_root "{DATA}" \
    --datasets {VIDEO_DATASETS} \
    --epochs 10 \
    --batch_size 4 \
    --save_dir "{CKPTS}" \
    --checkpoint "{CKPTS}/stage1_best.pth" \
    --num_frames 16 \
    --num_workers 2
"""


# ════════════════════════════════════════════════════════════════
# CELL 18 — GENERATE TRAINING PLOTS
# Run after training to get charts for your report/slides
# ════════════════════════════════════════════════════════════════
"""
import json, os, matplotlib.pyplot as plt

CKPTS = '/content/drive/MyDrive/SilentTrails/checkpoints'

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = {'Stage 1': ('#ff6b6b', '#4ecdc4'), 'Stage 2': ('#ff9f43', '#0abde3')}

for stage in [1, 2]:
    path = f'{CKPTS}/stage{stage}_history.json'
    if not os.path.exists(path):
        continue
    with open(path) as f:
        h = json.load(f)
    name = f'Stage {stage}'
    tc, vc = colors[name]
    axes[0].plot(h['train_loss'], label=f'{name} Train', color=tc)
    axes[0].plot(h['val_loss'],   label=f'{name} Val',   color=vc, linestyle='--')
    axes[1].plot(h['train_auc'], label=f'{name} Train', color=tc)
    axes[1].plot(h['val_auc'],   label=f'{name} Val',   color=vc, linestyle='--')

axes[0].set_title('Loss Curve'); axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].set_title('AUC Curve'); axes[1].set_xlabel('Epoch'); axes[1].axhline(0.85, color='gray', linestyle=':', label='Target 0.85')
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()

out_path = f'{CKPTS}/training_curves.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {out_path}")
"""


# ════════════════════════════════════════════════════════════════
# CELL 19 — QUICK SINGLE IMAGE TEST
# Upload any face image and get a prediction
# ════════════════════════════════════════════════════════════════
"""
import sys, torch, numpy as np
from PIL import Image
import matplotlib.pyplot as plt, cv2
sys.path.insert(0, '/content/silent-trails/deepfake_model')
from src.models import DeepfakeForensicsModel
from src.data import get_val_transforms

CKPTS = '/content/drive/MyDrive/SilentTrails/checkpoints'

# Load model
import os
ckpt_path = f'{CKPTS}/stage2_best.pth' if os.path.exists(f'{CKPTS}/stage2_best.pth') else f'{CKPTS}/stage1_best.pth'
ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=False)
model = DeepfakeForensicsModel(video_mode=False).cuda()
state = {k: v for k, v in ckpt['model_state_dict'].items() if k in model.state_dict()}
model.load_state_dict(state, strict=False)
model.eval()

from google.colab import files
print("Upload a face image:")
uploaded = files.upload()
fname = list(uploaded.keys())[0]

img = Image.open(fname).convert('RGB')
img_np = np.array(img)
tf = get_val_transforms(256)
tensor = tf(image=img_np)['image'].unsqueeze(0).cuda()

with torch.no_grad():
    out = model(tensor)

prob = out['probability'].item()
verdict = 'FAKE' if prob > 0.5 else 'REAL'
conf = prob * 100 if prob > 0.5 else (1 - prob) * 100

# Show image + attention heatmap
attn = out['attn_maps'].cpu().squeeze().mean(0).view(16, 16).numpy()
attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
heatmap = cv2.resize(attn, (256, 256))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(img_np); axes[0].set_title('Input'); axes[0].axis('off')
axes[1].imshow(np.array(img.resize((256,256)))); axes[1].imshow(heatmap, alpha=0.5, cmap='jet')
axes[1].set_title(f'{verdict} — {conf:.1f}% confidence'); axes[1].axis('off')
plt.tight_layout(); plt.show()
print(f"Result: {verdict} | Confidence: {conf:.1f}%")
"""
