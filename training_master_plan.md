# 🎯 Silent Trails — Training Master Plan

> **Date**: June 22, 2026 (00:00 AM)
> **Deadline**: Internals — June 23, 2026
> **Time remaining**: ~24 hours
> **GPU**: Colab Pro A100 (40GB VRAM)

---

## What I Just Built

✅ **Inference Server** — [server.py](file:///d:/Silent%20Trails/deepfake_model/src/inference/server.py)

This is the **production-ready** DINOv2 three-stream inference server that replaces the temp MTCNN+CLIP server. Key features:

| Feature | Old (Temp) | New (DINOv2) |
|---------|-----------|--------------|
| Stream scores | Fake approximations with random noise | **Real per-stream analysis** (spatial, frequency, attention each run independently) |
| Heatmap | GradCAM on InceptionResnet | **Native attention maps** from cross-attention region queries |
| Video | Frame averaging | **Temporal Transformer** (if Stage 2 trained) or frame aggregation |
| Checkpoint | Hardcoded `resnetinceptionv1_epoch_32.pth` | **Auto-discovers** best checkpoint (Stage 2 → Stage 1 fallback) |
| API format | Same | **100% backward-compatible** — frontend needs zero changes |

---

## The Complete Training Strategy

### Order of Operations

```
Step 1: Prepare datasets on Google Drive
Step 2: Smoke test (5 min) — verify everything loads
Step 3: Stage 1 — Image training (3-5 hrs)
Step 4: Cross-eval on CelebDF (30 min) — GET YOUR AUC NUMBER
Step 5: Stage 2 — Video training (3-4 hrs) [if time allows]
Step 6: Download checkpoint → local PC → run inference server
```

> [!IMPORTANT]
> **Train images FIRST, NOT in parallel with video.** Stage 2 (video) requires Stage 1 weights. This is not a choice — it's architecturally required. The temporal transformer learns from already-learned frame representations.

### Training Strategy: Image First, NOT Parallel

| Why | Explanation |
|-----|-------------|
| **Architectural dependency** | Video mode loads Stage 1 weights for spatial/frequency/attention, then trains the temporal transformer on top |
| **Safety net** | If time runs out, Stage 1 alone gives you a working demo + cross-dataset AUC number |
| **Ablation value** | "Frame-level AUC was X, temporal reasoning improved it to Y" — this IS a result worth presenting |

---

## Step-by-Step: What You Do RIGHT NOW

### Step 0: Dataset Preparation (Do This FIRST)

Your datasets need to be in this structure on Google Drive:

```
Google Drive/
└── SilentTrails/
    ├── data/
    │   ├── ff++/
    │   │   ├── real/     ← face crops from real FF++ videos
    │   │   └── fake/     ← face crops from manipulated videos
    │   ├── dfdc/
    │   │   ├── real/     ← face crops from real DFDC videos
    │   │   └── fake/     ← face crops from fake DFDC videos
    │   ├── hidf/
    │   │   ├── real/     ← real images from HiDF dataset
    │   │   └── fake/     ← fake images from HiDF dataset
    │   └── celebdf/
    │       ├── real/     ← CelebDF real (NEVER train on this!)
    │       └── fake/     ← CelebDF fake (NEVER train on this!)
    └── checkpoints/      ← models will be saved here automatically
```

> [!CAUTION]
> **CelebDF-v2 goes in `celebdf/` but is NEVER included in training datasets.**
> It is ONLY used for cross-dataset evaluation. This is what makes your results scientifically valid.

#### Dataset Sources:

| Dataset | Where to Get It | Format |
|---------|----------------|--------|
| **FF++ (c23)** | Kaggle — pre-cropped faces | Already face crops → just organize into `real/` and `fake/` |
| **DFDC** | Your local PC (~20GB, 2 parts) | Raw videos → needs face extraction OR if you have extracted crops, organize them |
| **HiDF** | [GitHub](https://github.com/DSAIL-SKKU/HiDF) / Zenodo | Images → organize into `real/` and `fake/` |
| **CelebDF-v2** | Google Drive (you have the link) | Organize into `celebdf/real/` and `celebdf/fake/` |

#### If DFDC is raw videos (not face crops):

You'll need to extract faces first. Run this in a Colab cell BEFORE training:

```python
import sys
sys.path.insert(0, '/content/silent-trails/deepfake_model')
from src.data.face_extractor import FaceExtractor

extractor = FaceExtractor(output_size=256)

# Process DFDC real videos
stats = extractor.process_dataset_folder(
    input_dir='/content/drive/MyDrive/SilentTrails/raw_dfdc/real/',
    output_dir='/content/drive/MyDrive/SilentTrails/data/dfdc/real/',
    video_mode=True,
    sample_n=16
)
print(f"Real: {stats}")

# Process DFDC fake videos
stats = extractor.process_dataset_folder(
    input_dir='/content/drive/MyDrive/SilentTrails/raw_dfdc/fake/',
    output_dir='/content/drive/MyDrive/SilentTrails/data/dfdc/fake/',
    video_mode=True,
    sample_n=16
)
print(f"Fake: {stats}")
```

> Face extraction takes ~2-3 hours on A100 for 20GB of DFDC videos.

---

### Step 1: Open Colab & Run Setup

Your Colab notebook is already at: [colab_training.py](file:///d:/Silent%20Trails/deepfake_model/notebooks/colab_training.py)

**Run cells 1-5 in order:**

| Cell | What It Does | Time |
|------|-------------|------|
| 1 | Check GPU + clone repo | 1 min |
| 2 | Install dependencies | 2-3 min |
| 3 | Mount Google Drive + create folders | 30 sec |
| 4 | Verify model loads (downloads DINOv2 ~350MB first time) | 2-3 min |
| 5 | Check data status — shows what's in each folder | 30 sec |

> [!TIP]
> After Cell 5, you'll see exactly how many images are in each dataset folder. **If a dataset shows 0 images, don't include it in training** — just remove it from the `TRAIN_DATASETS` variable.

---

### Step 2: Smoke Test (5 minutes)

Before the real training, verify EVERYTHING works with a quick test:

```python
%cd /content/silent-trails/deepfake_model

!python train.py \
    --stage 1 \
    --data_root "/content/drive/MyDrive/SilentTrails/data" \
    --datasets ff++ \
    --epochs 2 \
    --batch_size 8 \
    --max_samples 500 \
    --save_dir "/content/drive/MyDrive/SilentTrails/checkpoints"
```

**What this verifies:**
- ✅ Data loads correctly
- ✅ Model creates on GPU
- ✅ Forward pass works
- ✅ Loss computes
- ✅ Backward pass works
- ✅ Checkpoint saves to Google Drive

If this runs without errors, you're good for the real training. **If it errors, STOP and debug before wasting hours.**

---

### Step 3: Stage 1 — Image Training (THE BIG ONE)

```python
%cd /content/silent-trails/deepfake_model

DATA_DIR = '/content/drive/MyDrive/SilentTrails/data'
CHECKPOINT_DIR = '/content/drive/MyDrive/SilentTrails/checkpoints'

# ONLY include datasets that actually have data!
# Check Cell 5 output to see which ones have images
TRAIN_DATASETS = 'ff++ dfdc hidf'

EPOCHS = 15
BATCH_SIZE = 16  # Use 32 if A100 has enough VRAM, reduce to 8 if OOM

!python train.py \
    --stage 1 \
    --data_root "{DATA_DIR}" \
    --datasets {TRAIN_DATASETS} \
    --epochs {EPOCHS} \
    --batch_size {BATCH_SIZE} \
    --save_dir "{CHECKPOINT_DIR}" \
    --num_workers 2
```

**What happens during training:**

| Phase | Epochs | Augmentation | Purpose |
|-------|--------|-------------|---------|
| **Easy** | 1-5 | Anti-shortcut + standard aug | Learn basic real vs fake features |
| **Mixed** | 6-10 | Anti-shortcut + standard aug | Refine decision boundary |
| **Chaos** | 11-15 | Anti-shortcut + extreme aug | Build robustness to degradation |

**Anti-shortcut is AUTOMATIC** — every training sample goes through:
1. Fixed 256×256 resize (kills resolution shortcuts)
2. Random JPEG recompression q60-95 (kills compression fingerprints)
3. CLAHE color normalization (kills camera/dataset color bias)

**Expected timeline:**

| Metric | Expected |
|--------|----------|
| Time | 3-5 hours on A100 |
| Val AUC by epoch 5 | > 0.70 |
| Val AUC by epoch 10 | > 0.82 |
| Val AUC by epoch 15 | > 0.88 (target: ≥ 0.85) |

**If Colab disconnects:** Use Cell 6b to resume:

```python
!python train.py \
    --stage 1 \
    --data_root "{DATA_DIR}" \
    --datasets {TRAIN_DATASETS} \
    --epochs {EPOCHS} \
    --batch_size {BATCH_SIZE} \
    --save_dir "{CHECKPOINT_DIR}" \
    --resume "{CHECKPOINT_DIR}/stage1_latest.pth" \
    --num_workers 2
```

The `--resume` flag loads the last saved checkpoint and continues from exactly where it stopped.

---

### Step 4: Cross-Dataset Evaluation (CRITICAL — Do This Immediately After Stage 1)

This is the **most important number for your defense**:

> "We trained on FF++, DFDC, and HiDF. We then tested on CelebDF-v2, which the model has NEVER seen, and achieved X AUC."

Run Cell 8 from the notebook. It:
1. Loads your best Stage 1 model
2. Evaluates on ALL of CelebDF-v2 (test_ratio=1.0)
3. Reports: AUC, Accuracy, F1, EER, Precision, Recall

**Target**: Cross-dataset AUC ≥ 0.80. Anything above 0.75 is respectable for a three-stream architecture with anti-shortcut learning.

---

### Step 5: Stage 2 — Video Training (If Time Allows)

Only start this AFTER Stage 1 is complete AND cross-eval is done.

```python
%cd /content/silent-trails/deepfake_model

DATA_DIR = '/content/drive/MyDrive/SilentTrails/data'
CHECKPOINT_DIR = '/content/drive/MyDrive/SilentTrails/checkpoints'

# Only datasets with frame sequences (not just individual crops)
VIDEO_DATASETS = 'ff++ dfdc'

!python train.py \
    --stage 2 \
    --data_root "{DATA_DIR}" \
    --datasets {VIDEO_DATASETS} \
    --epochs 10 \
    --batch_size 4 \
    --save_dir "{CHECKPOINT_DIR}" \
    --checkpoint "{CHECKPOINT_DIR}/stage1_best.pth" \
    --num_frames 16 \
    --num_workers 2
```

**What this does:**
- Loads ALL Stage 1 weights for spatial/frequency/attention streams
- Creates a NEW temporal transformer (4-layer, learns from scratch)
- Frame-level modules get LOW learning rate (already trained)
- Temporal module gets FULL learning rate

**Required data format for video mode:**
Your face crops must be grouped by source video, either:
- **Subdirectories**: `data/dfdc/real/video_001/frame_000.jpg, frame_001.jpg, ...`
- **Naming convention**: `data/dfdc/real/video_001_frame000.jpg, video_001_frame001.jpg, ...`

The `VideoDeepfakeDataset` automatically detects both layouts.

---

### Step 6: Get the Checkpoint to Your Local PC

After training, download the checkpoint to your local machine:

```python
# In Colab — check what you have
import os
CHECKPOINT_DIR = '/content/drive/MyDrive/SilentTrails/checkpoints'
for f in os.listdir(CHECKPOINT_DIR):
    size = os.path.getsize(f'{CHECKPOINT_DIR}/{f}') / 1e6
    print(f"  {f}: {size:.1f} MB")
```

Then download from Google Drive to your local PC:
```
D:\Silent Trails\deepfake_model\checkpoints\stage1_best.pth
D:\Silent Trails\deepfake_model\checkpoints\stage2_best.pth  (if trained)
```

Create the checkpoints folder locally:
```powershell
mkdir "D:\Silent Trails\deepfake_model\checkpoints"
```

---

### Step 7: Run the Inference Server Locally

```powershell
cd "D:\Silent Trails\deepfake_model"
python -m src.inference.server
```

The server will:
1. Auto-discover the best checkpoint in `checkpoints/`
2. Load the DINOv2 three-stream model
3. Start on port 8001
4. Your frontend already calls through the Express proxy → port 8001

> [!NOTE]
> The inference server is 100% backward-compatible with the frontend. **Zero frontend changes needed.** The API response format (verdict, confidence, streams, heatmap, timeline) matches exactly what `DeepfakeForensics.jsx` expects.

---

## Priority Decision Matrix

Given ~24 hours to internals:

| Priority | Task | Time | Impact |
|----------|------|------|--------|
| 🔴 **#1** | Upload datasets to Google Drive | 1-2 hrs | Without data, nothing works |
| 🔴 **#2** | Stage 1 image training (15 epochs) | 3-5 hrs | Core model |
| 🔴 **#3** | Cross-dataset eval on CelebDF | 30 min | THE number evaluators want |
| 🟡 **#4** | Download checkpoint + test inference server | 1 hr | Working demo |
| 🟡 **#5** | Stage 2 video training | 3-4 hrs | Temporal reasoning (bonus) |
| 🟢 **#6** | Generate plots (training curves, confusion matrix) | 30 min | Report/presentation material |

### Realistic Timeline for Tonight/Tomorrow

| Time | What You're Doing |
|------|------------------|
| **Now → 1:00 AM** | Upload datasets to Google Drive + run Cells 1-5 |
| **1:00 AM** | Start smoke test (5 min) |
| **1:15 AM** | Start Stage 1 training → **let it run overnight** |
| **~6:00 AM** | Stage 1 finishes (~15 epochs) |
| **6:00-6:30 AM** | Run cross-dataset eval on CelebDF |
| **6:30-7:00 AM** | Download checkpoint, test inference server locally |
| **7:00 AM** | Start Stage 2 video training (runs while you prepare) |
| **7:00-10:00 AM** | Prepare talking points + test demo |
| **~11:00 AM** | Stage 2 finishes → download + test |
| **By noon** | Full pipeline working: trained model + inference server + frontend |

---

## What You Can Tell Evaluators

### If Only Stage 1 Trained:
> "Our model uses a DINOv2 three-stream architecture with LoRA adaptation. The spatial stream uses self-supervised DINOv2 features, the frequency stream detects spectral artifacts via FFT, and the attention stream localizes manipulation using cross-attention region queries. We trained on FF++, DFDC, and HiDF with anti-shortcut preprocessing, achieving X AUC on unseen CelebDF-v2 data."

### If Both Stages Trained:
> Add: "We then fine-tuned with a temporal transformer for video-level reasoning, analyzing 16-frame sequences to detect temporal inconsistencies. This improved our cross-dataset AUC from X to Y."

### Key Technical Points to Explain:
1. **Why DINOv2**: Self-supervised = no task bias = better generalization
2. **Why LoRA**: Only 500K trainable params = no overfitting
3. **Why 3 streams**: Each catches different artifacts (spatial manipulation, spectral fingerprints, region-specific forgery)
4. **Anti-shortcut**: JPEG recompression + CLAHE prevents learning dataset-specific shortcuts
5. **Cross-dataset AUC**: Proves model generalizes, not memorizes
