# Training Battle Plan — Final & Definitive

> **Date**: June 20, 2026
> **Goal**: Train the DINOv2 three-stream model, cross-evaluate on CelebDF-v2, and prepare for defense

---

## 1. Dataset Strategy — Confirmed

| Dataset | Role | Source | Format |
|---------|------|--------|--------|
| **FF++ (c23)** | Training | Kaggle (pre-cropped faces) | Face crops `.jpg` |
| **DFDC** (~20GB, 2 parts) | Training | Downloaded/Kaggle | Needs face extraction |
| **HiDF** (62K images, 8K videos) | Training | [GitHub](https://github.com/DSAIL-SKKU/HiDF) / Zenodo | KDD 2025 paper |
| **CelebDF-v2** | **Cross-eval ONLY** ❌ Never train on this | Google Drive | Ready |

**Cross-dataset evaluation story for defense:**
> "We trained on FF++, DFDC, and HiDF — spanning classic deepfakes, real-world diversity, and modern human-indistinguishable fakes. We then tested on CelebDF-v2, which the model has NEVER seen, achieving X AUC."

---

## 2. AI-Generated Image Detection — Your Concern Addressed

### The Problem
Your supervisors might ask: *"Can your model detect AI-generated faces from 2025-2026 models like DALL-E 3, Midjourney, Stable Diffusion 3, Flux?"*

### Why Your Architecture Already Handles This (partially)

Your **Frequency Stream** (FFT → log-magnitude → CNN) is the key here. All neural image generators — whether GANs (StyleGAN) or diffusion models (SD3, DALL-E 3) — leave **spectral fingerprints** in the frequency domain that don't exist in real photographs. Your frequency stream learns exactly these patterns.

Your **anti-shortcut learning** (JPEG recompression + CLAHE) forces the model to look at genuine artifacts rather than dataset-specific compression patterns, making it generator-agnostic.

### But You Want Explicit Training Coverage

**HiDF already covers this partially** — it contains modern AI-generated content that is *human-indistinguishable*. But to explicitly train on AI-generated faces from known generators:

### Recommended: DeepDetect-2025 (Kaggle)

| Property | Details |
|----------|---------|
| **Size** | 100,000+ images |
| **Generators** | DALL-E 3, Stable Diffusion 3, Midjourney, StyleGAN3 |
| **Format** | Real vs Fake images, clearly labeled |
| **Availability** | Kaggle — instant download |
| **Why this over GenImage** | GenImage (2023) only covers SD 1.x/2.x. DeepDetect-2025 covers SD3, DALL-E 3, Midjourney — the models evaluators will ask about |

> [!IMPORTANT]
> **However — this is a BONUS, not a priority.** Your core pipeline (FF++ + DFDC + HiDF → CelebDF-v2 cross-eval) must work first. If you have time after Stage 1 training is running, download DeepDetect-2025 and add it as a 4th training dataset. 
>
> **If asked at defense about AI-generated images:** "Our frequency stream detects spectral fingerprints common to ALL neural generators. We also include HiDF, a KDD 2025 dataset of human-indistinguishable deepfakes. Our anti-shortcut preprocessing makes the model generator-agnostic by design."

---

## 3. Anti-Shortcut Learning — Confirmed Active

Your three anti-shortcut techniques are **already baked into the training pipeline** in `augmentations.py`:

| Technique | Implementation | Line | Purpose |
|-----------|---------------|------|---------|
| **Fixed 256×256 resolution** | `A.Resize(face_size, face_size)` | L146 | Prevents resolution-based shortcuts |
| **Random JPEG recompression** (q60-95) | `A.ImageCompression(quality_lower=60, quality_upper=95, p=0.8)` | L147 | Destroys dataset-specific compression fingerprints |
| **CLAHE color normalization** | `A.CLAHE(clip_limit=2.0, p=0.3)` | L148 | Removes camera/dataset color bias |

These are applied **BEFORE** regular augmentation in both `compose_training_transforms()` and `compose_chaos_transforms()`. The curriculum system ensures they're active in every training phase.

**No action needed — anti-shortcut is automatic.**

---

## 4. Training Pipeline — Step by Step

### Step 0: Data Preparation (do this FIRST)
1. FF++ from Kaggle → already pre-cropped → just organize into `data/ff++/real/` and `data/ff++/fake/`
2. DFDC 2 parts → download to Colab → run face extraction with `FaceExtractor` 
3. HiDF → download from GitHub/Zenodo → organize into `data/hidf/real/` and `data/hidf/fake/`
4. CelebDF-v2 → already on Google Drive → organize into `data/celebdf/real/` and `data/celebdf/fake/`

### Step 1: Smoke Test (5 minutes)
```bash
python train.py --stage 1 --data_root /path/to/data --datasets ff++ --epochs 2 --max_samples 500
```
This verifies: data loads → model creates → forward pass works → loss computes → checkpoint saves.

### Step 2: Stage 1 — Image Training (3-5 hours on A100)
```bash
python train.py --stage 1 \
    --data_root /path/to/data \
    --datasets ff++ dfdc hidf \
    --epochs 15 \
    --batch_size 16 \
    --save_dir /path/to/checkpoints
```

**What happens:**
- Anti-shortcut preprocessing runs automatically (JPEG recomp + CLAHE)
- Curriculum: Easy (1-5) → Mixed (6-10) → Chaos (11-15)
- Balanced sampling across datasets AND real/fake classes
- Checkpoint saved every epoch to Google Drive (survives disconnects)
- Best model saved based on validation AUC

**Expected output:** `stage1_best.pth` with val AUC ≥ 0.85

### Step 3: Stage 2 — Video Training (3-4 hours on A100)
```bash
python train.py --stage 2 \
    --data_root /path/to/data \
    --datasets ff++ dfdc \
    --epochs 10 \
    --batch_size 4 \
    --checkpoint /path/to/checkpoints/stage1_best.pth \
    --num_frames 16 \
    --save_dir /path/to/checkpoints
```

**What happens:**
- Loads Stage 1 weights for spatial/frequency/attention streams
- Temporal Transformer trains from scratch with full LR
- Frame-level modules train with reduced LR (already learned)
- Face crops must be grouped by video (subdirectory or naming convention)

**Expected output:** `stage2_best.pth` with improved video-level AUC

### Step 4: Cross-Dataset Evaluation on CelebDF-v2
- Load best model (Stage 2 if available, else Stage 1)
- Evaluate on CelebDF-v2 data the model has NEVER seen
- Report: AUC, F1, Accuracy, EER, Confusion Matrix, ROC Curve

### Step 5: Generate Plots
- Training loss curves (train vs val)
- AUC-ROC curves per epoch
- Cross-dataset confusion matrix
- Per-dataset performance comparison

---

## 5. What the Colab Notebook Covers

The notebook I'm building (`colab_training_final.py`) is a **single, complete, copy-paste-ready** script with these cells:

| Cell | Purpose | Time |
|------|---------|------|
| 1 | GPU check + clone repo | 1 min |
| 2 | Install dependencies | 2-3 min |
| 3 | Mount Google Drive + create folders | 30 sec |
| 4 | Verify model loads (downloads DINOv2 ~350MB first time) | 2-3 min |
| 5 | Check data status | 30 sec |
| 6 | **Stage 1: Image training** | 3-5 hrs |
| 6b | Resume Stage 1 (if disconnected) | — |
| 7 | **Stage 2: Video training** | 3-4 hrs |
| 7b | Resume Stage 2 (if disconnected) | — |
| 8 | Cross-dataset evaluation on CelebDF-v2 | 30-60 min |
| 9 | Generate training curves + plots | 2 min |
| 10 | Quick single-image test | 1 min |

---

## 6. Key Notes

- **FF++ from Kaggle**: You can use `!kaggle datasets download ...` directly in Colab. The Kaggle API works perfectly there.
- **CelebDF from Drive**: Already accessible since Google Drive is mounted.
- **HiDF**: Check the GitHub repo for download instructions — likely a Zenodo/Google Drive link.
- **DFDC**: Download 2 parts from Kaggle using `!kaggle competitions download -f dfdc_train_part_X.tar`.
- **Stage 2 is NOT a bonus**: Both stages will be in the notebook. Stage 2 needs video-organized face crops (subdirectories per video or `videoname_frameXXX.jpg` naming).
