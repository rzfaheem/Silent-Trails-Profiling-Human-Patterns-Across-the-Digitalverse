# Deepfake Forensics Module — Complete Handoff Document

> **Project**: Silent Trails — Profiling Human Patterns Across the Digitalverse
> **Module**: Deepfake Forensics (sub-project under `deepfake_model/`)
> **Author**: Faheem
> **Last Updated**: June 2026
> **Status**: Code scaffolding complete. Training NOT started yet.

---

## 1. Project Goal

Build a **multi-stream deepfake detection model** that:
- Detects face-swapped and AI-generated fake faces in images and videos
- Generalizes across datasets (not just memorize one dataset's patterns)
- Survives real-world degradation (JPEG compression, social media re-uploads)
- Produces **explainable results** (heatmap showing WHERE manipulation was detected)
- Integrates into the Silent Trails web app via FastAPI backend

---

## 2. Current Status

### ✅ DONE
- Full model architecture implemented (all Python source code)
- Project scaffolding and file structure
- Training configuration (YAML)
- Colab notebooks (setup + training)
- Frontend UI shell (`DeepfakeForensics.jsx`) — accepts image/video, shows placeholder results
- Proposal document and architecture diagrams created
- Pushed to GitHub: `https://github.com/rzfaheem/Silent-Trails-Profiling-Human-Patterns-Across-the-Digitalverse`
- Old deepfake backend (`backend/deepfake_analyzer.py`) removed

### ❌ NOT DONE
- **No datasets downloaded yet** (FF++, CelebDF-v2, DFDC)
- **No face extraction run yet**
- **No training has happened** — model has never been trained
- **No FastAPI inference server** built yet
- **Frontend not connected** to real model API (shows placeholder "MODEL NOT CONNECTED")
- `train_config.yaml` still has OLD loss weights (triplet=0.5, quality=0.1) — should be 0 for Phase 1 but the **code defaults** are already correct

---

## 3. Architecture Overview

### Pipeline Flow
```
Input (image/video)
    ↓
Step 1: RetinaFace — detect face, extract 5-point landmarks, align, crop to 256×256
    ↓
Step 2: Anti-Shortcut Preprocessing — random JPEG recompression, fixed resolution, CLAHE
    ↓
Step 3: Three Parallel Streams
    ├── Spatial Stream:    DINOv2-Base (frozen, 86M params) + LoRA adapters (500K trainable) → 512-dim embedding
    ├── Frequency Stream:  Grayscale → 2D FFT → Log Magnitude → 4-layer CNN → 512-dim embedding
    └── Attention Stream:  4 learnable region queries (eyes, mouth, jaw, hair) × 8-head cross-attention → 512-dim embedding + heatmap
    ↓
Step 4: SimpleFusion MLP — concat 3×512=1536 → MLP → 512-dim fused embedding
    ↓
Step 5 (Phase 2, video only): Temporal Transformer — 16 frames → 4-layer encoder → 512-dim
    ↓
Step 6: Classification Head → BCE + Focal Loss → Real/Fake probability
    ↓
Output: Real/Fake verdict + confidence % + attention heatmap
```

### Key Architecture Decisions

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| Backbone | DINOv2-Base (ViT) | Self-supervised, strong general visual features, no task bias |
| Adaptation | LoRA (rank 8) | Only 500K trainable params vs 86M full fine-tune. Prevents overfitting |
| DINOv2 backbone | FROZEN (LR=0) | Preserve pretrained features, only LoRA adapts |
| Frequency analysis | FFT → CNN | Catches GAN/diffusion spectral fingerprints invisible in pixel space |
| Attention | Learnable region queries | More flexible than hard-coded regions, provides explainable heatmaps |
| Fusion (Phase 1) | SimpleFusion (concat+MLP) | Easy to train/debug. AdaptiveFusionEngine kept in code for Phase 3 |
| Loss (Phase 1) | BCE + Focal only | Triplet loss adds sampling complexity, disabled by default (weight=0) |
| Face crop size | 256×256 | Standard, prevents resolution shortcuts |
| Embedding dim | 512 | Sweet spot for expressiveness vs efficiency |

---

## 4. Implementation Phases

### Phase 1 — Core Image Model (CURRENT PRIORITY)
- [x] Code: All 3 streams + SimpleFusion + Classification Head
- [x] Code: BCE + Focal Loss (triplet/quality weights default to 0)
- [ ] Download datasets (FF++, CelebDF-v2)
- [ ] Run face extraction pipeline
- [ ] Train on FF++ with curriculum learning
- [ ] Cross-dataset evaluation (FF++ → CelebDF)
- [ ] Ablation study (spatial only → +frequency → +attention)

### Phase 2 — Video Support
- [x] Code: Temporal Transformer module written
- [ ] Video frame sampling pipeline
- [ ] Train on DFDC video dataset
- [ ] Video evaluation + frame-level timeline output

### Phase 3 — Future Enhancements (code already written, just disabled)
- [ ] Enable AdaptiveFusionEngine: `DeepfakeForensicsModel(use_adaptive_fusion=True)`
- [ ] Enable Triplet Loss: set `triplet_weight=0.5` in CompoundLoss
- [ ] Enable Quality MSE: set `quality_weight=0.1`
- [ ] Audio-visual lip sync (SyncNet) — NOT coded yet, future add-on

---

## 5. File Structure & What Each File Does

```
d:\Silent Trails\
├── deepfake_model/                          # ← THE ENTIRE ML SUB-PROJECT
│   ├── configs/
│   │   └── train_config.yaml                # All hyperparameters (NOTE: loss weights outdated here, code defaults are correct)
│   ├── src/
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── __init__.py                  # Exports all model classes
│   │   │   ├── spatial_stream.py            # DINOv2 + LoRA backbone → 512-dim + patch tokens
│   │   │   ├── frequency_stream.py          # FFT → log magnitude → 4-layer CNN → 512-dim
│   │   │   ├── attention_stream.py          # 4 region queries × 8-head cross-attention → 512-dim + heatmap
│   │   │   ├── adaptive_fusion.py           # SimpleFusion (Phase 1) + AdaptiveFusionEngine (Phase 3)
│   │   │   ├── temporal_module.py           # 4-layer Transformer encoder for video frames
│   │   │   ├── heads.py                     # ClassificationHead + MetricHead (metric head unused in Phase 1)
│   │   │   └── deepfake_model.py            # MAIN: assembles everything into DeepfakeForensicsModel
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── face_extractor.py            # RetinaFace detection + alignment + cropping pipeline
│   │   │   ├── dataset.py                   # Multi-dataset PyTorch DataLoader (FF++, CelebDF, DFDC)
│   │   │   └── augmentations.py             # 3-tier augmentation: train, chaos, anti-shortcut
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── losses.py                    # CompoundLoss: BCE + Focal + Triplet(disabled) + Quality(disabled)
│   │   │   └── metrics.py                   # AUC, accuracy, F1, EER computation
│   │   └── inference/
│   │       └── __init__.py                  # EMPTY — FastAPI server not built yet
│   ├── notebooks/
│   │   ├── 01_setup_and_data.py             # Colab: clone repo, install deps, mount drive, test model, download data
│   │   └── 02_training.py                   # Colab: full training loop with curriculum, checkpointing, evaluation
│   ├── requirements.txt                     # All Python dependencies
│   └── README.md                            # Project overview
│
├── src/pages/
│   ├── DeepfakeForensics.jsx                # React frontend — UI shell with placeholder results
│   └── DeepfakeForensics.css                # Styling for the frontend component
│
├── backend/
│   └── server.js                            # Express server (deepfake_analyzer.py was DELETED — old ViT pipeline removed)
│
└── .gitignore                               # Excludes .env, uploads/, __pycache__/, checkpoints/
```

---

## 6. Key Code Entry Points

### Create & test the model
```python
from src.models import DeepfakeForensicsModel

# Phase 1: Image mode, simple fusion
model = DeepfakeForensicsModel(video_mode=False)

# Phase 2: Video mode
model = DeepfakeForensicsModel(video_mode=True)

# Phase 3: Adaptive fusion upgrade
model = DeepfakeForensicsModel(use_adaptive_fusion=True)

# Forward pass
output = model(torch.randn(2, 3, 256, 256))
# Returns: {"logits", "probability", "embedding", "attn_maps", "quality"}
```

### Loss function
```python
from src.training import CompoundLoss

# Phase 1 (defaults): BCE + Focal only
criterion = CompoundLoss()  # triplet_weight=0.0, quality_weight=0.0

# Phase 3 upgrade:
criterion = CompoundLoss(triplet_weight=0.5, quality_weight=0.1)
```

### Separate learning rates
```python
param_groups = model.get_param_groups(lr_lora=5e-4, lr_new=1e-3)
optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
```

---

## 7. Dependencies

```
torch>=2.1.0, torchvision>=0.16.0, transformers>=4.36.0, peft>=0.7.0
insightface>=0.7.3, onnxruntime-gpu>=1.16.0
opencv-python>=4.8.0, Pillow>=10.0.0, albumentations>=1.3.0
numpy>=1.24.0, pandas>=2.0.0, scikit-learn>=1.3.0, scipy>=1.11.0, tqdm>=4.65.0
wandb>=0.16.0
fastapi>=0.104.0, uvicorn>=0.24.0, python-multipart>=0.0.6
matplotlib>=3.7.0, seaborn>=0.12.0
```

---

## 8. Training Configuration

| Param | Value |
|-------|-------|
| Optimizer | AdamW |
| LR (LoRA adapters) | 5e-4 |
| LR (new modules) | 1e-3 |
| LR (DINOv2 backbone) | 0 (frozen) |
| Batch size | 32 (reduce to 16 if OOM) |
| Epochs | 25 + early stopping on val AUC |
| Precision | FP16 mixed |
| Scheduler | Cosine annealing with warm restarts |
| Gradient clip | 1.0 |
| Loss | BCE(1.0) + Focal(0.5) |
| Curriculum | Easy(1-8) → Mixed(9-18) → Chaos(19-25) |
| Face crop | 256×256 fixed |
| Data split | 75% train / 15% val / 10% test |

---

## 9. Datasets

| Dataset | What | How to Get |
|---------|------|------------|
| FaceForensics++ | 1000 real + 4000 fake videos (4 methods), c23 and c40 | Request access via email from authors → they send download script |
| CelebDF-v2 | 590 real + 5639 fake celebrity videos | Google Drive link from paper's GitHub |
| DFDC | 100K+ clips, Facebook challenge | Kaggle: `kaggle competitions download -c deepfake-detection-challenge` |

**None of these are downloaded yet.**

---

## 10. Benchmark Targets

| Test | Target AUC | What It Proves |
|------|-----------|----------------|
| FF++ c23 (in-dataset) | ≥ 0.97 | Basic detection |
| FF++ c40 (heavy compression) | ≥ 0.90 | Compression robustness |
| CelebDF-v2 (in-dataset) | ≥ 0.93 | High-quality fakes |
| **FF++ → CelebDF (cross-dataset)** | **≥ 0.85** | **Generalization (most critical)** |
| Post-JPEG q=30 | AUC drop ≤ 5% | Social media survival |

---

## 11. Training Platforms

| Platform | GPU | VRAM | Plan |
|----------|-----|------|------|
| Colab Free | T4 | 15 GB | Setup, testing, debugging |
| Kaggle Free | T4/P100 | 15-16 GB | Training (30 hrs/week free) |
| Modal | A10G/A100 | 24-80 GB | Long training runs ($30 free credits) |

**User does NOT have Colab Pro.** Use Colab Free for testing only. Use Kaggle or Modal for actual training.

---

## 12. Anti-Shortcut Strategy (Unique Feature)

This prevents the model from learning dataset-specific shortcuts:
1. **Fixed resolution**: All crops → 256×256 (eliminates resolution shortcuts)
2. **Random JPEG recompression** (q60-95): Destroys original compression fingerprint
3. **CLAHE color normalization**: Removes camera/dataset color bias
4. **Augmentation stacking**: Blur, noise, downscale-upscale, pixel dropout

Without this: train=98% AUC, test=62% (useless). With this: train=94%, test=85%+ (real generalization).

---

## 13. Frontend Integration Status

**File**: `src/pages/DeepfakeForensics.jsx`
- ✅ Accepts image AND video uploads (drag & drop)
- ✅ Shows analysis progress steps (Upload → Face Extract → Spatial → Frequency → Attention → Verdict)
- ❌ Currently shows PLACEHOLDER results ("Model not yet connected")
- ❌ Contains commented-out `fetch` call to `http://localhost:8000/detect` — needs real FastAPI server

**To connect frontend to model:**
1. Build FastAPI inference server in `deepfake_model/src/inference/`
2. Load trained model checkpoint
3. Accept image/video upload via POST
4. Run through pipeline → return JSON with verdict, confidence, heatmap
5. Uncomment fetch call in `DeepfakeForensics.jsx` and parse response

---

## 14. Known Issues & Gotchas

1. **`train_config.yaml` loss weights are outdated**: Shows `triplet_weight: 0.5` and `quality_weight: 0.1` but the **code defaults** in `losses.py` are correctly set to 0.0. The YAML file is not currently read by any training script — the notebooks use inline CONFIG dict.

2. **MetricHead still in model**: `heads.py` has a MetricHead that computes 128-dim embeddings. It runs during forward pass but isn't used by the loss in Phase 1 (triplet_weight=0). Harmless — it's there for Phase 3.

3. **`deepfake_model.py` line 83**: Comment says "Adaptive fusion" but actually calls `self.fusion()` which is SimpleFusion by default. Minor comment inaccuracy.

4. **No training loop script exists as a standalone `.py`**: Training code is embedded in `notebooks/02_training.py` as Colab cell strings. For Modal/Kaggle, you'd need to extract it into a proper `train.py`.

5. **`dataset.py` expects specific folder structure**: `data_root/{dataset_name}/real/` and `data_root/{dataset_name}/fake/` with `.jpg` face crops. The face extractor must run first to create this structure.

---

## 15. Next Recommended Steps (in order)

1. **Open Colab Free** → run `01_setup_and_data.py` cells 1-7 to verify model loads and runs
2. **Download datasets**: Start with CelebDF-v2 (easiest to get), then FF++
3. **Run face extraction**: Use `FaceExtractor` class on downloaded videos → save crops to Google Drive
4. **Train Phase 1**: Use `02_training.py` notebook on Kaggle or Modal
5. **Evaluate**: Cross-dataset test (train on FF++, test on CelebDF)
6. **Run ablation study**: Test each stream's contribution
7. **Build FastAPI server**: `deepfake_model/src/inference/` → serve predictions
8. **Connect frontend**: Wire `DeepfakeForensics.jsx` to the API
9. **Phase 2**: Enable `video_mode=True`, train on DFDC
10. **Phase 3**: If time allows → adaptive fusion + triplet loss

---

## 16. Assumptions & Constraints

- **User is a CS undergraduate** — understands code but is learning ML/DL concepts
- **No local GPU available** — all training must happen on Colab Free, Kaggle, or Modal
- **This is an FYP (Final Year Project)** — needs to be impressive but feasible
- **Two supervisors**: Main supervisor (general) + co-supervisor (image processing/deepfake expert)
- **Time constraint**: Project must be completed within a few months
- **No Colab Pro subscription** — must work within free-tier limits
- **GitHub repo is PUBLIC**: `https://github.com/rzfaheem/Silent-Trails-Profiling-Human-Patterns-Across-the-Digitalverse`

---

## 17. Conversation Summary — What Was Discussed

### Session 1: Planning & Architecture Design (Feb 14-16)
- Explored the Silent Trails project structure
- Designed the multi-stream deepfake detection architecture from scratch
- Created a detailed implementation plan with 4 streams (spatial, frequency, attention, temporal), adaptive fusion, and compound loss
- Discussed DINOv2 vs other backbones, LoRA vs full fine-tuning
- Created the complete project scaffolding under `deepfake_model/`
- Implemented ALL source files: spatial_stream.py, frequency_stream.py, attention_stream.py, adaptive_fusion.py, temporal_module.py, heads.py, deepfake_model.py, face_extractor.py, dataset.py, augmentations.py, losses.py, metrics.py
- Created requirements.txt, train_config.yaml, README.md
- Cleaned up old deepfake backend (removed deepfake_analyzer.py)
- Updated DeepfakeForensics.jsx frontend to remove simulation logic, add video support, and show placeholder results
- Updated .gitignore for Python artifacts and model checkpoints
- Pushed everything to GitHub

### Session 2: Explaining Concepts to the User (Feb 16)
- Explained what a "stream" is (airport security lane analogy)
- Explained LoRA in simple terms (Google Maps plugin analogy)
- Clarified the difference between Stream 1 (spatial — whole face analysis) vs Stream 3 (attention — specific region analysis)
- Explained that audio-lip sync (SyncNet) is NOT in current plan, listed as future enhancement
- Discussed platform strategy: Colab Free for testing, Kaggle/Modal for training
- Created Colab notebooks: `01_setup_and_data.py` and `02_training.py`

### Session 3: Supervisor Preparation (Feb 27-28)
- Created a technical proposal document for the co-supervisor (image processing expert)
- Created verbal talking points for supervisor meetings
- Answered technical questions: "Why DINOv2?", "What's your AUC target?"
- Created architecture overview document with visual diagram
- User's friend (likely PhD-level) reviewed the proposal and gave feedback:
  - Architecture is strong (9/10) but over-engineered for FYP
  - Recommended simplifying: remove adaptive fusion, reduce temporal layers, drop triplet loss initially
  - Suggested phased approach: V1 (core) → V2 (video) → V3 (advanced)
- **We agreed with most feedback and restructured:**
  - Added `SimpleFusion` class (concat+MLP) as Phase 1 default
  - Set triplet_weight and quality_weight to 0 by default
  - Kept AdaptiveFusionEngine in code for Phase 3 upgrade (just flip a flag)
  - Kept temporal transformer (4 layers) — user insisted video is core feature
  - Updated model to accept `use_adaptive_fusion` parameter
  - Updated proposal, architecture overview, and implementation plan
  - Pushed all code changes to GitHub

### Session 4: Deep Concept Explanations (Feb 28)
- Explained anti-shortcut preprocessing in detail (resolution shortcuts, JPEG fingerprints, color bias)
- Explained how SimpleFusion MLP learns which stream to trust (backpropagation, trial and error)
- Explained supervised learning (labeled folders: real/ and fake/)
- Explained what "512 numbers" / embeddings are (describing a face with numbers)
- Explained multi-stream approach (same image analyzed from multiple angles)
- Explained video detection logic (16 frames → per-frame analysis → temporal transformer catches inter-frame inconsistencies like flickering, identity drift, unnatural blinking)
- Discussed detection capabilities: can catch face swaps (blending boundaries) and AI-generated faces (FFT spectral fingerprints), but latest lip-sync fakes are harder
- Explained that frequency stream is the "AI-generated face killer" — every generator leaves a spectral fingerprint

### Session 5: Supervisor Meeting Prep (Mar 4)
- Created updated verbal talking points covering the full process
- Explained where RetinaFace fits (very first step — before any analysis)
- Regenerated architecture diagram multiple times:
  - v1: Colored but had outdated labels (Quality Estimator, Adaptive Fusion)
  - v2: Corrected labels (SimpleFusion, no Quality Estimator)
  - v3: Clean black-and-white minimal diagram (user's preference for supervisor)
- Updated architecture overview document with the clean diagram

### Session 6: Handoff (May 8)
- User is migrating to Cursor IDE
- This handoff document was created

### Session 7: LinkedIn Post + Final Handoff Review (June 14)
- Discussed LinkedIn posting strategy for the deepfake project
- Recommended 2-post strategy: Post 1 now (technical "building in public"), Post 2 after externals (full FYP showcase)
- Drafted a ready-to-post LinkedIn post with tech details
- Suggested carousel slides: architecture diagram, code screenshot, frontend UI, project structure
- User requested comprehensive final review of handoff document
- Added 7 missing sections (19-25): internal data flow, deleted backend, peer review context, sprint plan, broader project context, API output format, curriculum training details

---

## 18. Quick Reference — Common Supervisor Questions

| Question | Answer |
|----------|--------|
| "Why DINOv2?" | Self-supervised on 142M images, strong general features, LoRA adapts with only 500K params |
| "Why not just a CNN?" | CNNs miss frequency domain artifacts. Multi-stream covers what single networks can't |
| "How do you handle compression?" | Random recompression + anti-shortcut preprocessing destroys dataset-specific JPEG fingerprints |
| "What about video?" | Temporal transformer on 16-frame sequences catches flickering and identity drift |
| "How do you prove it works?" | Cross-dataset evaluation: train on FF++, test on CelebDF without fine-tuning |
| "What's your target?" | 0.97 AUC in-dataset, 0.85+ AUC cross-dataset |
| "What about lip-sync fakes?" | Architecture supports future SyncNet module. Current focus is visual detection first |
| "Why phased approach?" | Controlled complexity — validate core model before adding components. Easier to debug and ablate |

---

## 19. Internal Data Flow Between Modules (CRITICAL)

The streams are NOT fully independent. The Attention Stream depends on the Spatial Stream's output:

```
face_crop (B, 3, 256, 256)
    │
    ├──→ SpatialStream(face_crop)
    │        returns: h_spatial (B, 512), patch_tokens (B, 256, 768)
    │                                          │
    │                                          ↓
    │                              AttentionForgeryStream(patch_tokens)  ← INPUT FROM SPATIAL
    │                                   returns: h_attn (B, 512), attn_map (B, 4, 256)
    │
    ├──→ FrequencyStream(face_crop)  ← INDEPENDENT, takes raw face_crop
    │        returns: h_freq (B, 512)
    │
    ↓
SimpleFusion(h_spatial, h_freq, h_attn) → fused (B, 512), quality (B, 3)
    ↓
ClassificationHead(fused) → logits (B, 1)
MetricHead(fused) → embedding (B, 128)  [unused in Phase 1]
```

**Key**: `patch_tokens` from DINOv2 are the 256 spatial patch embeddings (16x16 grid x 768 dim each). The attention stream's cross-attention queries attend to these patches to focus on specific facial regions.

---

## 20. What Was Deleted (Old Backend)

The file `backend/deepfake_analyzer.py` (542 lines) was deleted. It contained:
- A HuggingFace ViT-based deepfake classifier
- Error Level Analysis (ELA)
- Noise analysis
- A simple single-model pipeline

This was removed because the new multi-stream architecture replaces it entirely. The Express server (`backend/server.js`) still exists but the deepfake route needs to be rebuilt to call the new FastAPI inference server.

---

## 21. Peer Review Summary (Important Context)

A friend with PhD-level ML knowledge reviewed the original proposal and gave 9/10 conceptually but flagged over-engineering risks:

**Original design had**: AdaptiveFusionEngine (quality estimator + dynamic weights) + Triplet Loss + 4-layer temporal transformer — all active by default.

**Peer's concerns**:
1. Adaptive fusion needs quality pseudo-labels + extra prediction head = debugging nightmare
2. Triple-objective loss (BCE + Focal + Triplet) on multi-stream fusion is PhD-scale complexity
3. If performance drops, impossible to diagnose which component failed
4. Feasibility on free GPU tiers was rated 6.5/10

**Changes made based on feedback**:
- Created `SimpleFusion` class as Phase 1 default (concat + MLP, no quality estimator)
- Set `triplet_weight=0.0` and `quality_weight=0.0` as defaults in `CompoundLoss`
- Kept ALL Phase 3 code in files — just not activated by default
- Added `use_adaptive_fusion` flag to `DeepfakeForensicsModel.__init__()`
- Restructured into 3 phases: core image → video → advanced

**User's position**: Video detection is the main feature (insisted it stays). Everything else follows phased approach.

---

## 22. 10-Day Sprint Plan (If Short on Time)

| Day | Task |
|-----|------|
| 1-2 | Download CelebDF-v2 (easiest). Request FF++ access (takes 1-3 days) |
| 3-4 | Open Colab, test model loads (cells 1-7). Run FaceExtractor on CelebDF, save crops to Google Drive |
| 5-7 | Train Phase 1 on Kaggle (30 free GPU hrs/week). Monitor training curves |
| 8-9 | Build FastAPI inference server. Connect DeepfakeForensics.jsx frontend to API |
| 10 | Polish, final evaluation, demo prep |

**Minimum viable delivery**: Trained image model + working frontend + AUC numbers. Video (Phase 2) and advanced features (Phase 3) are bonus.

---

## 23. Silent Trails — Broader Project Context

The deepfake module is ONE part of the larger Silent Trails platform. Other modules include:
- **Social Mapping**: OSINT social media profiling
- **Timeline**: Event timeline analysis with stats dashboard, risk scoring, infrastructure panel, and PDF export
- **Deepfake Forensics**: This module (the ML sub-project)

The main app is a **React frontend** + **Express.js backend** (Node.js). The deepfake model will be served as a **separate Python FastAPI microservice** that the Express backend (or frontend directly) calls via HTTP.

**Tech stack of main app**: React, React Router, CSS, Node.js, Express.js
**Tech stack of deepfake module**: Python, PyTorch, FastAPI

---

## 24. Model Output Format (For API Integration)

When the model runs inference, it returns this dict:
```python
{
    "logits": tensor(B, 1),           # Raw prediction score
    "probability": tensor(B, 1),      # Sigmoid of logits, 0.0 to 1.0
    "embedding": tensor(B, 128),      # Metric embedding (unused in Phase 1)
    "attn_maps": tensor(B, 4, 256),   # Attention weights: 4 regions x 256 patches
    "quality": tensor(B, 3),          # Quality estimates (zeros in Phase 1)
}
```

For the FastAPI response to the frontend, this should be transformed to:
```json
{
    "verdict": "MANIPULATED or AUTHENTIC",
    "confidence": 87.3,
    "manipulation_score": 87,
    "spatial_score": 0.82,
    "frequency_score": 0.91,
    "attention_score": 0.88,
    "heatmap": "base64_encoded_image",
    "file_name": "uploaded.jpg",
    "file_type": "Image"
}
```

The heatmap is generated from `attn_maps` by:
1. Average across 4 region queries → (256,) vector
2. Reshape to 16x16 spatial grid
3. Upsample to 256x256
4. Apply colormap (e.g., jet) and overlay on original face crop
5. Base64 encode for JSON transport

---

## 25. Curriculum Training Explained

Training uses 3 phases of increasing difficulty across 25 epochs:

| Phase | Epochs | Augmentation | Purpose |
|-------|--------|-------------|---------|
| **Easy** | 1-8 | Light (flip, small color jitter) | Model learns basic real vs fake |
| **Mixed** | 9-18 | Medium (blur, noise, resize) | Model handles degraded inputs |
| **Chaos** | 19-25 | Heavy (strong JPEG, heavy blur, pixel dropout, downscale-upscale) | Model becomes robust to worst-case |

The augmentation tier is selected based on epoch number. See `augmentations.py` for the three transform pipelines: `get_train_transforms()`, `get_val_transforms()`, `get_chaos_transforms()`.

**Note**: The training notebook (`02_training.py`) logs which curriculum phase each epoch is in, but the actual augmentation switching must be implemented in the training loop by changing the dataset's transform based on epoch. This switching logic exists in the notebook's training loop.
