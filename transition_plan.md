# Transition Plan: HuggingFace Temp → Original DINOv2 Three-Stream Architecture

> **Date**: June 20, 2026
> **Time remaining**: ~1-2 days for training
> **Goal**: Disable temporary dual-stream, prepare for original architecture training

---

## 1. My Understanding of the Situation

You had a supervisor meeting ~2 days ago (around June 18). At that point:
- Your original DINOv2 + LoRA + Three-Stream architecture existed only as code scaffolding
- Zero training had been performed on your own architecture
- To demonstrate a working system, you integrated a **pretrained InceptionResnetV1 checkpoint** (MTCNN face detection + InceptionResnetV1 classifier) and **CLIP zero-shot** as a second stream
- This was always a temporary fallback — your real architecture remains the planned one

You now want to:
1. **Disable** (not delete) the temporary HuggingFace/MTCNN/CLIP implementation
2. **Preserve** all reusable infrastructure
3. **Prepare** the codebase for training the original DINOv2 + Three-Stream model
4. Get my training strategy recommendations

---

## 2. File-by-File Audit: What to Disable vs. Preserve

### Files to DISABLE (comment out / rename)

| File | What It Contains | Action |
|------|-----------------|--------|
| `deepfake_model/inference_server.py` | **Entire temporary implementation**: MTCNN, InceptionResnetV1, CLIP, dual-stream fusion, fake stream scores | **Rename** to `inference_server_temp.py` — preserves it entirely while clearly marking it as superseded |
| `deepfake_model/resnetinceptionv1_epoch_32.pth` | 282MB pretrained checkpoint for the temp model | **Keep as-is** — it's a binary file, no code changes needed. Just don't reference it from new code |
| `deepfake_model/inference_requirements.txt` | Dependencies for the temp server (facenet-pytorch, transformers for CLIP) | **Rename** to `inference_requirements_temp.txt` |

### Files to PRESERVE UNTOUCHED (Original Architecture — Ready for Training)

These are your planned architecture files. They are **complete, correct, and ready to use**:

| File | What It Contains | Status |
|------|-----------------|--------|
| `deepfake_model/src/models/deepfake_model.py` | Main model assembler (DeepfakeForensicsModel) | ✅ Ready |
| `deepfake_model/src/models/spatial_stream.py` | DINOv2-Base + LoRA → 512-dim + patch tokens | ✅ Ready |
| `deepfake_model/src/models/frequency_stream.py` | FFT → log-mag → 4-layer CNN → 512-dim | ✅ Ready |
| `deepfake_model/src/models/attention_stream.py` | 4 region queries × 8-head cross-attention → 512-dim + heatmap | ✅ Ready |
| `deepfake_model/src/models/adaptive_fusion.py` | SimpleFusion (Phase 1) + AdaptiveFusionEngine (Phase 3) | ✅ Ready |
| `deepfake_model/src/models/temporal_module.py` | 4-layer Temporal Transformer for video | ✅ Ready |
| `deepfake_model/src/models/heads.py` | ClassificationHead + MetricHead | ✅ Ready |
| `deepfake_model/src/models/__init__.py` | Exports all model classes | ✅ Ready |
| `deepfake_model/src/data/dataset.py` | DeepfakeDataset + VideoDeepfakeDataset + create_dataloaders | ✅ Ready |
| `deepfake_model/src/data/face_extractor.py` | RetinaFace detection + alignment pipeline | ✅ Ready |
| `deepfake_model/src/data/augmentations.py` | Anti-shortcut + training + chaos transforms | ✅ Ready |
| `deepfake_model/src/data/__init__.py` | Exports all data classes | ✅ Ready |
| `deepfake_model/src/training/losses.py` | CompoundLoss (BCE + Focal + Triplet + Quality) | ✅ Ready |
| `deepfake_model/src/training/metrics.py` | AUC, accuracy, F1, EER computation | ✅ Ready |
| `deepfake_model/train.py` | Standalone two-stage training script | ✅ Ready |
| `deepfake_model/configs/train_config.yaml` | Hyperparameter config | ✅ Ready |
| `deepfake_model/requirements.txt` | Python dependencies for original architecture | ✅ Ready |
| `deepfake_model/notebooks/01_setup_and_data.py` | Colab setup notebook | ✅ Ready |
| `deepfake_model/notebooks/02_training.py` | Colab training notebook | ✅ Ready |

### Files to MODIFY

| File | Change Needed | Reason |
|------|--------------|--------|
| `deepfake_model/src/inference/__init__.py` | Will need a **new** inference server built for the DINOv2 architecture | Currently empty — this is where the real inference server goes |

### Reusable Utilities from the Temp Implementation

These elements from `inference_server.py` are **architecture-agnostic** and should be carried forward into the new inference server:

| Utility | Lines | Why Reusable |
|---------|-------|-------------|
| `extract_frames()` | 227-244 | Video → PIL frames (identical logic needed for new model) |
| `image_to_base64()` | 247-251 | PIL → base64 for JSON transport (needed for heatmaps) |
| `clamp()` | 104-106 | Score clamping (good UX practice — no 0% or 100%) |
| `get_verdict()` | 108-113 | Three-tier verdict logic (AUTHENTIC/SUSPICIOUS/MANIPULATED) |
| `get_confidence()` | 115-118 | Confidence calculation |
| FastAPI structure | 42-49, 256-264 | App setup, CORS, health endpoint |
| Video analysis loop | 331-369 | Frame-by-frame analysis with timeline output |
| Threshold constants | 101-102 | SUSPICIOUS_THRESHOLD, MANIPULATED_THRESHOLD |

### Frontend — NO Changes Needed

| File | Why |
|------|-----|
| `src/pages/DeepfakeForensics.jsx` | The frontend is **model-agnostic** — it sends a file to an API and displays results. The API response format (verdict, confidence, streams, heatmap, timeline) should stay the same. |
| `src/pages/DeepfakeForensics.css` | Pure styling — no model dependency |

### Backend — NO Changes Needed

| File | Why |
|------|-----|
| `backend/server.js` (deepfake proxy, lines 1338-1384) | The proxy just forwards the file to port 8001 and returns the JSON. It doesn't care what model runs on 8001. |

---

## 3. Anti-Shortcut Learning Strategy — Confirmed Found

> [!IMPORTANT]
> The Anti-Shortcut Learning strategy is **fully implemented** in your codebase. Here are all references:

### Implementation in `augmentations.py`

| Function | Lines | What It Does |
|----------|-------|-------------|
| `get_anti_shortcut_transforms()` | 106-126 | Standalone anti-shortcut pipeline: fixed resize → JPEG recompression (q60-95) → CLAHE color normalization |
| `compose_training_transforms()` | 129-177 | **Full pipeline**: anti-shortcut preprocessing FIRST → then regular augmentation → normalize → tensor |
| `compose_chaos_transforms()` | 180-213 | **Chaos pipeline**: anti-shortcut preprocessing FIRST → then extreme augmentation |

### The three anti-shortcut techniques:
1. **Fixed 256×256 resolution** (line 146) — eliminates resolution-based shortcuts
2. **Random JPEG recompression** at q60-95 (line 147) — destroys dataset-specific compression fingerprints
3. **CLAHE color normalization** (line 148) — removes camera/dataset color bias

### How it integrates into training:
- `train.py` line 289: `train_tf = compose_training_transforms(config['face_size'])` — anti-shortcut is applied **automatically** as the first stage of every training transform
- The curriculum system (`get_curriculum_transform()`, lines 63-78) switches between `compose_training_transforms` and `compose_chaos_transforms` — **both include anti-shortcut as their first stage**
- Validation uses `get_val_transforms()` — intentionally NO anti-shortcut (you want to test on clean data)

### References in documentation:
- `deepfake_proposal.md` Section 4.2 — full explanation
- `DEEPFAKE_MODEL_HANDOFF.md` Section 12 — detailed strategy
- `final_implementation_plan.md` — mentioned as critical for generalization
- `architecture_overview.md` Step 3 — included in pipeline diagram
- `project_accuracy_notes.md` — referenced in training context
- `fyp_battle_plan.md` — rated as 🔴 CRITICAL impact factor

**Bottom line: Anti-shortcut is code-complete and will automatically activate when training begins.** No additional work needed.

---

## 4. Training Strategy Recommendations

### Should you buy Colab Pro?

**Yes, absolutely.** Here's why it's non-negotiable for your timeline:

| Factor | Colab Free (T4, 15GB) | Colab Pro (A100, 40GB) |
|--------|----------------------|------------------------|
| DINOv2-Base forward pass | ~45ms/image | ~15ms/image |
| Training 15 epochs (image, batch=16) | ~10 hours | **~3-4 hours** |
| Video training (16 frames, batch=4) | **Likely OOM** | ~5-6 hours |
| Session length | 12 hrs (often disconnects at 4-6) | 24 hrs stable |
| RAM | 12.7 GB | 51 GB |

With 1-2 days, you **cannot afford** 10-hour training runs or OOM crashes. The $12 buys you 3x speed and eliminates the OOM risk entirely.

### Estimated Training Times (Colab Pro A100)

| Training Phase | Estimated Time | What You Get |
|----------------|---------------|-------------|
| **Stage 1: Image model** (15 epochs on FF++ c23 face crops) | **3-4 hours** | Frame-level AUC, working image model |
| **Stage 2: Video fine-tune** (10 epochs on DFDC/FF++ videos) | **4-6 hours** | Temporal reasoning, video detection |
| **Cross-dataset evaluation** (trained model → CelebDF-v2) | **30-60 min** | Generalization proof (the critical metric) |
| **Total** | **~8-11 hours** | Full pipeline with cross-dataset numbers |

### The Fastest Practical Training Strategy

Given 1-2 days, here is the optimal plan:

#### Option A: Maximum Speed (1 day available)
**Skip Stage 2 (video). Train image-only. Get cross-dataset AUC.**

| Step | Time | What |
|------|------|------|
| 1. Upload FF++ face crops to Google Drive | 30 min | Data prep |
| 2. Run `train.py --stage 1 --epochs 12 --datasets ff++` | 3 hrs | Image training |
| 3. Cross-eval on CelebDF-v2 | 30 min | Generalization proof |
| 4. Build new inference server | 2 hrs | Connect trained model to frontend |
| **Total** | **~6 hours** | Working demo with YOUR model + real AUC numbers |

#### Option B: Full Pipeline (2 days available)
**Train image model → video fine-tune → cross-eval → new inference server.**

| Step | Time | What |
|------|------|------|
| 1. Upload datasets to Google Drive | 1 hr | FF++ crops + DFDC subset |
| 2. `train.py --stage 1 --epochs 15 --datasets ff++ dfdc` | 4 hrs | Image training (overnight) |
| 3. `train.py --stage 2 --epochs 10 --checkpoint stage1_best.pth` | 5 hrs | Video fine-tune |
| 4. Cross-eval on CelebDF-v2 | 30 min | Generalization proof |
| 5. Build new inference server for DINOv2 model | 2-3 hrs | Connect to frontend |
| **Total** | **~12-13 hours** | Full pipeline with video + cross-dataset |

### Key Optimizations & Shortcuts

1. **Use pre-extracted face crops** — If your FF++ dataset is already face-cropped (many Kaggle versions are), skip the `FaceExtractor` step entirely. Saves 3-6 hours.

2. **Reduce epochs for first run** — 12-15 epochs instead of 25. Your curriculum will still cover Easy + Mixed phases. You can always resume later.

3. **Start with image-only** — Get Stage 1 working and cross-dataset numbers first. This alone is impressive. Stage 2 (video) is a bonus.

4. **Use `--max_samples` for debugging** — Before the real run, do `--max_samples 500 --epochs 2` to verify the entire pipeline works (data loading, training loop, checkpointing, metrics). Takes 5 minutes, saves hours of debugging.

5. **The checkpoint resume feature is critical** — `train.py` saves `stage1_latest.pth` every epoch. If Colab disconnects, `--resume stage1_latest.pth` picks up exactly where you left off.

6. **Don't train on 4 datasets** — Use FF++ c23 as primary. Add DFDC if time allows. Reserve CelebDF-v2 for cross-dataset evaluation only.

7. **Batch size matters** — On A100 (40GB), try `--batch_size 32` for image training. If it fits, training is ~2x faster than batch=16 due to better GPU utilization.

---

## 5. Summary of Proposed Changes

| # | Action | File(s) | Type |
|---|--------|---------|------|
| 1 | **Rename** `inference_server.py` → `inference_server_temp.py` | `deepfake_model/inference_server.py` | Disable (preserve) |
| 2 | **Rename** `inference_requirements.txt` → `inference_requirements_temp.txt` | `deepfake_model/inference_requirements.txt` | Disable (preserve) |
| 3 | All original architecture code | `deepfake_model/src/**` | **No changes** — already ready |
| 4 | Training script | `deepfake_model/train.py` | **No changes** — already ready |
| 5 | Frontend | `src/pages/DeepfakeForensics.jsx` | **No changes** |
| 6 | Backend proxy | `backend/server.js` | **No changes** |
| 7 | The 282MB checkpoint | `resnetinceptionv1_epoch_32.pth` | **No changes** — keep for reference |

> [!NOTE]
> The rename approach is cleaner than commenting out 375 lines. The `_temp` suffix makes it crystal clear what's temporary vs. permanent. The file is fully preserved and can be restored instantly by renaming back.

**Awaiting your approval before proceeding.**
