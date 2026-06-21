# Deepfake Video Detection — Final Technical Decisions & Implementation Plan

---

## 1. Model Architecture Decision: Keep DINOv2 + LoRA

### Comparison Table

| Factor | DINOv2-Base + LoRA | EfficientNet-B4 | ViT-Base (supervised) | Swin-T |
|--------|-------------------|-----------------|----------------------|--------|
| Pretraining | Self-supervised (142M images) | Supervised ImageNet | Supervised ImageNet | Supervised ImageNet |
| Task bias | **None** — no classification labels | Classification bias | Classification bias | Classification bias |
| Trainable params (w/ adaptation) | ~500K (LoRA) | ~19M (full fine-tune) or ~1M (head only) | ~86M or ~500K (LoRA) | ~29M |
| Patch-level features | ✅ 256 × 768 tokens | ❌ Only final feature map | ✅ But less rich | ✅ Multi-scale |
| Generalization to unseen fakes | **Best** — no supervised bias | Good but ImageNet-biased | Moderate | Moderate |
| VRAM (batch=8, 256×256) | ~6GB | ~3GB | ~5GB | ~4GB |
| Deepfake detection papers using it | Growing rapidly (2024-2026) | DFDC challenge winner (2020) | Some papers | Few papers |
| Compatible with your attention stream | ✅ Patch tokens feed directly | ❌ Would need rewrite | ✅ But weaker tokens | ✅ But different dims |

### My Decision: **DINOv2-Base + LoRA. Do not change.**

**Why DINOv2 wins for YOUR project:**

1. **Self-supervised = no task bias.** This is THE reason. Supervised models learn "this looks like a dog/car/person" features. DINOv2 learns "what makes visual patterns" without any label bias. For deepfake detection, you want the model to detect subtle inconsistencies, not classify objects. Self-supervised features are strictly better for this.

2. **Patch tokens power your attention stream.** DINOv2 produces 256 patch tokens at 768-dim each. Your `AttentionForgeryStream` cross-attends to these patches with 4 region queries. If you switch to EfficientNet, you lose this entirely — EfficientNet produces a feature map, not the rich token sequence your architecture depends on.

3. **LoRA = overfitting protection.** With only ~500K trainable params on the backbone side, you cannot overfit to dataset-specific artifacts. Full fine-tuning of EfficientNet-B4 (19M params) on small datasets like FF++ is a known recipe for poor cross-dataset performance.

4. **Architecture is peer-reviewed and built.** Your PhD-level friend rated it 9/10. All the code is written. Changing the backbone now means rewriting `spatial_stream.py`, `attention_stream.py`, potentially `adaptive_fusion.py`, and both notebooks. That's 2-3 days of work you don't have.

**Why NOT EfficientNet:**
- The DFDC challenge winner used EfficientNet-B4, but that was 2020. The field has moved to ViT-based approaches.
- EfficientNet gives you no patch tokens → your attention stream becomes useless.
- Full fine-tuning EfficientNet on small datasets leads to severe overfitting.

**Why NOT a hybrid CNN + Transformer:**
- Your architecture IS already a hybrid. The frequency stream is a CNN. The spatial stream is a Transformer. The fusion MLP combines both. Adding another CNN backbone alongside DINOv2 would double VRAM usage and add complexity with questionable benefit.

> [!IMPORTANT]
> **Final answer: Keep DINOv2-Base + LoRA. Zero changes to the backbone. This is the correct choice for generalization, explainability, and your timeline.**

---

## 2. Dataset Strategy — Updated Understanding

Your corrected dataset situation:

| Dataset | Status | Size | Format | Role |
|---------|--------|------|--------|------|
| FaceForensics++ | ✅ Available (Kaggle, pre-cropped faces) | Manageable | Face crops (`.jpg`) | **Training** |
| DFDC | ✅ Partial (~20GB subset) | ~20GB | Needs face extraction from videos | **Training** |
| Celeb-DF v2 | ✅ Available (Google Drive link) | ~20GB | Downloadable | **Cross-dataset test ONLY** |
| Modern dataset | 🟡 Link pending | TBD | TBD | **Training** (if time allows) |

This changes the plan significantly — the data bottleneck is largely solved for FF++ and CelebDF.

### DFDC Note
DFDC is raw videos, not pre-extracted face crops. You'll need to run face extraction on the DFDC subset. With Colab Pro GPU, face extraction on ~20GB of DFDC videos should take ~2-3 hours.

For FF++ pre-cropped faces from Kaggle — verify the folder structure matches what your `dataset.py` expects: `ff++/real/*.jpg` and `ff++/fake/*.jpg`. If the Kaggle version uses a different structure, you'll need a quick reorganization script.

---

## 3. Video vs Image Training Strategy — DEFINITIVE ANSWER

### My Decision: **Option C — Two-stage hybrid (frame model → temporal aggregation)**

This is NOT a compromise. This is the approach used by every top-performing video deepfake detector:

| Paper / System | Approach | Result |
|---------------|----------|--------|
| DFDC Challenge Winner (Seferbekov, 2020) | EfficientNet per-frame → mean pooling | 1st place |
| FTCN (Zheng et al., 2021) | CNN per-frame → Temporal Transformer | State-of-the-art cross-dataset |
| LipForensics (Haliassos et al., 2021) | ResNet per-frame → Temporal network | Strong generalization |
| AltFreezing (Wang et al., 2023) | ViT per-frame → Temporal module | CVPR 2023 |
| Your Architecture | DINOv2+LoRA per-frame → Temporal Transformer | Same paradigm ✅ |

### Why Option C, not Option B (end-to-end video from scratch):

| Factor | Option B (end-to-end) | Option C (two-stage) |
|--------|----------------------|---------------------|
| VRAM per batch | 16 frames × full forward = ~24GB+ for batch=2 | Stage 1: normal image training. Stage 2: lightweight temporal | 
| Debugging | If AUC is bad, is it spatial features or temporal? Unknown. | Stage 1 validates spatial features independently |
| Training time | 3-5x longer per epoch | Stage 1 converges fast on images, Stage 2 is quick |
| Literature support | Almost no papers do this | Every top paper uses this approach |
| Your architecture compatibility | Your code already supports this | Your code already supports this |

### The Two-Stage Training Plan:

**Stage 1: Frame-Level Model (image mode)**
```
DeepfakeForensicsModel(video_mode=False)
```
- Train on FF++ face crops + DFDC face crops (individual images)
- 10-15 epochs on Colab Pro A100
- ~3-4 hours training time
- Validate: expect ≥ 0.90 AUC on in-dataset val set
- Save best checkpoint

**Stage 2: Temporal Fine-Tuning (video mode)**
```
DeepfakeForensicsModel(video_mode=True)
# Load Stage 1 weights for all layers EXCEPT temporal transformer
# Lower learning rate for frame-level modules (already learned)
# Higher learning rate for temporal transformer (learning from scratch)
```
- Load Stage 1 checkpoint into the frame-level components
- Train with video sequences (16 frames per video)
- Use DFDC video data (since it has actual videos, not just crops)
- 8-10 epochs, lower LR for spatial/frequency/attention, normal LR for temporal
- ~3-4 hours training time
- Save best video model checkpoint

**Why this is the strongest approach for FYP evaluation:**
1. You get a working image model first (safety net)
2. You then upgrade it to video (the impressive part)
3. You can show the evaluator: "Frame-level AUC was X, but with temporal reasoning it improved to Y"
4. You can explain the two-stage approach as a deliberate training strategy (which it is)
5. The ablation (image vs video) itself IS a result worth presenting

---

## 4. Dataset Training Strategy — My Honest Evaluation

### Your friend's suggestion: Train on FF++ + DFDC + Modern → Cross-validate on CelebDF

**Verdict: This is correct. Keep it.**

Here's why and how to refine it:

### Training Set Composition

| Dataset | Role in Training | Contribution |
|---------|-----------------|--------------|
| **FF++ (c23)** | Primary training | 4 manipulation methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures). Teaches the model diverse forgery types |
| **DFDC (subset)** | Secondary training | Real-world diversity — varied subjects, lighting, backgrounds, compression. Adds robustness |
| **Modern dataset** | Tertiary training | Latest-generation deepfakes. Fills the gap where FF++ and DFDC are outdated |

### CelebDF-v2: Testing ONLY — Do NOT Train On It

> [!CAUTION]
> **CelebDF-v2 must be held out entirely for cross-dataset evaluation. Do NOT include it in training.**

**Why:**
- Cross-dataset AUC (train on A, test on B) is THE metric that proves generalization
- If you train on CelebDF, you can only report in-dataset AUC, which is far less impressive
- Every deepfake detection paper reports cross-dataset results. Evaluators expect this.
- "We trained on FF++ and DFDC, then tested on CelebDF-v2 WITHOUT any fine-tuning and achieved X AUC" — this is a powerful statement

### How to Mix Datasets Without Overfitting

**Problem:** FF++ has ~80K face crops, DFDC subset might have ~30K, modern dataset is unknown. If you naively combine them, the larger dataset dominates training.

**Solution: Dataset-balanced sampling.**

```python
# Per-batch sampling strategy:
# - 40% of each batch from FF++
# - 40% from DFDC
# - 20% from modern dataset (if smaller)
# This ensures equal dataset exposure regardless of dataset size
```

Implementation: Use a custom sampler or oversample smaller datasets. Your existing `WeightedRandomSampler` in `dataset.py` balances real/fake — you also need to balance across datasets.

### Real/Fake Class Balance

- FF++ is roughly 1:4 real:fake ratio (1000 real, 4000 fake videos)
- DFDC is roughly 1:1
- CelebDF is roughly 1:10

**Use your `WeightedRandomSampler`** (already written in `dataset.py`) to ensure each batch is ~50% real, ~50% fake. This is critical — without it, the model learns to just predict "fake" for everything and gets 80% accuracy from class imbalance.

---

## 5. What Actually Maximizes FYP Marks

### Evaluator Psychology

| What They See | What They Think | Score Impact |
|---------------|----------------|-------------|
| Upload video → real-time frame analysis → verdict with heatmap | "This actually works. Impressive." | **+++ (30-40%)** |
| Cross-dataset AUC number (e.g., 0.83 on unseen CelebDF) | "This generalizes. Not just memorization." | **++ (20-25%)** |
| "We use DINOv2 because self-supervised features avoid task bias..." | "This student understands the WHY, not just the HOW." | **++ (20-25%)** |
| Clean architecture diagram + ablation results | "Well-structured research methodology." | **+ (10-15%)** |
| Frame-level timeline showing suspicious frames in a video | "This is video detection, not just image detection." | **++ (bonus)** |

### What Will Lose You Marks

| Mistake | Score Impact |
|---------|-------------|
| Demo crashes during evaluation | **--- (catastrophic)** |
| "Video detection is coming in Phase 2" (i.e., not done) | **-- (major)** |
| Only show image detection | **- (missed requirement)** |
| No cross-dataset results | **- (no generalization evidence)** |
| Can't explain why you chose DINOv2 over alternatives | **- (shallow understanding)** |

---

## 6. Critical Implementation Fixes Needed

### Fix 1: Video Dataset Loader (MUST BUILD)

Your current `DeepfakeDataset` treats every face crop as independent. For video training, you need frame sequences grouped by source video.

**For pre-extracted face crops (FF++):**
The crops need to be organized so you know which frames belong to the same video. The Kaggle FF++ dataset typically names files like `000_003_frame_42.jpg` — the video ID groups them.

**For DFDC:**
You'll extract faces from videos using `FaceExtractor.extract_from_video()`, which already saves crops as `{video_name}_frame{idx:03d}.jpg`.

**The video dataset needs to:**
1. Group face crops by source video (parse filename to get video ID)
2. Sample T=16 frames uniformly from each video
3. Return `(T, 3, 256, 256)` tensor + video-level label
4. Handle videos with fewer than 16 extracted faces (pad + mask)

### Fix 2: Training Loop for Video Mode

The current training loop in `02_training.py` passes `(B, 3, 256, 256)` images. Video mode needs `(B, T, 3, 256, 256)`. The loop needs:
- Video-aware batching (collate function for variable-length sequences)
- Padding mask passed to the model's `forward()` 
- Stage 2 learning rate setup (lower LR for pretrained frame modules, higher for temporal)

### Fix 3: Two-Stage Training Script

Need a proper `train.py` with:
- `--stage 1` (image) vs `--stage 2` (video) mode
- `--resume` checkpoint loading (critical for Colab disconnects)
- `--datasets` selection
- Proper train/val split that DOESN'T reload all data twice (bug in current `create_dataloaders`)

### Fix 4: Inference Server

Need `deepfake_model/src/inference/server.py`:
- FastAPI with `/detect` POST endpoint
- Accept image OR video upload
- For video: extract frames → face detection → model inference → aggregate predictions
- Return JSON with verdict, confidence, heatmap, per-frame timeline

---

## 7. Refined 5-Day Battle Plan

### Day 1 — Tonight + Tomorrow Morning (June 14-15) 🔴 FOUNDATION

**Evening (3-4 hours):**

| Task | Time | Priority |
|------|------|----------|
| Buy Colab Pro | 5 min | 🔴 |
| Upload FF++ face crops to Google Drive (or link Kaggle dataset) | 30 min | 🔴 |
| Download CelebDF-v2 to Google Drive | Background task | 🔴 |
| Verify model loads on Colab Pro GPU (notebook cells 1-7) | 20 min | 🔴 |
| Fix `dataset.py` data loading bugs | 45 min | 🔴 |
| Reorganize FF++ crops into `data/ff++/real/` and `data/ff++/fake/` structure | 30 min | 🔴 |
| Start image training on FF++ (Stage 1) — let run overnight | 15 min setup | 🔴 |

**Overnight: Image model trains on Colab Pro A100 (~10-15 epochs, ~3-4 hrs)**

---

### Day 2 — June 15 🔴 VIDEO PIPELINE + INFERENCE

**Morning (4-5 hours):**

| Task | Time | Priority |
|------|------|----------|
| Check overnight training results (expect val AUC > 0.85) | 15 min | 🔴 |
| Save best image model checkpoint | 5 min | 🔴 |
| Build `VideoDeepfakeDataset` class | 2 hrs | 🔴 |
| Run face extraction on DFDC subset (for video training data) | 2-3 hrs (background) | 🔴 |
| Write two-stage training script with `--resume` support | 1.5 hrs | 🔴 |

**Afternoon/Evening (4-5 hours):**

| Task | Time | Priority |
|------|------|----------|
| Start Stage 2 video training (load Stage 1 weights, train temporal module) | 15 min setup | 🔴 |
| Build FastAPI inference server (`server.py`) | 2-3 hrs | 🔴 |
| Connect `DeepfakeForensics.jsx` frontend to inference server | 1-2 hrs | 🟡 |

**Overnight: Video model trains (~8-10 epochs, ~3-4 hrs)**

---

### Day 3 — June 16 🟡 EVALUATION + DEMO POLISH

**Morning:**

| Task | Time | Priority |
|------|------|----------|
| Check video training results | 15 min | 🔴 |
| Run cross-dataset evaluation: trained model → CelebDF-v2 test set | 1 hr | 🔴 |
| Generate evaluation metrics: AUC, accuracy, F1, EER, confusion matrix | 1 hr | 🔴 |
| Generate plots: ROC curve, training curves, per-dataset performance | 1 hr | 🟡 |

**Afternoon:**

| Task | Time | Priority |
|------|------|----------|
| Test full video demo: upload video → frame extraction → analysis → verdict + heatmap | 2 hrs | 🔴 |
| Build frame-level timeline visualization (which frames are suspicious) | 1-2 hrs | 🟡 |
| Quick ablation: spatial-only vs full multi-stream (shows each stream's value) | 1 hr | 🟡 |
| Polish demo UI: loading animations, error handling, clean results display | 1-2 hrs | 🟡 |

---

### Day 4 — June 17 🟡 PREPARATION

**DO NOT write new code today unless something is broken.**

| Task | Time | Priority |
|------|------|----------|
| Final demo rehearsal — run full flow 3 times with different test videos | 1 hr | 🔴 |
| Prepare talking points (why DINOv2, why 3 streams, why anti-shortcut, why two-stage) | 1.5 hrs | 🔴 |
| Prepare 5-6 test cases: real video, obvious fake, subtle fake, compressed fake | 1 hr | 🔴 |
| Prepare answers for evaluator questions (from handoff doc Section 18) | 1 hr | 🟡 |
| If modern dataset is ready: quick training run on combined data (bonus) | 2-3 hrs | 🟢 |
| **Sleep well** | 8 hrs | 🔴🔴🔴 |

---

### Day 5 — June 18 (External Evaluation) ⭐

- Demo: video upload → detection → heatmap → frame timeline → verdict
- Present cross-dataset AUC numbers
- Explain architecture decisions with confidence
- Note evaluator feedback for final display (Day 8)

---

## Summary: Final Decisions

| Decision | Answer | Confidence |
|----------|--------|------------|
| Backbone model? | **DINOv2-Base + LoRA. Do not change.** | 98% |
| Training approach? | **Two-stage: image model first → video fine-tune** | 95% |
| CelebDF in training? | **No. Test-only for cross-dataset evaluation.** | 99% |
| Dataset mixing? | **FF++ + DFDC + Modern (balanced sampling). Test on CelebDF.** | 92% |
| Colab Pro? | **Yes, buy immediately.** | 99% |
| Architecture changes? | **None. Freeze the architecture.** | 99% |
| Priority #1? | **Working video demo that doesn't crash** | 100% |
| Priority #2? | **Cross-dataset AUC number on CelebDF** | 95% |
| Modern 4th dataset? | **Add if time allows after Day 3. Do not delay core work for it.** | 85% |

> [!IMPORTANT]
> **The path to maximum FYP marks: Working video demo + cross-dataset AUC + ability to explain every design decision. In that order.**
