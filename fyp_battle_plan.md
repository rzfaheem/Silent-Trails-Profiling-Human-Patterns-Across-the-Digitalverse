# 🚨 FYP Battle Plan — 4 Days to Externals, 8 Days to Display

> **Date**: June 14, 2026
> **Situation**: Model code scaffolded. Zero training done. No datasets downloaded. No inference server. Frontend shows placeholder.
> **External Eval**: June 18 (4 days)
> **Final Display**: June 22 (8 days)

---

## The Brutal Truth First

Let me be completely honest: **you cannot train a research-grade multi-stream deepfake model from scratch and achieve your 0.85+ cross-dataset AUC targets in 4 days.** No amount of GPU power changes this. Here's why:

| Bottleneck | Time Required | Can GPU Speed Fix It? |
|------------|--------------|----------------------|
| Dataset download (CelebDF = ~20GB, FF++ = ~100GB+) | 4-12 hours | ❌ No |
| Face extraction (thousands of videos → crops) | 6-12 hours | Partially |
| Training 25 epochs with curriculum | 8-20 hours | ✅ Yes |
| Debugging inevitable issues (OOM, data loading, bugs) | 4-8 hours | ❌ No |
| Building FastAPI inference server | 3-5 hours | ❌ No |
| Connecting frontend to API | 2-3 hours | ❌ No |
| **Total minimum** | **~30-60 hours** | — |

You have **96 hours until externals**. That's technically enough IF nothing goes wrong. But things ALWAYS go wrong in ML. **You need a strategy that guarantees a working demo on day 1, then improves from there.**

---

## 1. Should You Buy Colab Pro? — YES, But Know Why

> [!IMPORTANT]
> **Buy Colab Pro ($12/month). Cancel after one month.** The $12 is the best ROI you'll get in the next 8 days.

### What Colab Pro Actually Gives You

| Feature | Free Tier | Colab Pro ($12/mo) | Impact for You |
|---------|-----------|-------------------|----------------|
| GPU | T4 (15GB) | **A100 (40GB)** or V100 (16GB) | **2-4x faster training** |
| Session length | 12 hrs (often disconnects earlier) | **24 hrs** | Can run overnight without babysitting |
| RAM | 12.7 GB | **51 GB** | Larger batch sizes, no OOM crashes |
| Idle timeout | ~90 min | **~90 min** (same, but reconnects better) | Less babysitting |
| Priority | Low | **High** (less queueing for GPUs) | Get A100 when you need it |

### Real Speed Comparison

| Task | T4 (Free) | A100 (Pro) |
|------|-----------|------------|
| Training 1 epoch (batch=32, FF++ subset) | ~25 min | **~8 min** |
| 25 epochs full training | ~10 hrs | **~3-4 hrs** |
| Face extraction (1000 videos) | ~3 hrs | ~2 hrs |
| DINOv2 forward pass | ~45ms/image | **~15ms/image** |

**Verdict**: Colab Pro turns a 10-hour training run into a 3-4 hour one. For $12, that's a no-brainer when you have 4 days. **But the GPU is not your biggest risk — your biggest risk is having nothing working to demo.**

---

## 2. The Strategy That Actually Works: TWO PARALLEL TRACKS

> [!CAUTION]
> **The #1 mistake students make**: spending all their time trying to get perfect training, then having NOTHING to demo on evaluation day. Evaluators grade what they SEE, not what you planned.

### Track A: GUARANTEED DEMO (Days 1-2) — Priority 🔴
Get a **working end-to-end deepfake detection demo** using a pretrained model as a fallback. This ensures that even if your custom training fails, you have something impressive to show.

### Track B: YOUR CUSTOM MODEL (Days 1-8) — Running in Parallel 🟡
Download data, train your multi-stream model on Colab Pro. If training succeeds, swap in your model. If it doesn't finish in time, you still have Track A.

**This is not cheating.** This is engineering. Real products use pretrained models all the time. You can explain your custom architecture to evaluators while demoing with a working system.

---

## 3. Biggest Factors Affecting Model Quality (Besides GPU)

Ranked by impact:

| # | Factor | Impact on Quality | Your Status |
|---|--------|------------------|-------------|
| 1 | **Data quality & diversity** | 🔴 CRITICAL — garbage in, garbage out | ❌ No data yet |
| 2 | **Anti-shortcut preprocessing** | 🔴 CRITICAL — the difference between 62% and 85%+ | ✅ Code written |
| 3 | **Training data volume** | 🟡 HIGH — more diverse fakes = better generalization | Depends on download |
| 4 | **Augmentation strategy** | 🟡 HIGH — curriculum training helps significantly | ✅ Code written |
| 5 | **Hyperparameters** | 🟢 MODERATE — your defaults are reasonable | ✅ Configured |
| 6 | **GPU speed** | 🟢 MODERATE — faster iteration, same final quality | Colab Pro fixes this |
| 7 | **Architecture changes** | 🔵 LOW — your architecture is already strong | ✅ Done |

> [!TIP]
> **Do NOT change your architecture.** It's already peer-reviewed and solid. Spend zero time on architecture modifications. Focus on data + training execution.

---

## 4. Where to Spend Your Time

### ✅ DO Focus On
- **Getting data downloaded and faces extracted** — this is the #1 bottleneck
- **Getting a working demo end-to-end** — upload → result on screen
- **Training with your existing code and defaults** — don't change hyperparameters unless training clearly diverges
- **The demo experience** — heatmap overlay, confidence score, clean UI

### ❌ DO NOT Spend Time On
- Hyperparameter tuning (your defaults are fine for a first run)
- Architecture modifications (your architecture is already strong)
- Adding new features (video mode, adaptive fusion — these are Phase 2/3)
- Perfecting the training pipeline (one good run > five tweaked runs)
- Writing a standalone `train.py` (use the notebook as-is)

---

## 5. Common Deadline Mistakes to Avoid

| Mistake | Why It's Deadly | What to Do Instead |
|---------|----------------|-------------------|
| Trying to achieve perfect AUC before building the demo | You end up with numbers but nothing to show | Demo first, then improve |
| Changing architecture at the last minute | Introduces bugs, wastes debugging time | Freeze the architecture NOW |
| Training on one dataset only | Evaluators ask "does it generalize?" and you can't answer | Use at least 2 datasets |
| Not having a fallback plan | If training fails at hour 70, you have nothing | Track A gives you a safety net |
| Spending hours on training curves/plots | Nice to have, not essential | Generate these AFTER you have a working model |
| Not sleeping | Your debugging ability drops 50% after 24 hrs | Sleep. Seriously. |

---

## 6. What Evaluators Actually Care About

Based on typical FYP external evaluations:

| What They Grade | Weight | How to Score High |
|-----------------|--------|-------------------|
| **Working demo** | 🔴 30-40% | Live upload → real result. Must work smoothly |
| **Technical depth** | 🟡 25-30% | Explain multi-stream architecture, anti-shortcut strategy, loss functions |
| **Results & evaluation** | 🟡 20-25% | AUC numbers, cross-dataset results, confusion matrix |
| **Documentation & presentation** | 🟢 10-15% | Clean slides, architecture diagram, code structure |

> [!IMPORTANT]
> A **working demo with 80% accuracy** scores HIGHER than a **broken demo with theoretical 95% accuracy**. Evaluators want to see it WORK.

---

## 7. The 8-Day Battle Plan

### Day 1 (Today — June 14) 🔴 CRITICAL

**Morning/Afternoon — Track A (Guaranteed Demo):**
- [ ] Get a pretrained deepfake detection model working locally or on Colab
  - Option: Use HuggingFace models (e.g., `dima806/deepfake_vs_real_faces_detection` or similar)
  - Option: Revisit the selimsef/dfdc_deepfake_challenge model you tested before
- [ ] Build a minimal FastAPI inference server that wraps this pretrained model
- [ ] Test: upload image → get real/fake verdict + confidence

**Evening — Track B (Your Custom Model):**
- [ ] Buy Colab Pro
- [ ] Start downloading CelebDF-v2 (Google Drive link, ~20GB) to Google Drive
- [ ] If FF++ access not yet requested, **email the authors NOW** (they respond in 1-3 days)
- [ ] Open Colab, run notebook cells 1-7 to verify your model code loads correctly

**End of Day 1 Checkpoint**: Pretrained model serving predictions via API ✅

---

### Day 2 (June 15)

**Morning — Track A (Connect Frontend):**
- [ ] Connect `DeepfakeForensics.jsx` to the FastAPI inference server
- [ ] Test full flow: upload in browser → see verdict + confidence on screen
- [ ] Generate a fake heatmap overlay (even if from pretrained model's attention, or a gradient-based method)

**Afternoon/Evening — Track B (Data Prep):**
- [ ] CelebDF-v2 should be downloaded by now
- [ ] Run `FaceExtractor` on CelebDF videos → save face crops to Google Drive
- [ ] Verify folder structure: `data/celebdf/real/` and `data/celebdf/fake/` with `.jpg` crops
- [ ] Start training on CelebDF with your multi-stream model (let it run overnight on Colab Pro)

**End of Day 2 Checkpoint**: Working demo in browser ✅ + Training started ✅

---

### Day 3 (June 16)

**Morning — Check Training:**
- [ ] Check training progress (should be 8-15 epochs by morning if started last night)
- [ ] If training crashed: debug, restart
- [ ] If training is running: let it continue

**Afternoon — Polish Demo:**
- [ ] Make the demo smooth: loading animations, error handling, clean result display
- [ ] Test with various image types (selfie, group photo, screenshot)
- [ ] Prepare 5-6 test images (mix of real and known fakes) for the evaluation demo

**Evening — Evaluation Prep:**
- [ ] If your model training finished: swap it into the inference server, test accuracy
- [ ] If still training: keep the pretrained model as demo, note your custom model's partial results
- [ ] Prepare talking points for evaluators (architecture, why multi-stream, anti-shortcut, etc.)

**End of Day 3 Checkpoint**: Polished demo ✅ + Custom model partially/fully trained ✅

---

### Day 4 (June 17 — Day Before Externals) 🔴

**DO NOT change code today unless something is broken.**

- [ ] Final demo rehearsal — run through the entire flow 3 times
- [ ] Prepare slides/talking points:
  - Architecture diagram (you already have the clean one)
  - Why DINOv2 + LoRA
  - Why 3 streams
  - Anti-shortcut strategy (this impresses evaluators)
  - Results (whatever AUC numbers you have)
- [ ] If custom model is trained: generate evaluation metrics (AUC, accuracy, confusion matrix)
- [ ] Prepare answers for likely questions (see Section 18 of your handoff doc)
- [ ] **Sleep well tonight**

**End of Day 4**: Ready for externals ✅

---

### Day 5 (June 18 — External Evaluation) ⭐

- [ ] Demo day. Show confidence. You know the architecture inside-out.
- [ ] After evaluation: note any feedback for improvements

---

### Days 6-7 (June 19-20 — Post-Externals Polish)

- [ ] If custom model wasn't ready for externals, swap it in now
- [ ] If FF++ access came through, train on FF++ for cross-dataset evaluation
- [ ] Run cross-dataset test: train on CelebDF → test on FF++ (or vice versa)
- [ ] Generate proper evaluation plots (ROC curve, confusion matrix, AUC comparison)
- [ ] Polish the frontend for display day (better heatmap visualization, frame timeline for video)
- [ ] Prepare the full FYP display materials

---

### Day 8 (June 22 — Final Display) ⭐

- [ ] Show the complete system: upload → face detection → 3-stream analysis → verdict + heatmap
- [ ] Present your results with confidence numbers
- [ ] Have the architecture diagram and evaluation metrics ready

---

## 8. What I Would Prioritize If I Were Your Supervisor

> [!IMPORTANT]
> **Priority #1**: A working, demoable system — end to end, upload to result
> **Priority #2**: Your custom model trained on at least one dataset with real AUC numbers
> **Priority #3**: Cross-dataset evaluation proving generalization
> **Priority #4**: Everything else (video mode, adaptive fusion, etc.)

The hardest thing for students to accept: **a working 80% system beats a theoretical 95% system that crashes during the demo.** Every. Single. Time.

---

## Summary Decision Matrix

| Decision | Recommendation | Confidence |
|----------|---------------|------------|
| Buy Colab Pro? | **Yes, immediately** ($12, cancel next month) | 95% |
| Change architecture? | **No, freeze it** | 99% |
| Train from scratch or use pretrained? | **Both — parallel tracks** | 90% |
| Focus on accuracy or demo? | **Demo first, accuracy second** | 95% |
| Try video mode (Phase 2)? | **No, skip for now** | 85% |
| All-nighter before externals? | **No, sleep > debugging at 3 AM** | 100% |
