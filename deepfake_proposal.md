# Deepfake Detection Module — Technical Approach

**Project**: Silent Trails — Profiling Human Patterns Across the Digitalverse
**Module**: Deepfake Forensics
**Prepared by**: Faheem

---

## 1. Problem Statement

Current deepfake detectors suffer from poor generalization — they score well on the same dataset they trained on but perform badly on new, unseen data. This happens because most models learn dataset-specific shortcuts (resolution patterns, compression artifacts from a particular dataset) rather than understanding what actually makes a face manipulated. My goal is to build a detector that works reliably across different forgery methods and real-world conditions like social media compression.

---

## 2. Proposed Architecture

I am building a multi-stream detection system. Instead of feeding an image through a single CNN and hoping it catches everything, I split the analysis into separate specialized pathways, each designed to capture a different type of forensic evidence.

### 2.1 Stream 1 — Spatial Analysis

The spatial stream handles appearance-level features: skin texture discontinuities, blending boundaries, unnatural smoothing, and lighting inconsistencies.

**Backbone**: I am using DINOv2-Base (Oquab et al., 2023) as a frozen feature extractor. DINOv2 is a self-supervised Vision Transformer trained on 142M curated images by Meta AI. Since it has learned very strong general-purpose visual representations, I do not want to fine-tune all its weights — that would risk overfitting to training-set-specific patterns.

**Adaptation via LoRA**: Instead of full fine-tuning (~86M parameters), I apply Low-Rank Adaptation (Hu et al., 2021) on the query and value projection matrices of each transformer block. This introduces only ~500K trainable parameters. LoRA lets the backbone adapt to the forensics domain without losing its broad visual understanding.

**Output**: A 512-dim spatial embedding + patch-level features (256 tokens × 768 dim) passed to the attention stream.

### 2.2 Stream 2 — Frequency Analysis

Many forgery methods leave telltale patterns in the frequency domain that are not visible in pixel space. GAN-based generators produce periodic checkerboard artifacts due to transposed convolutions. Diffusion-based generators leave a different kind of spectral fingerprint related to their denoising process.

**Method**: I convert each face crop to grayscale, apply 2D FFT, center-shift, and take the log-magnitude spectrum. This spectrum image is then processed through a lightweight 4-block CNN (Conv → BN → GELU → Pool) to produce a 512-dim frequency embedding.

### 2.3 Stream 3 — Attention-based Forgery Localization

The spatial stream analyzes the whole face holistically. The attention stream complements it by focusing specifically on facial regions where manipulation boundaries typically appear — around the eyes, mouth, jawline, and hairline.

**Method**: I define 4 learnable region query vectors. These queries attend to the DINOv2 patch tokens via multi-head cross-attention (8 heads). The attention weights naturally learn to focus on forgery-sensitive regions after training.

**Explainability**: The attention weight map can be reshaped and upsampled to produce a heatmap showing WHERE manipulation was detected — providing interpretability without a separate Grad-CAM step.

### 2.4 Stream Fusion

The three stream embeddings (each 512-dim) are concatenated into a 1536-dim vector and passed through a gating MLP that learns to combine them into a single 512-dim fused embedding. The network implicitly learns which stream to trust more based on the input characteristics.

---

## 3. Implementation Phases

The system is implemented in three phases, prioritizing a working core model before adding complexity.

### Phase 1 — Core Image Model
- DINOv2 + LoRA (spatial stream)
- FFT + CNN (frequency stream)
- Region-query cross-attention (attention stream)
- Simple MLP fusion
- BCE + Focal Loss
- Anti-shortcut augmentation
- Cross-dataset evaluation on FF++, CelebDF-v2

### Phase 2 — Video Support
- Temporal Transformer (4-layer encoder with learned positional embeddings)
- Uniform frame sampling (16 frames per video)
- Temporal consistency analysis: flickering detection, identity drift, blinking anomalies
- Evaluation on DFDC video dataset

### Phase 3 — Advanced Enhancements *(see Section 7)*
- Adaptive quality-weighted fusion
- Metric learning via triplet loss
- Additional optimizations based on Phase 1–2 results

---

## 4. Training Strategy

### 4.1 Datasets

| Dataset | Source | Content | Purpose |
|---------|--------|---------|---------|
| FaceForensics++ | Rössler et al., 2019 | 1000 real + 4000 fake videos (4 methods), c23 and c40 compression | Primary training set |
| CelebDF-v2 | Li et al., 2020 | 590 real + 5639 fake celebrity videos, high visual quality | Cross-dataset evaluation |
| DFDC | Dolhansky et al., 2020 | 100K+ clips, diverse subjects, varied conditions | Phase 2 video evaluation |

### 4.2 Data Pipeline

1. **Face extraction** — RetinaFace (Deng et al., 2020) for detection. Faces aligned via similarity transform using 5-point landmarks. Output: 256×256 crops. Discard faces with detection confidence below 0.7.

2. **Anti-shortcut preprocessing** — To prevent the model from learning dataset-specific shortcuts:
   - All crops resized to a fixed 256×256 (eliminates resolution-based shortcuts)
   - Random JPEG recompression at quality 60-95 (removes dataset-specific compression fingerprints)
   - Color normalization via CLAHE (handles different camera color profiles)

3. **Augmentation** — During training: random horizontal flip, color jitter, Gaussian blur (σ 0-3), Gaussian noise (σ 0-15), JPEG compression (q 30-95), downscale-upscale cycles, and coarse pixel dropout.

### 4.3 Curriculum Training

| Phase | Epochs | Strategy |
|-------|--------|----------|
| Easy | 1–8 | Clean, obvious fakes only — model learns basic forgery patterns |
| Mixed | 9–18 | Full dataset with moderate augmentation |
| Chaos | 19–25 | Aggressive multi-degradation stacking — builds robustness |

### 4.4 Loss Function

- **Binary Cross-Entropy** (weight 1.0) — standard classification loss
- **Focal Loss** (weight 0.5, γ=2) — reduces contribution of easy samples, focuses learning on hard-to-classify examples (Lin et al., 2017)

### 4.5 Optimization

- Optimizer: AdamW (proper weight decay decoupling)
- Learning rates: 5e-4 for LoRA parameters, 1e-3 for new modules
- DINOv2 backbone: frozen (LR = 0)
- Scheduler: Cosine annealing with warm restarts
- Precision: FP16 mixed-precision
- Early stopping on validation AUC

---

## 5. Evaluation Plan

### 5.1 Metrics
- AUC-ROC (primary — threshold-independent)
- Accuracy, F1
- Equal Error Rate (EER)

### 5.2 Benchmark Targets

| Evaluation | Target AUC | What it proves |
|------------|-----------|----------------|
| FF++ c23 (in-dataset) | ≥ 0.97 | Basic detection capability |
| FF++ c40 (heavy compression) | ≥ 0.90 | Robustness to JPEG degradation |
| CelebDF-v2 (in-dataset) | ≥ 0.93 | Performance on high-quality fakes |
| FF++ → CelebDF (cross-dataset) | ≥ 0.85 | Generalization — most critical test |
| Post-JPEG q=30 | AUC drop ≤ 5% | Social media survival |

### 5.3 Ablation Study

Systematic evaluation of each component's contribution:
1. Spatial stream only (baseline)
2. Spatial + Frequency
3. Spatial + Frequency + Attention (full Phase 1)
4. Full model with temporal (Phase 2)

---

## 6. Deployment

The trained model will be served through a FastAPI endpoint integrated with the existing Silent Trails web application. Users can upload images or videos through the frontend, and the model returns:
- Real/Fake classification with confidence score
- Attention heatmap showing manipulation regions
- Frame-level timeline for videos (Phase 2)

---

## 7. Future Enhancements

The following techniques are designed into the architecture but deferred to Phase 3 to manage implementation complexity:

### 7.1 Adaptive Quality-Weighted Fusion
Replace the simple MLP fusion with a quality-aware fusion engine. An auxiliary Quality Estimator network predicts input degradation levels (compression, blur, motion) and dynamically reweights stream contributions. This would handle edge cases where specific streams are unreliable due to input quality.

### 7.2 Metric Learning via Triplet Loss
Add a metric learning head alongside the classification head. Triplet loss pushes real-face embeddings into a tight cluster while separating them from fake-face embeddings. This would improve generalization to completely unseen forgery methods by learning a general "real vs fake" concept in embedding space rather than method-specific artifacts.

### 7.3 Audio-Visual Lip Sync Analysis
An optional SyncNet-based module that compares mouth movements with audio to detect lip-sync deepfakes. This would extend detection coverage to audio-driven manipulation methods.

---

## 8. Key References

- Oquab, M. et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. *arXiv:2304.07193*
- Hu, E. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv:2106.09685*
- Rössler, A. et al. (2019). FaceForensics++: Learning to Detect Manipulated Facial Images. *ICCV 2019*
- Li, Y. et al. (2020). CelebDF: A Large-scale Challenging Dataset for DeepFake Forensics. *CVPR 2020*
- Dolhansky, B. et al. (2020). The DeepFake Detection Challenge Dataset. *arXiv:2006.07397*
- Deng, J. et al. (2020). RetinaFace: Single-shot Multi-level Face Localisation in the Wild. *CVPR 2020*
- Lin, T. et al. (2017). Focal Loss for Dense Object Detection. *ICCV 2017*
