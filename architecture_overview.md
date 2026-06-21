# Deepfake Forensics — Architecture Overview

![Detailed Architecture Diagram](C:/Users/Faheem.DESKTOP-MQLKQK1/.gemini/antigravity/brain/78cd20b7-5abf-4149-813a-cb11b2c2cc10/architecture_clean_1772597760145.png)

---

## Complete Pipeline (Every Step)

### Step 1: Input
User uploads **image** (JPG/PNG/WebP) or **video** (MP4/AVI) to Silent Trails.

### Step 2: Face Detection & Alignment (RetinaFace)
- **Detect** face location in the image/frame
- **Extract 5 landmarks** (2 eyes, nose tip, 2 mouth corners)
- **Align** face using similarity transform (straighten tilted/rotated faces)
- **Crop & resize** to exactly **256×256 pixels**
- Discard if detection confidence < 0.7

### Step 3: Anti-Shortcut Preprocessing
| Technique | What It Does | Why |
|---|---|---|
| Random JPEG recompression (q60-95) | Destroys original compression fingerprint | Prevents learning dataset-specific JPEG patterns |
| Fixed 256×256 resolution | All images same size | Prevents learning resolution shortcuts |
| CLAHE color normalization | Equalizes color distribution | Prevents learning camera/dataset color bias |

### Step 4: Three Parallel Analysis Streams

| Stream | Internal Steps | Output |
|---|---|---|
| 🧊 **Spatial** | Face → DINOv2-Base ViT (frozen, 86M params) → LoRA adapters (500K trainable) | 512-dim embedding + 256×768 patch tokens |
| 📊 **Frequency** | Face → Grayscale → 2D FFT → Center shift → Log magnitude → 4-layer CNN | 512-dim embedding |
| 🎯 **Attention** | Patch tokens (from Spatial) → 4 region queries × 8-head cross-attention | 512-dim embedding + 16×16 heatmap |

### Step 5: Simple Fusion MLP (Phase 1)
- Concatenate: [spatial₅₁₂ + frequency₅₁₂ + attention₅₁₂] = **1536-dim vector**
- Pass through gating MLP → **512-dim fused embedding**

### Step 6: Temporal Transformer *(Phase 2 — Video Only)*
- Sample **16 frames** evenly from video
- Each frame → Steps 2-5 → per-frame 512-dim embedding
- 16 embeddings → 4-layer Transformer encoder → **512-dim video embedding**

### Step 7: Classification Output
- **Classification head** → BCE + Focal Loss → **Real/Fake probability (0-100%)**
- **Attention heatmap** → shows WHERE manipulation was detected
- **Frame timeline** (video) → which frames are suspicious

---

## Implementation Phases

| Phase | What | Status |
|-------|------|--------|
| **Phase 1** | Image model: Steps 1-5 + Step 7 | 🔨 Current |
| **Phase 2** | Video: Add Step 6 (Temporal Transformer) | ⏳ Next |
| **Phase 3** | Upgrade fusion to adaptive + add triplet loss | 📋 Planned |
