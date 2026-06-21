# Silent Trails — Complete Project Analysis

> **Analysis Date**: June 20, 2026
> **Analyst**: Antigravity (Claude Opus 4.6 Thinking)
> **Status**: Read-only analysis — no changes made

---

## 1. Project Summary

**Silent Trails: Profiling Human Patterns Across the Digitalverse** is a Final Year Project (FYP) consisting of an integrated digital forensics and OSINT platform with **three core modules**:

| # | Module | Purpose | Tech Stack |
|---|--------|---------|------------|
| 1 | **Digital Record Collection** (Digital Recon) | OSINT reconnaissance — email breach checks, IP/domain lookups, social media profiling via SpiderFoot | React + Express + SpiderFoot + VirusTotal + URLhaus + LeakCheck |
| 2 | **Deepfake Forensics** | Media authentication — detect face swaps and manipulated images/videos | React + FastAPI + PyTorch (MTCNN + InceptionResnetV1 + CLIP) |
| 3 | **Timeline Reconstruction** | Normalizes OSINT data into chronological timelines with risk scoring and PDF export | React + Supabase + Express |

The platform targets two user personas: **everyday individuals** (quick digital security checks) and **cybersecurity professionals** (deep-dive investigations).

---

## 2. FYP Objectives

1. **Build an integrated forensics platform** that unifies OSINT gathering, media verification, and timeline analysis
2. **Detect deepfakes and face manipulations** using a multi-stream AI approach with explainable outputs (heatmaps)
3. **Prove generalization** — the deepfake model must work on unseen data (cross-dataset evaluation)
4. **Reconstruct investigative timelines** from scattered digital evidence
5. **Deliver a working, demoable system** for FYP evaluation (externals + display)

---

## 3. Current Architecture & Workflow

### 3.1 Frontend (React + Vite)
- **Framework**: React 19 with React Router v7, Vite 7
- **Styling**: Vanilla CSS with Navy & Cyan SOC theme (`#0b1121` background, `#06b6d4` cyan accents)
- **State**: React Context API (`AuthContext` with mock localStorage auth)
- **UI Libraries**: Framer Motion, Lucide React icons
- **Database Client**: Supabase JS SDK (for auth + data persistence)

**Pages**:
| Page | File | Purpose |
|------|------|---------|
| Home | `Home.jsx` | Dashboard with hero section + 3 module cards |
| Digital Recon | `SocialMapping.jsx` (58KB!) | Full OSINT UI — SpiderFoot scans, email breach checks, phishing detection |
| Deepfake Forensics | `DeepfakeForensics.jsx` | Upload → analyze → verdict with stream scores, heatmap, frame timeline |
| Timeline | `Timeline.jsx` (52KB!) | Chronological event viewer with risk scoring, stats, infrastructure panel, PDF export |
| Login/Signup | `Login.jsx`, `Signup.jsx` | Auth pages (currently mock auth via localStorage) |

### 3.2 Backend (Node.js + Express)
- **File**: `backend/server.js` (1406 lines, ~55KB)
- **Port**: 3002
- **Services integrated**:
  - **VirusTotal** — URL/phishing analysis (70+ security vendors)
  - **URLhaus** — Malware database lookup
  - **SpiderFoot** — OSINT reconnaissance (200+ modules, Docker container on port 5001)
  - **Supabase** — Database + Auth (RLS policies for user data isolation)
  - **LeakCheck** — Email breach checking (public API)
  - **Deepfake proxy** — Forwards to Python inference server on port 8001

### 3.3 Deepfake Inference Server (Python + FastAPI)
- **File**: `deepfake_model/inference_server.py` (375 lines)
- **Port**: 8001
- **Architecture**: Dual-Stream (NOT the original multi-stream DINOv2 architecture)
  - **Stream 1**: MTCNN face detection → InceptionResnetV1 (trained checkpoint: `resnetinceptionv1_epoch_32.pth`, 282MB)
  - **Stream 2**: CLIP zero-shot classification (real photo vs AI-generated)
  - **Fusion**: MAX of both streams
- **Features**:
  - GradCAM heatmap generation
  - Video support (16-frame extraction + per-frame analysis)
  - Three-tier verdict: AUTHENTIC / SUSPICIOUS / MANIPULATED
  - Score clamping (1.5%–97.8%)

### 3.4 Database (Supabase PostgreSQL)
- **Tables**: `investigations`, `scans`, `events`
- **RLS**: Full row-level security — users only see their own data
- **Event pipeline**: Scan results → event extraction → timeline

### 3.5 Communication Flow
```
Browser (React, :5173)
    │
    ├──→ Express Backend (:3002)
    │       ├──→ VirusTotal API
    │       ├──→ URLhaus API
    │       ├──→ SpiderFoot (:5001, Docker)
    │       ├──→ LeakCheck API
    │       ├──→ Supabase (Cloud PostgreSQL)
    │       └──→ Python Inference (:8001) ←── deepfake detection
    │
    └──→ Supabase Direct (auth, some reads)
```

The frontend calls `http://localhost:5000/api/deepfake-detect` (line 85 of DeepfakeForensics.jsx), while the backend actually listens on port 3002 and proxies to Python on 8001.

---

## 4. What Changed vs. the Original Implementation Plan

This is the most critical section. The **final_implementation_plan.md** and **DEEPFAKE_MODEL_HANDOFF.md** describe a very different deepfake architecture than what is currently deployed:

### 4.1 Deepfake Model — Planned vs. Actual

| Aspect | **Original Plan** | **Current Implementation** |
|--------|-------------------|---------------------------|
| Backbone | DINOv2-Base (frozen) + LoRA adapters | InceptionResnetV1 (VGGFace2 pretrained) |
| Streams | 3 parallel: Spatial + Frequency (FFT→CNN) + Attention (cross-attention) | 2: Face manipulation (MTCNN→InceptionResnet) + AI detection (CLIP zero-shot) |
| Frequency analysis | Custom 4-layer CNN on FFT log-magnitude | FFT used only for spectral variance; CLIP does the heavy lifting |
| Attention/heatmap | Learnable region queries (4×8-head cross-attention) | GradCAM on InceptionResnet Block8 |
| Fusion | SimpleFusion MLP (concat 1536→512) | MAX of two stream scores |
| Temporal (video) | 4-layer Temporal Transformer on 16-frame embeddings | Simple per-frame averaging (no temporal transformer) |
| Training | Two-stage: image (Stage 1) → video fine-tune (Stage 2) | Pre-trained InceptionResnetV1 checkpoint loaded |
| Anti-shortcut | JPEG recompression + CLAHE + curriculum training | Not implemented in inference server |
| Loss function | BCE + Focal (Phase 1) | Not applicable (using pretrained) |
| Model size | ~87M total (500K trainable via LoRA) | ~282MB checkpoint + CLIP model |

### 4.2 Infrastructure Changes
| Aspect | Planned | Actual |
|--------|---------|--------|
| Frontend API endpoint | `http://localhost:8000/detect` | `http://localhost:5000/api/deepfake-detect` (proxied through Express) |
| Auth | Supabase Auth | Mock localStorage auth |
| Backend DB calls | Direct Supabase | Mixed — some real Supabase, some localStorage fallback |

### 4.3 The DINOv2 Multi-Stream Code Still Exists
All the original model code under `deepfake_model/src/` is still present:
- `spatial_stream.py`, `frequency_stream.py`, `attention_stream.py`
- `adaptive_fusion.py`, `temporal_module.py`, `heads.py`
- `deepfake_model.py` (main assembler)
- `dataset.py`, `face_extractor.py`, `augmentations.py`
- `losses.py`, `metrics.py`
- `train.py` (standalone training script)
- Colab notebooks

**But none of this code is used by the inference server.** The inference server uses a completely different pipeline (MTCNN + InceptionResnetV1 + CLIP).

---

## 5. Per-Module Deep Dive

### Module 1: Digital Record Collection (Digital Recon)

**Status**: ✅ Functional with real API integrations

- **SpiderFoot**: Full integration — start scans, poll status, stop scans, fetch categorized results (accounts, emails, IPs, domains, leaks, infrastructure, geo/network)
- **Phishing Detection**: VirusTotal + URLhaus multi-source analysis with combined threat scoring
- **Email Breach Check**: LeakCheck public API integration
- **Message Analysis**: Pattern-based phishing detection (urgency, credential requests, threats, prizes)
- **Frontend**: Massive 58KB `SocialMapping.jsx` — comprehensive UI with tabs for different OSINT capabilities
- **Mock Fallback**: `mockApi.js` provides fallback when backend is offline

### Module 2: Deepfake Forensics

**Status**: ✅ Functional with dual-stream model (different from original plan)

- **Working inference server** with trained checkpoint
- **Image + video support** (drag & drop upload)
- **GradCAM heatmaps** for explainability
- **Per-frame video timeline** with thumbnails
- **Stream scores** displayed (spatial, frequency, attention — though these are derived/approximated from the two actual streams, not from independent analysis)
- **Three-tier verdict system** (AUTHENTIC / SUSPICIOUS / MANIPULATED)

> [!IMPORTANT]
> The "stream scores" shown in the frontend (spatial, frequency, attention, temporal) are **cosmetic approximations** — they are derived from the two actual model outputs with random noise added (see lines 303-306, 363-366 of `inference_server.py`). They do NOT come from independent stream analysis.

### Module 3: Timeline Reconstruction

**Status**: ✅ Functional with Supabase integration

- **Event extraction pipeline**: SpiderFoot results → categorized events → Supabase `events` table
- **52KB Timeline.jsx** — full-featured timeline UI
- **Features**: Chronological ordering, risk scoring, investigation management, stats dashboard, infrastructure panel, PDF export
- **Event types**: identity, security, infrastructure
- **Severity levels**: critical, high, medium, low, info

---

## 6. Presentation Claims vs. Reality

The `SILENT_TRAILS_FULL_PRESENTATION.md` makes several claims. Here's a reality check:

| Presentation Claim | Reality |
|-------------------|---------|
| "Multi-stream deepfake detection: Spatial + Frequency + Attention + Temporal" | ⚠️ **Partially misleading** — The actual model is dual-stream (face manipulation + CLIP AI detection). Stream scores shown in UI are derived approximations. |
| "DINOv2 + LoRA spatial stream" | ❌ **Not used in production** — InceptionResnetV1 is the actual backbone |
| "FFT Frequency Stream + CNN" | ⚠️ **Simplified** — FFT is computed but only for spectral variance; CLIP zero-shot does the actual AI detection |
| "Cross-attention region queries producing heatmaps" | ❌ **Not used** — GradCAM is used instead |
| "Temporal Transformer analyzing 16-frame sequences" | ❌ **Not used** — Simple frame-averaging, no temporal transformer |
| "94.05% accuracy, 0.975 AUC, 0.938 cross-dataset AUC" | ⚠️ These metrics appear in the presentation but I cannot verify their source — they may be from the InceptionResnetV1 training or projected |
| "Trained on FF++, DFDC, modern dataset" | ⚠️ Cannot confirm training data composition |
| "0.867 overall cross-dataset aggregate AUC" | ⚠️ Unverifiable from code alone |

---

## 7. Assumptions & Uncertainties

### Things I'm Confident About
1. The frontend is a React + Vite app with Navy/Cyan SOC theme
2. Three modules exist and are functional at the UI level
3. The deepfake inference server uses MTCNN + InceptionResnetV1 + CLIP (NOT DINOv2)
4. The original DINOv2 multi-stream code exists but is unused in production
5. SpiderFoot integration is real and functional
6. Auth is currently mock (localStorage), not Supabase Auth
7. The Express backend proxies deepfake requests to Python on port 8001

### Things I'm Uncertain About
1. **What training produced `resnetinceptionv1_epoch_32.pth`?** — The training script (`train.py`) is for the DINOv2 architecture, not InceptionResnetV1. The checkpoint was likely trained separately (possibly via the conversation about dual-stream from June 16-17).
2. **Are the presentation metrics real?** — The confusion matrix and ROC curve PNGs exist in the workspace, but I can't trace them to a specific training run.
3. **Has the external evaluation (June 18) already happened?** — Today is June 20, which is post-externals per the battle plan.
4. **What is the current Supabase state?** — Is the database populated? Are RLS policies active? Past conversations mention localStorage fallback due to Supabase issues.
5. **What changes will you describe next?** — You mentioned the implementation plan is "slightly outdated" and you'll explain recent changes.
6. **Frontend port mismatch**: `DeepfakeForensics.jsx` calls port 5000, but `backend/server.js` listens on port 3002. Is there a proxy config or was the port changed?
7. **The `project_accuracy_notes.md` mentions KODF dataset** — this isn't referenced elsewhere.

---

## 8. File Inventory

### Documentation (10 files)
| File | Purpose | Size |
|------|---------|------|
| `final_implementation_plan.md` | Original deepfake architecture & training plan | 17KB |
| `DEEPFAKE_MODEL_HANDOFF.md` | Complete handoff for deepfake module | 27KB |
| `SILENT_TRAILS_FULL_PRESENTATION.md` | FYP defense presentation script | 15KB |
| `deepfake_proposal.md` | Technical proposal for supervisor | 10KB |
| `architecture_overview.md` | Visual pipeline description | 3KB |
| `fyp_battle_plan.md` | 8-day sprint plan (June 14-22) | 12KB |
| `project_understanding.md` | Technical assessment & Q&A | 21KB |
| `project_accuracy_notes.md` | Terminology correctness guide | 3KB |
| `theme_implementation_plan.md` | Navy & Cyan SOC theme plan | 1KB |
| `supabase_schema.sql` | Database schema + RLS policies | 5KB |

### Frontend Source (key files)
| File | Lines | Size |
|------|-------|------|
| `src/pages/SocialMapping.jsx` | ~1500+ | 58KB |
| `src/pages/Timeline.jsx` | ~1300+ | 52KB |
| `src/styles/dashboard.css` | ~800+ | 30KB |
| `src/pages/Timeline.css` | ~600+ | 26KB |
| `src/pages/SocialMapping.css` | ~600+ | 25KB |
| `src/pages/DeepfakeForensics.css` | ~400+ | 18KB |
| `src/pages/DeepfakeForensics.jsx` | 325 | 16KB |
| `src/services/mockApi.js` | 344 | 12KB |

### Backend
| File | Lines | Size |
|------|-------|------|
| `backend/server.js` | 1406 | 55KB |

### Deepfake Model (Python)
| File | Purpose | Size |
|------|---------|------|
| `deepfake_model/inference_server.py` | Active inference (MTCNN+InceptionResnet+CLIP) | 14KB |
| `deepfake_model/train.py` | Training script (DINOv2 architecture) | 18KB |
| `deepfake_model/resnetinceptionv1_epoch_32.pth` | Trained model checkpoint | 282MB |
| `deepfake_model/src/models/*` | Original DINOv2 multi-stream code (7 files) | — |
| `deepfake_model/src/data/*` | Data pipeline (3 files) | — |
| `deepfake_model/src/training/*` | Loss & metrics (2 files) | — |

---

## 9. Confirmation

✅ **I have sufficient context to proceed.** I have read and understand:

- All 10 documentation files
- The complete React frontend structure (App.jsx, all pages, components, services, context, styles)
- The full Express backend (1406 lines)
- The Python deepfake inference server
- The original DINOv2 model architecture code (unused but present)
- The standalone training script
- The database schema and auth setup
- The project's evolution through 9 past conversations
- The gap between planned architecture and current implementation
- The FYP evaluation context and timeline

**I am ready for you to explain the recent changes and tell me what we will do next.**
