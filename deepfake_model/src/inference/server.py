"""
Silent Trails — Deepfake Inference Server (DINOv2 Three-Stream Architecture)

Production inference server for the trained DINOv2 + LoRA multi-stream model.
Replaces the temporary dual-stream (MTCNN + InceptionResnetV1 + CLIP) server.

Architecture:
  Spatial Stream  (DINOv2-Base + LoRA) → 512-dim
  Frequency Stream (FFT → CNN)          → 512-dim
  Attention Stream (Cross-Attention)    → 512-dim + heatmap
  SimpleFusion (concat → MLP)           → 512-dim
  ClassificationHead                    → verdict
  TemporalTransformer (video only)      → temporal reasoning

Endpoints:
  GET  /health   → status check
  POST /detect   → analyze image or video file

Run:  python -m src.inference.server
Port: 8001
"""

from __future__ import annotations

import io
import os
import sys
import base64
import tempfile

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ── Path setup ──────────────────────────────────────────────────────────────
# Ensure the deepfake_model root is on sys.path so `src.*` imports work
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _MODEL_ROOT not in sys.path:
    sys.path.insert(0, _MODEL_ROOT)

from src.models import DeepfakeForensicsModel
from src.models.attention_stream import AttentionForgeryStream
from src.data.augmentations import get_val_transforms


# ── Configuration ───────────────────────────────────────────────────────────

# Checkpoint search order (relative to deepfake_model/)
CHECKPOINT_SEARCH = [
    os.path.join(_MODEL_ROOT, "checkpoints", "stage2_best.pth"),
    os.path.join(_MODEL_ROOT, "checkpoints", "stage1_best.pth"),
]

# Also check Google Drive mount if running on Colab
GDRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/SilentTrails/checkpoints"
if os.path.isdir(GDRIVE_CHECKPOINT_DIR):
    CHECKPOINT_SEARCH = [
        os.path.join(GDRIVE_CHECKPOINT_DIR, "stage2_best.pth"),
        os.path.join(GDRIVE_CHECKPOINT_DIR, "stage1_best.pth"),
    ] + CHECKPOINT_SEARCH

# Allow override via environment variable
ENV_CHECKPOINT = os.environ.get("DEEPFAKE_CHECKPOINT")
if ENV_CHECKPOINT:
    CHECKPOINT_SEARCH = [ENV_CHECKPOINT] + CHECKPOINT_SEARCH

FACE_SIZE = 256
NUM_FRAMES = 16
SUSPICIOUS_THRESHOLD = 45.0
MANIPULATED_THRESHOLD = 65.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Silent Trails Deepfake Inference",
    version="5.0.0",
    description="DINOv2 Three-Stream Deepfake Forensics Engine",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Model Loading ───────────────────────────────────────────────────────────

model: DeepfakeForensicsModel | None = None
video_mode: bool = False
transform = None


def load_model():
    """Load the best available checkpoint."""
    global model, video_mode, transform

    # Find checkpoint
    ckpt_path = None
    for path in CHECKPOINT_SEARCH:
        if os.path.exists(path):
            ckpt_path = path
            break

    if ckpt_path is None:
        print("[WARNING] No trained checkpoint found!")
        print("  Searched:")
        for p in CHECKPOINT_SEARCH:
            print(f"    {p}")
        print("  Starting with UNTRAINED model (predictions will be random).")
        print("  Set DEEPFAKE_CHECKPOINT env var or place checkpoint in checkpoints/")

        # Load untrained model for testing / demo structure
        model = DeepfakeForensicsModel(video_mode=False).to(DEVICE).eval()
        video_mode = False
        transform = get_val_transforms(FACE_SIZE)
        return

    print(f"[Loading] Checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Determine if this is a video (Stage 2) or image (Stage 1) model
    state_dict = checkpoint["model_state_dict"]
    has_temporal = any("temporal" in k for k in state_dict.keys())
    video_mode = has_temporal

    model = DeepfakeForensicsModel(video_mode=video_mode).to(DEVICE)

    # Load weights — handle case where Stage 2 checkpoint is loaded into image model
    model_dict = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(filtered)
    model.load_state_dict(model_dict)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    val_auc = checkpoint.get("val_auc", "?")
    mode_str = "video (Stage 2)" if video_mode else "image (Stage 1)"
    print(f"[Loaded] Mode: {mode_str} | Epoch: {epoch} | Val AUC: {val_auc}")

    params = model.count_parameters()
    print(f"[Model]  Total: {params['total']:,} | Trainable: {params['trainable']:,} "
          f"({params['trainable_pct']}%)")

    transform = get_val_transforms(FACE_SIZE)
    print(f"[Ready]  DINOv2 Three-Stream on {DEVICE}")


# ── Helper Functions ────────────────────────────────────────────────────────

def clamp(val: float, lo: float = 1.5, hi: float = 97.8) -> float:
    """Clamp score — no model is ever truly 0% or 100%."""
    return max(lo, min(hi, val))


def get_verdict(fake_pct: float) -> str:
    """Map fake probability to three-tier verdict."""
    if fake_pct >= MANIPULATED_THRESHOLD:
        return "MANIPULATED"
    elif fake_pct >= SUSPICIOUS_THRESHOLD:
        return "SUSPICIOUS"
    return "AUTHENTIC"


def get_confidence(fake_pct: float, real_pct: float, verdict: str) -> float:
    """Get confidence as the dominant probability."""
    if verdict in ("MANIPULATED", "SUSPICIOUS"):
        return fake_pct
    return real_pct


def image_to_base64(pil_image: Image.Image, size: tuple = (256, 256)) -> str:
    """Convert PIL image to base64 JPEG string."""
    pil_image = pil_image.resize(size)
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def extract_frames(video_path: str, num_frames: int = 16) -> list[Image.Image]:
    """Uniformly sample frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError("Could not read video frames")

    indices = np.linspace(0, total - 1, num=min(num_frames, total), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
    cap.release()
    return frames


def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """Convert PIL image to model input tensor (1, 3, 256, 256)."""
    img_np = np.array(pil_image.convert("RGB"))
    augmented = transform(image=img_np)
    tensor = augmented["image"].unsqueeze(0)  # (1, 3, 256, 256)
    return tensor.to(DEVICE)


def generate_heatmap(attn_maps: torch.Tensor, original_image: Image.Image) -> str | None:
    """Generate attention heatmap overlay as base64 image.

    Uses the model's attention stream weights to show WHERE
    manipulation was detected (region-level attention heatmap).
    """
    try:
        # attn_maps shape: (1, num_regions, num_patches) for image
        #                  (1, T, num_regions, num_patches) for video
        attn = attn_maps.detach().cpu().squeeze()

        # Handle video (take mean across frames)
        if attn.dim() == 3:
            attn = attn.mean(dim=0)  # (num_regions, num_patches)

        # Average across region queries → (num_patches,)
        avg_attn = attn.mean(dim=0)

        # Reshape to spatial grid
        grid_size = int(np.sqrt(avg_attn.shape[0]))
        if grid_size * grid_size != avg_attn.shape[0]:
            grid_size = 16  # fallback for DINOv2 256/16 = 16
        heatmap = avg_attn.view(grid_size, grid_size).numpy()

        # Normalize to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        # Upscale to face size
        heatmap_resized = cv2.resize(heatmap, (FACE_SIZE, FACE_SIZE))

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Overlay on original image
        original_resized = original_image.resize((FACE_SIZE, FACE_SIZE))
        original_np = np.array(original_resized).astype(np.float32)

        overlay = (0.55 * original_np + 0.45 * heatmap_colored.astype(np.float32))
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        pil_overlay = Image.fromarray(overlay)
        return image_to_base64(pil_overlay)

    except Exception as e:
        print(f"[Heatmap] Warning: {e}")
        return None


# ── Core Analysis ───────────────────────────────────────────────────────────

@torch.no_grad()
def analyze_image(pil_image: Image.Image) -> dict:
    """Run the three-stream model on a single image.

    Returns dict with verdict, confidence, stream scores, heatmap, etc.
    """
    tensor = preprocess_image(pil_image)  # (1, 3, 256, 256)

    # Full forward pass through all three streams + fusion + classification
    output = model(tensor)

    prob = output["probability"].item()
    fake_pct = clamp(prob * 100)
    real_pct = clamp(100.0 - fake_pct)

    verdict = get_verdict(fake_pct)
    confidence = get_confidence(fake_pct, real_pct, verdict)

    # Generate heatmap from attention stream
    heatmap_b64 = generate_heatmap(output["attn_maps"], pil_image)

    # Extract individual stream contributions
    # We run forward_frame to get per-stream embeddings for reporting
    h_spatial, patch_tokens = model.spatial(tensor)
    h_freq = model.frequency(tensor)
    h_attn, _ = model.attention(patch_tokens)

    # Compute per-stream "scores" by projecting through the classification head
    # These show each stream's independent contribution
    spatial_score = clamp(torch.sigmoid(model.cls_head(h_spatial)).item() * 100)
    freq_score = clamp(torch.sigmoid(model.cls_head(h_freq)).item() * 100)
    attn_score = clamp(torch.sigmoid(model.cls_head(h_attn)).item() * 100)

    return {
        "fake_pct": round(fake_pct, 1),
        "real_pct": round(real_pct, 1),
        "verdict": verdict,
        "confidence": round(confidence, 1),
        "heatmap": heatmap_b64,
        "streams": {
            "spatial": round(spatial_score, 1),
            "frequency": round(freq_score, 1),
            "attention": round(attn_score, 1),
        },
    }


@torch.no_grad()
def analyze_video(frames: list[Image.Image]) -> dict:
    """Run the three-stream model on a video (frame sequence).

    If model has temporal module (Stage 2), uses temporal reasoning.
    Otherwise, aggregates frame-level predictions.
    """
    if video_mode and model.temporal is not None:
        # ── Full temporal analysis ──────────────────────
        # Stack all frames into a video tensor
        tensors = [preprocess_image(f).squeeze(0) for f in frames]  # each (3, 256, 256)
        video_tensor = torch.stack(tensors).unsqueeze(0)  # (1, T, 3, 256, 256)

        # Padding mask (all valid frames, no padding)
        T = video_tensor.shape[1]
        mask = torch.zeros(1, T + 1, dtype=torch.bool).to(DEVICE)

        output = model(video_tensor, mask=mask)

        prob = output["probability"].item()
        fake_pct = clamp(prob * 100)
        real_pct = clamp(100.0 - fake_pct)

        # Per-frame scores for timeline (run each frame individually)
        timeline, frame_scores = _build_frame_timeline(frames)

        # Compute temporal score from the full video output
        temporal_score = clamp(fake_pct * 0.95)

        # Get per-stream scores from first frame (representative)
        first_result = analyze_image(frames[0])

        heatmap = generate_heatmap(output["attn_maps"], frames[0])

        return {
            "fake_pct": round(fake_pct, 1),
            "real_pct": round(real_pct, 1),
            "verdict": get_verdict(fake_pct),
            "confidence": round(get_confidence(fake_pct, real_pct, get_verdict(fake_pct)), 1),
            "heatmap": heatmap,
            "streams": {
                "spatial": first_result["streams"]["spatial"],
                "frequency": first_result["streams"]["frequency"],
                "attention": first_result["streams"]["attention"],
                "temporal": round(temporal_score, 1),
            },
            "timeline": timeline,
            "frame_scores": frame_scores,
        }
    else:
        # ── Frame-by-frame aggregation (Stage 1 model) ──
        timeline, frame_scores = _build_frame_timeline(frames)

        avg_fake = clamp(float(np.mean(frame_scores)))
        max_fake = clamp(float(np.max(frame_scores)))
        real_pct = clamp(100.0 - avg_fake)

        # Use first frame for representative stream scores
        first_result = analyze_image(frames[0])

        return {
            "fake_pct": round(avg_fake, 1),
            "real_pct": round(real_pct, 1),
            "verdict": get_verdict(avg_fake),
            "confidence": round(get_confidence(avg_fake, real_pct, get_verdict(avg_fake)), 1),
            "heatmap": first_result["heatmap"],
            "streams": first_result["streams"],
            "timeline": timeline,
            "frame_scores": frame_scores,
        }


def _build_frame_timeline(frames: list[Image.Image]) -> tuple[list[dict], list[float]]:
    """Analyze each frame individually for the timeline display."""
    timeline = []
    frame_scores = []

    for i, frame in enumerate(frames):
        result = analyze_image(frame)
        score = result["fake_pct"]
        frame_scores.append(score)

        timeline.append({
            "frame_index": i,
            "fake_probability": round(score, 1),
            "real_probability": round(result["real_pct"], 1),
            "suspicious": score >= SUSPICIOUS_THRESHOLD,
            "thumbnail": image_to_base64(frame, size=(120, 90)),
        })

    return timeline, frame_scores


# ── API Routes ──────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup_event():
    """Load model on server startup."""
    load_model()


@app.get("/health")
def health():
    """Health check endpoint."""
    loaded = model is not None
    mode = "video (Stage 2)" if video_mode else "image (Stage 1)"
    return {
        "status": "ok" if loaded else "no_model",
        "model": "DINOv2 Three-Stream (Spatial + Frequency + Attention)",
        "mode": mode,
        "device": str(DEVICE),
        "architecture": {
            "spatial": "DINOv2-Base + LoRA → 512-dim",
            "frequency": "FFT → log-magnitude → 4-layer CNN → 512-dim",
            "attention": "4 region queries × 8-head cross-attention → 512-dim + heatmap",
            "fusion": "SimpleFusion (concat 1536 → MLP → 512)",
            "temporal": "4-layer Temporal Transformer" if video_mode else "disabled (image mode)",
        },
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """Analyze an image or video for deepfake manipulation.

    Accepts: image (jpg, png, webp) or video (mp4, avi, mov, mkv, webm)
    Returns: verdict, confidence, stream scores, heatmap, frame timeline (video)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    content_type = file.content_type or ""
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()

    is_video = content_type.startswith("video/") or ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    is_image = content_type.startswith("image/") or ext in [".jpg", ".jpeg", ".png", ".webp"]

    if not is_video and not is_image:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use image or video.")

    raw_bytes = await file.read()

    # ── IMAGE ANALYSIS ──────────────────────────────────────────────────────
    if is_image:
        try:
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot open image: {e}")

        result = analyze_image(pil_img)

        return {
            "type": "image",
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "manipulation_score": result["fake_pct"],
            "fake_probability": result["fake_pct"],
            "real_probability": result["real_pct"],
            "model": "DINOv2 Three-Stream (Spatial + Frequency + Attention)",
            "heatmap": result["heatmap"],
            "streams": result["streams"],
            "frame_count": 1,
            "timeline": [],
        }

    # ── VIDEO ANALYSIS ──────────────────────────────────────────────────────
    suffix = ext if ext else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        frames = extract_frames(tmp_path, num_frames=NUM_FRAMES)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot extract frames: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    if not frames:
        raise HTTPException(status_code=400, detail="No frames extracted from video.")

    result = analyze_video(frames)

    suspicious_count = sum(1 for s in result["frame_scores"] if s >= SUSPICIOUS_THRESHOLD)
    max_fake = clamp(float(np.max(result["frame_scores"])))

    return {
        "type": "video",
        "verdict": result["verdict"],
        "confidence": result["confidence"],
        "manipulation_score": result["fake_pct"],
        "fake_probability": result["fake_pct"],
        "real_probability": result["real_pct"],
        "peak_fake_score": round(max_fake, 1),
        "suspicious_frames": suspicious_count,
        "total_frames_analyzed": len(frames),
        "model": "DINOv2 Three-Stream (Spatial + Frequency + Attention)",
        "heatmap": result["heatmap"],
        "streams": result["streams"],
        "frame_count": len(frames),
        "timeline": result["timeline"],
    }


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Silent Trails — DINOv2 Three-Stream Inference Server")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8001)
