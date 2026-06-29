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
import re
import sys
import base64
import tempfile
import struct

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN

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

# Logit calibration: bias shift + temperature.
# The model outputs strongly negative logits for BOTH real and fake images
# (e.g., real: -5.0, fake: -4.0) due to training with heavy regularization.
# Step 1 — Bias correction: shift up by LOGIT_BIAS so the decision boundary
#           is near 0 (real images go negative, fake images go positive).
# Step 2 — Temperature: divide by T to spread the distribution.
# These values are tuned empirically. Adjust LOGIT_BIAS if needed.
LOGIT_BIAS = 7.5        # shift center up — tuned to observed raw_logit ≈ -7.5
CALIBRATION_TEMPERATURE = 0.5  # spread factor

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
transform  = None
mtcnn      = None
face_model = None   # InceptionResnetV1 face-swap engine

# Checkpoint for the face-swap detector (lives next to server package root)
FACE_SWAP_CKPT = os.path.join(_MODEL_ROOT, "resnetinceptionv1_epoch_32.pth")


def load_model():
    """Load DINOv2 three-stream model AND InceptionResnetV1 face-swap detector."""
    global model, video_mode, transform, mtcnn, face_model

    print("[Loading] MTCNN Face Detector...")
    mtcnn = MTCNN(image_size=256, margin=40, keep_all=False,
                  select_largest=True, post_process=False, device=DEVICE).eval()

    # ── Engine 1: InceptionResnetV1 face-swap detector ───────────────────────
    try:
        from facenet_pytorch import InceptionResnetV1
        print(f"[Engine 1] Loading face-swap detector: {FACE_SWAP_CKPT}")
        if os.path.exists(FACE_SWAP_CKPT):
            _fm = InceptionResnetV1(pretrained="vggface2", classify=True,
                                    num_classes=1, device=DEVICE)
            _ckpt = torch.load(FACE_SWAP_CKPT, map_location="cpu", weights_only=False)
            _fm.load_state_dict(_ckpt["model_state_dict"])
            _fm.to(DEVICE).eval()
            face_model = _fm
            print("[Engine 1] InceptionResnetV1 face-swap detector READY")
        else:
            print(f"[Engine 1] Checkpoint not found — face-swap detection disabled")
    except Exception as e:
        print(f"[Engine 1] Failed to load face-swap model: {e}")

    # ── Engine 2: DINOv2 three-stream AI-gen detector ────────────────────────
    ckpt_path = None
    for path in CHECKPOINT_SEARCH:
        if os.path.exists(path):
            ckpt_path = path
            break

    if ckpt_path is None:
        print("[WARNING] No trained DINOv2 checkpoint found!")
        model = DeepfakeForensicsModel(video_mode=False).to(DEVICE).eval()
        video_mode = False
        transform = get_val_transforms(FACE_SIZE)
        return

    print(f"[Engine 2] Loading DINOv2 checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    has_temporal = any("temporal" in k for k in state_dict.keys())
    video_mode = has_temporal

    model = DeepfakeForensicsModel(video_mode=video_mode).to(DEVICE)
    model_dict = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(filtered)
    model.load_state_dict(model_dict)
    model.eval()

    epoch   = checkpoint.get("epoch", "?")
    val_auc = checkpoint.get("val_auc", "?")
    mode_str = "video (Stage 2)" if video_mode else "image (Stage 1)"
    print(f"[Engine 2] DINOv2 Mode: {mode_str} | Epoch: {epoch} | Val AUC: {val_auc}")

    params = model.count_parameters()
    print(f"[Engine 2] Total: {params['total']:,} | Trainable: {params['trainable']:,} "
          f"({params['trainable_pct']}%)")

    transform = get_val_transforms(FACE_SIZE)
    print(f"[Ready]  Dual-Engine on {DEVICE} — Face Swap + AI Generation")


# ── Helper Functions ────────────────────────────────────────────────────────

def clamp(val: float, lo: float = 0.1, hi: float = 99.9) -> float:
    """Clamp score — prevents exact 0% or 100% display values."""
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


def get_face_crop(pil_image: Image.Image) -> np.ndarray:
    """Return the MTCNN face crop as a numpy uint8 array (H, W, 3).
    Falls back to the full image resized to FACE_SIZE if no face is found."""
    face_tensor = mtcnn(pil_image)
    if face_tensor is not None:
        return face_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return np.array(pil_image.convert("RGB").resize((FACE_SIZE, FACE_SIZE)))


def preprocess_image(pil_image: Image.Image) -> tuple[torch.Tensor, np.ndarray]:
    """Convert PIL image to model input tensor AND return the raw face crop.

    Returns:
        tensor   (1, 3, 256, 256) on DEVICE
        face_np  (256, 256, 3) uint8 numpy array (for FFT / visualisation)
    """
    face_np = get_face_crop(pil_image)
    augmented = transform(image=face_np)
    tensor = augmented["image"].unsqueeze(0).to(DEVICE)
    return tensor, face_np


def generate_frequency_map(face_np: np.ndarray) -> str | None:
    """Generate a forensic frequency-domain (FFT) spectrum map from a face crop.

    Takes the already-extracted face numpy array (avoids calling MTCNN twice).
    Returns a false-colour log-magnitude spectrum with anomalous spike circles
    as a base64 JPEG string.
    """
    try:
        # Convert to float32 grayscale
        gray = cv2.cvtColor(face_np, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # 2-D DFT → shift zero-frequency to centre
        dft   = np.fft.fft2(gray)
        shift = np.fft.fftshift(dft)

        # Log-magnitude spectrum (compress dynamic range)
        magnitude = np.log1p(np.abs(shift))

        # Normalise to [0, 255]
        mag_norm = cv2.normalize(magnitude, None, 0, 255,
                                 cv2.NORM_MINMAX).astype(np.uint8)

        # Apply INFERNO colourmap
        coloured = cv2.applyColorMap(mag_norm, cv2.COLORMAP_INFERNO)
        coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)

        # Mark the DC centre
        h, w = coloured.shape[:2]
        cx, cy = w // 2, h // 2
        cv2.circle(coloured, (cx, cy), 6, (255, 255, 255), 1)

        # Highlight anomalous high-frequency spikes
        mask = mag_norm.copy()
        cv2.circle(mask, (cx, cy), 20, 0, -1)           # suppress DC lobe
        thresh     = np.percentile(mask, 99)
        spike_mask = (mask > thresh).astype(np.uint8) * 255
        kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        spike_mask = cv2.dilate(spike_mask, kernel, iterations=1)
        contours, _ = cv2.findContours(spike_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x2, y2, bw, bh = cv2.boundingRect(cnt)
            cr, cc = y2 + bh // 2, x2 + bw // 2
            radius = max(bw, bh) // 2 + 4
            cv2.circle(coloured, (cc, cr), radius, (0, 255, 255), 1)

        pil_out = Image.fromarray(coloured)
        buf = io.BytesIO()
        pil_out.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()

    except Exception as e:
        import traceback
        print(f"[WARN] frequency map failed: {e}")
        traceback.print_exc()
        return None


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
        import traceback
        print(f"[Heatmap] Warning: {e}")
        traceback.print_exc()
        return None


# ── Core Analysis ───────────────────────────────────────────────────────────

@torch.no_grad()
def detect_face_swap(face_np: np.ndarray) -> dict:
    """Engine 1: InceptionResnetV1 face-swap / manipulation detector.

    Args:
        face_np: MTCNN face crop (H, W, 3) uint8 — already extracted.
    Returns:
        dict with face_swap_score (0-100), engine_available (bool).
    """
    if face_model is None:
        return {"face_swap_score": 0.0, "engine_available": False}

    try:
        # Resize and normalise to [0, 1] float
        face_tensor = torch.from_numpy(face_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        face_tensor = torch.nn.functional.interpolate(
            face_tensor, size=(256, 256), mode="bilinear", align_corners=False
        ).to(DEVICE)

        with torch.no_grad():
            raw = torch.sigmoid(face_model(face_tensor).squeeze(0)).item()

        score = clamp(raw * 100)
        print(f"[Engine 1] face_swap_raw={raw:.4f} | face_swap_score={score:.1f}%")
        return {"face_swap_score": round(score, 1), "engine_available": True}
    except Exception as e:
        print(f"[Engine 1] inference failed: {e}")
        return {"face_swap_score": 0.0, "engine_available": False}


@torch.no_grad()
def analyze_image(pil_image: Image.Image) -> dict:
    """Dual-engine analysis: face-swap detector + DINOv2 AI-gen detector.

    Engine 1 (InceptionResnetV1) catches face swaps and traditional manipulations.
    Engine 2 (DINOv2 Three-Stream) catches AI-generated / GAN images.
    Final score = MAX(engine1, engine2) so EITHER finding triggers the alert.
    """
    # Single MTCNN call — face_np reused for Engine 1 AND FFT map
    tensor, face_np = preprocess_image(pil_image)

    # ── Engine 1: face-swap detection ────────────────────────────────────────
    swap_result = detect_face_swap(face_np)
    face_swap_score = swap_result["face_swap_score"]

    # ── Engine 2: DINOv2 AI-gen detection ────────────────────────────────────
    output = model(tensor)
    raw_logit        = output["logits"].item()
    calibrated_logit = (raw_logit + LOGIT_BIAS) / CALIBRATION_TEMPERATURE
    prob             = torch.sigmoid(torch.tensor(calibrated_logit)).item()
    ai_gen_score     = clamp(prob * 100)

    # ── MAX fusion ───────────────────────────────────────────────────────────
    # Whichever engine fires higher wins — catches BOTH manipulation types.
    fake_pct   = clamp(max(face_swap_score, ai_gen_score))
    real_pct   = clamp(100.0 - fake_pct)
    verdict    = get_verdict(fake_pct)
    confidence = get_confidence(fake_pct, real_pct, verdict)

    # Determine which engine drove the verdict
    if face_swap_score >= ai_gen_score:
        detection_engine = "Face Swap Detector (InceptionResnetV1)"
    else:
        detection_engine = "AI Generation Detector (DINOv2 Three-Stream)"

    print(f"[DEBUG] raw_logit={raw_logit:.4f} | calibrated={calibrated_logit:.4f} "
          f"| ai_gen={ai_gen_score:.1f}% | face_swap={face_swap_score:.1f}% "
          f"| final={fake_pct:.1f}%")

    # ── Heatmap + frequency map ───────────────────────────────────────────────
    heatmap_b64  = generate_heatmap(output["attn_maps"], pil_image)
    freq_map_b64 = generate_frequency_map(face_np)

    # ── Per-stream scores (DINOv2 ablation) ──────────────────────────────────
    h_spatial, patch_tokens = model.spatial(tensor)
    h_freq  = model.frequency(tensor)
    h_attn, _ = model.attention(patch_tokens)
    zeros   = torch.zeros_like(h_spatial)

    fused_s, _ = model.fusion(h_spatial, zeros, zeros)
    fused_f, _ = model.fusion(zeros, h_freq, zeros)
    fused_a, _ = model.fusion(zeros, zeros, h_attn)

    spatial_score = clamp(torch.sigmoid(torch.tensor(
        model.cls_head(fused_s).item() / CALIBRATION_TEMPERATURE)).item() * 100)
    freq_score    = clamp(torch.sigmoid(torch.tensor(
        model.cls_head(fused_f).item() / CALIBRATION_TEMPERATURE)).item() * 100)
    attn_score    = clamp(torch.sigmoid(torch.tensor(
        model.cls_head(fused_a).item() / CALIBRATION_TEMPERATURE)).item() * 100)

    return {
        "fake_pct":        round(fake_pct, 1),
        "real_pct":        round(real_pct, 1),
        "verdict":         verdict,
        "confidence":      round(confidence, 1),
        "detection_engine": detection_engine,
        "engines": {
            "face_swap":  round(face_swap_score, 1),
            "ai_gen":     round(ai_gen_score, 1),
        },
        "heatmap":         heatmap_b64,
        "frequency_map":   freq_map_b64,
        "streams": {
            "spatial":   round(spatial_score, 1),
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
        tensors = [preprocess_image(f)[0].squeeze(0) for f in frames]  # each (3, 256, 256)
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


# ── Metadata Forensics ──────────────────────────────────────────────────────

# Known AI generator signatures found in EXIF/XMP/PNG metadata
_AI_SIGNATURES = [
    # Google Gemini / Imagen
    (re.compile(r'google|imagen|gemini|bard', re.I),   "Google Gemini (Imagen 3)",   "CONFIRMED_AI"),
    # OpenAI DALL-E
    (re.compile(r'openai|dall.?e|gpt', re.I),          "OpenAI DALL-E",              "CONFIRMED_AI"),
    # Midjourney
    (re.compile(r'midjourney', re.I),                  "Midjourney",                 "CONFIRMED_AI"),
    # Stable Diffusion / ComfyUI / AUTOMATIC1111
    (re.compile(r'stable.diffusion|comfyui|automatic1111|a1111|diffusers|sd-webui', re.I),
                                                       "Stable Diffusion",           "CONFIRMED_AI"),
    # Adobe Firefly
    (re.compile(r'adobe.firefly|firefly', re.I),       "Adobe Firefly",              "CONFIRMED_AI"),
    # Canva / other AI editors
    (re.compile(r'canva|ai.generated|ai_generated|generated.by.ai', re.I),
                                                       "AI Image Editor",            "CONFIRMED_AI"),
    # C2PA standard marker
    (re.compile(r'c2pa|content.credentials|cai.adobe', re.I),
                                                       "C2PA Content Credential",    "CONFIRMED_AI"),
]


def analyze_metadata(raw_bytes: bytes, pil_img: Image.Image) -> dict:
    """
    Forensic metadata analysis: inspect EXIF, XMP, PNG chunks, and
    raw binary content for AI generation signatures.

    Returns:
        dict with keys:
            detected   (bool)  — True if AI signature found
            generator  (str)   — identified generator name (or "Unknown")
            confidence (str)   — "CONFIRMED_AI" | "SUSPICIOUS" | "CLEAN"
            signals    (list)  — list of evidence strings
            warning    (str)   — human-readable summary
    """
    signals = []
    generator = "Unknown"
    confidence = "CLEAN"

    # ── 1. Standard EXIF ────────────────────────────────────────────────────
    try:
        exif_data = pil_img._getexif() or {}
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, str(tag_id))
            val_str = str(value)
            for pattern, gen_name, conf in _AI_SIGNATURES:
                if pattern.search(tag_name) or pattern.search(val_str):
                    signals.append(f"EXIF[{tag_name}]: {val_str[:120]}")
                    generator = gen_name
                    confidence = conf
    except Exception:
        pass

    # ── 2. XMP / raw string scan (catches C2PA and most AI tools) ───────────
    try:
        raw_str = raw_bytes.decode("latin-1", errors="ignore")
        for pattern, gen_name, conf in _AI_SIGNATURES:
            matches = pattern.findall(raw_str)
            if matches:
                signals.append(f"XMP/raw match: '{matches[0]}' (×{len(matches)})")
                if confidence != "CONFIRMED_AI":  # don't downgrade
                    generator = gen_name
                    confidence = conf
    except Exception:
        pass

    # ── 3. PNG text chunks (tEXt / iTXt / zTXt) ────────────────────────────
    try:
        if hasattr(pil_img, 'text') and pil_img.text:
            for key, val in pil_img.text.items():
                combined = f"{key}: {val}"
                for pattern, gen_name, conf in _AI_SIGNATURES:
                    if pattern.search(combined):
                        signals.append(f"PNG chunk [{key}]: {val[:120]}")
                        generator = gen_name
                        confidence = conf
    except Exception:
        pass

    # ── 4. JFIF / APP marker scan for Google / Adobe markers ────────────────
    try:
        # Look for known binary markers: Google uses 0xFFE1 APP1 with specific UUIDs
        if raw_bytes[:2] == b'\xff\xd8':  # JPEG magic bytes
            pos = 2
            while pos < min(len(raw_bytes), 65536):  # scan first 64KB of headers
                if raw_bytes[pos] != 0xFF:
                    break
                marker = raw_bytes[pos + 1]
                length = struct.unpack('>H', raw_bytes[pos + 2:pos + 4])[0]
                segment = raw_bytes[pos + 4: pos + 2 + length]
                seg_str = segment.decode('latin-1', errors='ignore')
                for pattern, gen_name, conf in _AI_SIGNATURES:
                    if pattern.search(seg_str):
                        signals.append(f"JPEG APP marker 0xFF{marker:02X}: AI signature found")
                        generator = gen_name
                        confidence = conf
                pos += 2 + length
    except Exception:
        pass

    # ── 5. Missing EXIF heuristic (suspicious for AI images) ────────────────
    has_exif = False
    try:
        has_exif = pil_img._getexif() is not None
    except Exception:
        pass

    if not has_exif and not signals:
        signals.append("No EXIF metadata present — suspicious for AI-generated or screenshot")
        confidence = "SUSPICIOUS"

    detected = confidence == "CONFIRMED_AI"

    if detected:
        warning = f"⚠️ AI-Generated image confirmed via metadata — Generator: {generator}"
    elif confidence == "SUSPICIOUS":
        warning = "⚠️ No camera metadata found — possible AI image or screenshot"
    else:
        warning = "✅ Metadata consistent with authentic camera photograph"

    return {
        "detected": detected,
        "generator": generator if detected else None,
        "confidence": confidence,
        "signals": signals,
        "warning": warning,
    }


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

        result   = analyze_image(pil_img)
        meta     = analyze_metadata(raw_bytes, pil_img)

        return {
            "type": "image",
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "manipulation_score": result["fake_pct"],
            "fake_probability": result["fake_pct"],
            "real_probability": result["real_pct"],
            "model": "DINOv2 Three-Stream (Spatial + Frequency + Attention)",
            "heatmap": result["heatmap"],
            "frequency_map": result["frequency_map"],
            "streams": result["streams"],
            "frame_count": 1,
            "timeline": [],
            "metadata_forensics": meta,
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
