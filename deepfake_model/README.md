# Deepfake Forensics — Silent Trails

Multi-stream deepfake detection model for the Silent Trails forensics platform.

## Architecture

| Stream | Method | Detects |
|---|---|---|
| Spatial | DINOv2 + LoRA | Texture, identity artifacts |
| Frequency | FFT → CNN | GAN checkerboard, spectral patterns |
| Attention | Region queries | Eye/mouth/jaw manipulation regions |
| Temporal | Transformer (video) | Flicker, identity drift, blinking |

Streams are combined via **Adaptive Fusion** which dynamically weights contributions based on predicted input quality (compression, blur, motion).

## Quick Start

```bash
# Install Dependencies
pip install -r requirements.txt

# Face extraction
python -m src.data.face_extractor --input /path/to/dataset --output /path/to/faces

# Training (in Colab/Modal)
python train.py --config configs/train_config.yaml

# Inference
python -m src.inference.predictor --image /path/to/image.jpg --checkpoint best_model.pth
```

## Project Structure

```
deepfake_model/
├── configs/              # Training configurations
├── src/
│   ├── models/           # Spatial, Frequency, Attention, Fusion, Temporal
│   ├── data/             # Face extraction, dataset, augmentation
│   ├── training/         # Losses, metrics
│   └── inference/        # Predictor, Grad-CAM
├── checkpoints/          # Saved model weights
├── requirements.txt
└── README.md
```

## Training Pipeline

1. **Face Extraction**: RetinaFace → aligned 256×256 crops
2. **Curriculum Training**: Easy → Mixed → Chaos augmentation
3. **Compound Loss**: BCE + Focal + Triplet + Quality MSE
4. **Evaluation**: AUC-ROC, F1, EER on cross-dataset benchmarks

## Hardware

- **Free GPU options**: Colab Free, Kaggle (30h/week), Modal ($30 free credits)
- **VRAM**: ~6-7 GB in FP16 (fits on T4/A10G)
