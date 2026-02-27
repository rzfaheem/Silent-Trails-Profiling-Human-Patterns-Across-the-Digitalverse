"""
=============================================================
 Silent Trails — Deepfake Forensics: Setup Notebook
 Run this in Google Colab (GPU runtime required)
=============================================================

Instructions:
1. Open Google Colab: https://colab.research.google.com
2. Create a new notebook
3. Change Runtime to GPU: Runtime → Change runtime type → T4 GPU
4. Copy each section below into a separate Colab cell
5. Run cells in order

=============================================================
"""

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 1: Check GPU & Clone Repo                         ║
# ╚══════════════════════════════════════════════════════════╝

# --- Paste this into Colab Cell 1 ---
"""
!nvidia-smi
!git clone https://github.com/rzfaheem/Silent-Trails-Profiling-Human-Patterns-Across-the-Digitalverse.git /content/silent-trails
%cd /content/silent-trails/deepfake_model
!ls
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 2: Install Dependencies                           ║
# ╚══════════════════════════════════════════════════════════╝

# --- Paste this into Colab Cell 2 ---
"""
!pip install -q torch torchvision transformers peft
!pip install -q insightface onnxruntime-gpu
!pip install -q opencv-python albumentations
!pip install -q scikit-learn scipy tqdm wandb
!pip install -q fastapi uvicorn python-multipart
!pip install -q matplotlib seaborn

# Verify
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 3: Mount Google Drive (for saving data/models)    ║
# ╚══════════════════════════════════════════════════════════╝

# --- Paste this into Colab Cell 3 ---
"""
from google.colab import drive
drive.mount('/content/drive')

# Create project folders on Google Drive
import os
BASE_DIR = '/content/drive/MyDrive/SilentTrails'
DATA_DIR = f'{BASE_DIR}/data'
CHECKPOINT_DIR = f'{BASE_DIR}/checkpoints'

os.makedirs(f'{DATA_DIR}/ff++/real', exist_ok=True)
os.makedirs(f'{DATA_DIR}/ff++/fake', exist_ok=True)
os.makedirs(f'{DATA_DIR}/celebdf/real', exist_ok=True)
os.makedirs(f'{DATA_DIR}/celebdf/fake', exist_ok=True)
os.makedirs(f'{DATA_DIR}/dfdc/real', exist_ok=True)
os.makedirs(f'{DATA_DIR}/dfdc/fake', exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"Data dir: {DATA_DIR}")
print(f"Checkpoints: {CHECKPOINT_DIR}")
print("Folders created on Google Drive!")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 4: Test Model Loads Correctly                     ║
# ╚══════════════════════════════════════════════════════════╝

# --- Paste this into Colab Cell 4 ---
"""
import sys
sys.path.insert(0, '/content/silent-trails/deepfake_model')

import torch
from src.models import DeepfakeForensicsModel

# Create model (image mode)
model = DeepfakeForensicsModel(video_mode=False)
model = model.cuda()

# Print parameter counts
params = model.count_parameters()
print(f"Total parameters:     {params['total']:,}")
print(f"Trainable parameters: {params['trainable']:,}")
print(f"Frozen parameters:    {params['frozen']:,}")
print(f"Trainable %:          {params['trainable_pct']}%")

# Test forward pass with dummy input
dummy = torch.randn(2, 3, 256, 256).cuda()  # batch of 2 face crops
with torch.no_grad():
    output = model(dummy)

print(f"\\nLogits shape: {output['logits'].shape}")        # (2, 1)
print(f"Probability: {output['probability'].squeeze()}")    # 2 values
print(f"Embedding shape: {output['embedding'].shape}")      # (2, 128)
print(f"Attn maps shape: {output['attn_maps'].shape}")      # (2, 4, 256)
print(f"Quality shape: {output['quality'].shape}")           # (2, 3)

print("\\n✅ Model loads and runs correctly!")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 5: Test Video Mode                                ║
# ╚══════════════════════════════════════════════════════════╝

# --- Paste this into Colab Cell 5 ---
"""
# Test video model
video_model = DeepfakeForensicsModel(video_mode=True).cuda()

# Simulate 16 frames
dummy_video = torch.randn(1, 16, 3, 256, 256).cuda()
with torch.no_grad():
    video_output = video_model(dummy_video)

print(f"Video logits: {video_output['logits'].shape}")        # (1, 1)
print(f"Video probability: {video_output['probability'].item():.4f}")
print(f"Video embedding: {video_output['embedding'].shape}")  # (1, 128)

print("\\n✅ Video model works!")

# Clean up GPU memory
del video_model, dummy_video
torch.cuda.empty_cache()
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 6: Test Loss Function                             ║
# ╚══════════════════════════════════════════════════════════╝

# --- Paste this into Colab Cell 6 ---
"""
from src.training import CompoundLoss

criterion = CompoundLoss(
    bce_weight=1.0,
    focal_weight=0.5,
    triplet_weight=0.5,
    quality_weight=0.1,
)

# Simulate a batch
dummy_input = torch.randn(8, 3, 256, 256).cuda()
dummy_labels = torch.tensor([0, 0, 1, 1, 0, 1, 1, 0]).cuda()  # 4 real, 4 fake
dummy_quality = torch.rand(8, 3).cuda()  # random quality labels

with torch.cuda.amp.autocast():
    output = model(dummy_input)
    loss, loss_dict = criterion(output, dummy_labels, dummy_quality)

print(f"Total loss: {loss.item():.4f}")
for k, v in loss_dict.items():
    print(f"  {k}: {v:.4f}")

print("\\n✅ Loss function works!")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 7: Test Augmentation Pipeline                     ║
# ╚══════════════════════════════════════════════════════════╝

# --- Paste this into Colab Cell 7 ---
"""
from src.data import get_train_transforms, get_val_transforms
import numpy as np

train_transform = get_train_transforms(face_size=256)
val_transform = get_val_transforms(face_size=256)

# Create a dummy face image
dummy_face = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

# Apply augmentation
augmented = train_transform(image=dummy_face)
print(f"Augmented shape: {augmented['image'].shape}")  # (3, 256, 256)
print(f"Min: {augmented['image'].min():.2f}, Max: {augmented['image'].max():.2f}")

print("\\n✅ Augmentation pipeline works!")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 8: Download CelebDF-v2 (if you have the link)    ║
# ╚══════════════════════════════════════════════════════════╝

# --- Paste this into Colab Cell 8 ---
# REPLACE the Google Drive link with your actual CelebDF download link
"""
# Option A: If CelebDF is shared via Google Drive link
# !gdown "YOUR_GOOGLE_DRIVE_LINK_HERE" -O /content/celebdf.zip

# Option B: If you already uploaded it to your Drive
# !cp /content/drive/MyDrive/celebdf.zip /content/

# Unzip
# !unzip -q /content/celebdf.zip -d /content/celebdf_raw

print("Update this cell with your actual CelebDF download link")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 9: Download FF++ (using the script from email)   ║
# ╚══════════════════════════════════════════════════════════╝

# --- Paste this into Colab Cell 9 ---
"""
# Upload the download script you received via email
# Then run:
# !python download-FaceForensics.py /content/ff++_raw -d all -c c23 -t videos

# For compressed version (robustness testing):
# !python download-FaceForensics.py /content/ff++_raw_c40 -d all -c c40 -t videos

print("Upload your FF++ download script and uncomment the commands above")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 10: Download DFDC from Kaggle                    ║
# ╚══════════════════════════════════════════════════════════╝

# --- Paste this into Colab Cell 10 ---
"""
# Setup Kaggle API
# First: go to kaggle.com → Account → Create New API Token
# This downloads kaggle.json — upload it here

from google.colab import files
# files.upload()  # Uncomment to upload kaggle.json

# !mkdir -p ~/.kaggle
# !mv kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json

# Download DFDC parts 0-4 only
# !kaggle competitions download -c deepfake-detection-challenge -f dfdc_train_part_0.tar
# !kaggle competitions download -c deepfake-detection-challenge -f dfdc_train_part_1.tar
# !kaggle competitions download -c deepfake-detection-challenge -f dfdc_train_part_2.tar
# !kaggle competitions download -c deepfake-detection-challenge -f dfdc_train_part_3.tar
# !kaggle competitions download -c deepfake-detection-challenge -f dfdc_train_part_4.tar

print("Setup Kaggle API credentials and uncomment download commands")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 11: Run Face Extraction                          ║
# ╚══════════════════════════════════════════════════════════╝

# --- Paste this into Colab Cell 11 ---
"""
from src.data import FaceExtractor

extractor = FaceExtractor(output_size=256, min_score=0.7)

# Process CelebDF
# stats = extractor.process_dataset_folder(
#     input_dir='/content/celebdf_raw/real',
#     output_dir=f'{DATA_DIR}/celebdf/real',
#     video_mode=True, sample_n=16,
# )
# print(f"CelebDF Real: {stats}")

# stats = extractor.process_dataset_folder(
#     input_dir='/content/celebdf_raw/fake',
#     output_dir=f'{DATA_DIR}/celebdf/fake',
#     video_mode=True, sample_n=16,
# )
# print(f"CelebDF Fake: {stats}")

print("Uncomment and run after datasets are downloaded")
print("Face crops will be saved to Google Drive")
"""


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 12: Verify Data on Drive                         ║
# ╚══════════════════════════════════════════════════════════╝

# --- Paste this into Colab Cell 12 ---
"""
import os

for dataset in ['ff++', 'celebdf', 'dfdc']:
    for split in ['real', 'fake']:
        path = f'{DATA_DIR}/{dataset}/{split}'
        if os.path.exists(path):
            count = sum(1 for f in os.listdir(path) if f.endswith('.jpg'))
            print(f"  {dataset}/{split}: {count} face crops")
        else:
            print(f"  {dataset}/{split}: NOT YET PROCESSED")
"""
