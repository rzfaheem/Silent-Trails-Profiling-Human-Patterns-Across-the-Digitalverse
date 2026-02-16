"""
Face Extraction Pipeline — RetinaFace detector + alignment.

Extracts, aligns, and crops faces from images and video frames.
Used as the first step in the data preprocessing pipeline.

For video: samples N frames uniformly, extracts face from each.
"""

import os
import cv2
import numpy as np
from pathlib import Path


class FaceExtractor:
    """
    Face detection and alignment using InsightFace/RetinaFace.

    Produces 256×256 aligned face crops suitable for model input.
    """

    def __init__(self, det_size=(640, 640), min_score=0.7, output_size=256):
        self.det_size = det_size
        self.min_score = min_score
        self.output_size = output_size
        self.app = None  # lazy init — avoid import errors when not needed

    def _init_detector(self):
        """Lazy-initialize the InsightFace app (requires GPU)."""
        if self.app is not None:
            return

        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=self.det_size)

    def extract_from_image(self, image):
        """
        Extract the largest face from an image.

        Args:
            image: numpy array (H, W, 3) BGR format

        Returns:
            face_crop: (output_size, output_size, 3) aligned face, or None
            landmarks: (5, 2) facial landmarks, or None
            score: detection confidence
        """
        self._init_detector()

        faces = self.app.get(image)

        if not faces:
            return None, None, 0.0

        # Sort by detection score, take the best face
        faces = sorted(faces, key=lambda f: f.det_score, reverse=True)
        face = faces[0]

        if face.det_score < self.min_score:
            return None, None, face.det_score

        # Align and crop face
        crop = self._align_face(image, face.kps)
        return crop, face.kps, face.det_score

    def extract_from_path(self, image_path):
        """Extract face from an image file path."""
        image = cv2.imread(str(image_path))
        if image is None:
            return None, None, 0.0
        return self.extract_from_image(image)

    def extract_from_video(self, video_path, sample_n=16):
        """
        Extract faces from uniformly sampled video frames.

        Args:
            video_path: path to video file
            sample_n: number of frames to sample

        Returns:
            list of (face_crop, score) tuples
        """
        frames = self._load_video_frames(video_path)
        if not frames:
            return []

        # Uniform sampling
        indices = np.linspace(0, len(frames) - 1, sample_n, dtype=int)
        results = []

        for i in indices:
            crop, _, score = self.extract_from_image(frames[i])
            if crop is not None:
                results.append((crop, score))

        return results

    def process_dataset_folder(self, input_dir, output_dir, video_mode=False, sample_n=16):
        """
        Batch process an entire dataset folder.

        Args:
            input_dir: directory containing images or videos
            output_dir: directory to save face crops
            video_mode: if True, treat files as videos
            sample_n: frames per video

        Returns:
            dict with stats: processed, skipped, failed
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stats = {"processed": 0, "skipped": 0, "failed": 0}

        if video_mode:
            extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        else:
            extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        files = [f for f in input_dir.rglob("*") if f.suffix.lower() in extensions]

        for filepath in files:
            try:
                if video_mode:
                    frame_faces = self.extract_from_video(filepath, sample_n=sample_n)
                    if not frame_faces:
                        stats["skipped"] += 1
                        continue

                    # Save each frame face
                    for idx, (crop, score) in enumerate(frame_faces):
                        rel_path = filepath.relative_to(input_dir)
                        out_name = f"{rel_path.stem}_frame{idx:03d}.jpg"
                        out_path = output_dir / rel_path.parent / out_name
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(out_path), crop)
                    stats["processed"] += 1
                else:
                    crop, _, score = self.extract_from_path(filepath)
                    if crop is None:
                        stats["skipped"] += 1
                        continue

                    rel_path = filepath.relative_to(input_dir)
                    out_path = output_dir / rel_path
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_path), crop)
                    stats["processed"] += 1

            except Exception as e:
                print(f"  [FAIL] {filepath}: {e}")
                stats["failed"] += 1

        return stats

    def _align_face(self, image, landmarks):
        """
        Align face using 5-point landmarks (similarity transform).

        Args:
            image: (H, W, 3) BGR image
            landmarks: (5, 2) facial landmarks

        Returns:
            aligned: (output_size, output_size, 3) aligned face crop
        """
        # Reference landmarks for 256x256 aligned face
        size = self.output_size
        ref = np.array([
            [0.34191607 * size, 0.46157411 * size],
            [0.65653393 * size, 0.45983393 * size],
            [0.50022500 * size, 0.64050536 * size],
            [0.37097589 * size, 0.82469196 * size],
            [0.63151696 * size, 0.82325089 * size],
        ], dtype=np.float32)

        src = np.array(landmarks, dtype=np.float32)

        # Compute similarity transform
        transform = cv2.estimateAffinePartial2D(src, ref)[0]

        if transform is None:
            # Fallback: simple bounding box crop
            x1 = max(0, int(landmarks[:, 0].min() - size * 0.3))
            y1 = max(0, int(landmarks[:, 1].min() - size * 0.3))
            x2 = min(image.shape[1], int(landmarks[:, 0].max() + size * 0.3))
            y2 = min(image.shape[0], int(landmarks[:, 1].max() + size * 0.3))
            crop = image[y1:y2, x1:x2]
            return cv2.resize(crop, (size, size))

        aligned = cv2.warpAffine(image, transform, (size, size))
        return aligned

    def _load_video_frames(self, video_path):
        """Load all frames from a video file."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        return frames
