"""
Video prediction pipeline.

Purpose: End-to-end video deepfake prediction.
Responsibilities: Video loading, frame extraction, preprocessing, classification.
Dependencies: torch, opencv-python, src.preprocessing, src.inference.video_classifier

Research Traceability:
    Research Objective: Complete video-level deepfake detection
    Methodology: Frame extraction → face detection → model inference → aggregation
    Implementation: src/inference/predict_video.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import transforms

from src.inference.video_classifier import VideoClassifier, VideoPrediction
from src.preprocessing.frame_extractor import FrameExtractor
from src.preprocessing.face_detector import FaceDetector
from src.preprocessing.face_cropper import FaceCropper
from src.preprocessing.image_resizer import ImageResizer
from src.preprocessing.normalizer import ImageNormalizer
from src.utils.logger import setup_logger
from src.utils.helpers import get_device

logger = setup_logger(__name__)


class VideoPredictor:
    """End-to-end video deepfake prediction.

    Pipeline:
    1. Extract frames from video
    2. Detect and crop faces
    3. Preprocess faces for model input
    4. Classify each frame
    5. Aggregate frame predictions to video level
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | None = None,
        threshold: float = 0.5,
        aggregation_method: str = "mean",
        frame_sample_rate: int = 5,
        target_size: int = 299,
    ) -> None:
        """Initialize video predictor.

        Args:
            model: Trained frame-level classifier.
            device: Device for inference.
            threshold: Classification threshold.
            aggregation_method: Frame aggregation method.
            frame_sample_rate: Extract 1 frame every N frames.
            target_size: Target image size for model input.
        """
        self.device = device or get_device()
        self.frame_sample_rate = frame_sample_rate
        self.target_size = target_size

        # Initialize pipeline components
        self.frame_extractor = FrameExtractor(sampling_rate=frame_sample_rate)
        self.face_detector = FaceDetector()
        self.face_cropper = FaceCropper()
        self.image_resizer = ImageResizer(target_size=target_size)
        self.normalizer = ImageNormalizer()

        # Initialize video classifier
        self.classifier = VideoClassifier(
            model=model,
            device=self.device,
            threshold=threshold,
            aggregation_method=aggregation_method,
        )

        logger.info(
            f"VideoPredictor initialized: "
            f"sample_rate={frame_sample_rate}, target_size={target_size}"
        )

    def predict(
        self,
        video_path: str | Path,
    ) -> VideoPrediction:
        """Predict whether a video is real or fake.

        Args:
            video_path: Path to the video file.

        Returns:
            VideoPrediction with classification result.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        logger.info(f"Processing video: {video_path}")

        # Step 1: Extract frames
        frames, frame_indices = self.frame_extractor.extract(video_path)
        if len(frames) == 0:
            raise ValueError(f"No frames extracted from {video_path}")

        logger.info(f"Extracted {len(frames)} frames")

        # Step 2: Detect and crop faces
        face_frames = []
        valid_indices = []

        for i, frame in enumerate(frames):
            faces = self.face_detector.detect(frame)
            if len(faces) > 0:
                # Crop the largest face
                face = self.face_cropper.crop(frame, faces[0])
                face_frames.append(face)
                valid_indices.append(frame_indices[i])

        if len(face_frames) == 0:
            logger.warning(f"No faces detected in {video_path}")
            # Fall back to using full frames
            face_frames = frames
            valid_indices = frame_indices

        logger.info(f"Detected {len(face_frames)} faces")

        # Step 3: Preprocess
        processed_frames = []
        for face in face_frames:
            # Resize
            resized = self.image_resizer.resize(face)
            # Normalize
            normalized = self.normalizer.normalize(resized)
            processed_frames.append(normalized)

        # Stack into batch tensor
        frames_tensor = torch.stack(processed_frames)

        # Step 4 & 5: Classify and aggregate
        prediction = self.classifier.classify_video(
            frames_tensor,
            video_path=str(video_path),
        )

        return prediction

    def predict_batch(
        self,
        video_paths: list[str | Path],
    ) -> list[VideoPrediction]:
        """Predict for multiple videos.

        Args:
            video_paths: List of video file paths.

        Returns:
            List of VideoPrediction results.
        """
        predictions = []
        for video_path in video_paths:
            try:
                prediction = self.predict(video_path)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
                # Create error prediction
                predictions.append(VideoPrediction(
                    video_path=str(video_path),
                    is_fake=False,
                    confidence=0.0,
                    fake_probability=0.0,
                    real_probability=1.0,
                    num_frames=0,
                    aggregation_method="error",
                ))

        return predictions

    def to_dict(self, prediction: VideoPrediction) -> dict[str, Any]:
        """Convert prediction to dictionary.

        Args:
            prediction: VideoPrediction to convert.

        Returns:
            Dictionary representation.
        """
        return self.classifier.to_dict(prediction)
