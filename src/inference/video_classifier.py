"""
Video-level classifier.

Purpose: Aggregate frame-level predictions into video-level classifications.
Responsibilities: Frame aggregation, confidence scoring, temporal analysis.
Dependencies: torch, numpy, src.models, src.preprocessing

Research Traceability:
    Research Objective: Video-level deepfake detection
    Methodology: Frame-level to video-level aggregation (mean, majority, confidence-weighted)
    Implementation: src/inference/video_classifier.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from src.utils.logger import setup_logger
from src.utils.helpers import get_device

logger = setup_logger(__name__)


@dataclass
class VideoPrediction:
    """Video-level prediction result."""

    video_path: str
    is_fake: bool
    confidence: float
    fake_probability: float
    real_probability: float
    num_frames: int
    aggregation_method: str
    frame_predictions: list[dict[str, Any]] = field(default_factory=list)
    temporal_analysis: dict[str, Any] = field(default_factory=dict)


class VideoClassifier:
    """Classify videos as real or fake by aggregating frame predictions.

    Supports:
    - Mean probability aggregation
    - Majority voting
    - Confidence-weighted aggregation
    - Temporal analysis of manipulation patterns
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str | None = None,
        threshold: float = 0.5,
        aggregation_method: str = "mean",
    ) -> None:
        """Initialize video classifier.

        Args:
            model: Trained frame-level classifier.
            device: Device for inference.
            threshold: Classification threshold.
            aggregation_method: Method for frame aggregation ('mean', 'majority', 'confidence_weighted').
        """
        if isinstance(device, str):
            device = torch.device(device)
        self.model = model
        self.device = device or get_device()
        self.threshold = threshold
        self.aggregation_method = aggregation_method

        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        self._preprocess = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ])

        logger.info(
            f"VideoClassifier initialized: "
            f"method={aggregation_method}, threshold={threshold}"
        )

    def _load_and_preprocess(self, paths: list[Path]) -> torch.Tensor:
        """Load images from paths and convert to a batch tensor.

        Args:
            paths: List of image file paths.

        Returns:
            Tensor of shape (N, C, H, W).
        """
        images = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            images.append(self._preprocess(img))
        return torch.stack(images, dim=0)

    @torch.no_grad()
    def classify_frames(
        self,
        frames: list[Path] | torch.Tensor,
    ) -> list[dict[str, Any]]:
        """Classify a batch of frames.

        Args:
            frames: List of image file paths, or a preprocessed tensor of shape (N, C, H, W).

        Returns:
            List of dictionaries, one per frame, each with 'fake_prob', 'real_prob', 'label'.
        """
        if isinstance(frames, list):
            tensor = self._load_and_preprocess(frames).to(self.device)
        else:
            tensor = frames.to(self.device)

        # Forward pass
        outputs = self.model(tensor)
        probs = torch.softmax(outputs, dim=1)

        fake_probs = probs[:, 1].cpu().numpy()
        real_probs = probs[:, 0].cpu().numpy()

        results = []
        for i in range(len(fake_probs)):
            fake_p = float(fake_probs[i])
            real_p = float(real_probs[i])
            label = "fake" if fake_p >= self.threshold else "real"
            results.append({
                "fake_prob": fake_p,
                "real_prob": real_p,
                "label": label,
            })

        return results

    def aggregate_predictions(
        self,
        frame_results: list[dict[str, Any]],
    ) -> tuple[bool, float]:
        """Aggregate frame predictions into video-level prediction.

        Args:
            frame_results: List of per-frame result dicts with 'fake_prob' keys.

        Returns:
            Tuple of (is_fake, confidence).
        """
        fake_probs = np.array([r["fake_prob"] for r in frame_results])

        if self.aggregation_method == "mean":
            video_prob = float(np.mean(fake_probs))

        elif self.aggregation_method == "majority":
            preds = np.array([1 if r["label"] == "fake" else 0 for r in frame_results])
            video_prob = float(np.mean(preds))

        elif self.aggregation_method == "confidence_weighted":
            confidences = np.abs(fake_probs - 0.5) * 2
            weights = confidences / confidences.sum() if confidences.sum() > 0 else np.ones_like(fake_probs) / len(fake_probs)
            video_prob = float(np.sum(fake_probs * weights))

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        is_fake = video_prob >= self.threshold
        confidence = video_prob if is_fake else 1 - video_prob

        return is_fake, confidence

    def classify_video(
        self,
        frames: list[Path] | torch.Tensor,
        video_path: str = "unknown",
    ) -> VideoPrediction:
        """Classify a complete video from its frames.

        Args:
            frames: List of frame image paths or preprocessed tensor (N, C, H, W).
            video_path: Path to the source video.

        Returns:
            VideoPrediction with video-level classification.
        """
        # Classify all frames
        frame_results = self.classify_frames(frames)

        # Aggregate to video level
        is_fake, confidence = self.aggregate_predictions(frame_results)

        # Build per-frame predictions
        frame_predictions = []
        for i, r in enumerate(frame_results):
            frame_predictions.append({
                "frame_index": i,
                "prediction": r["label"],
                "fake_probability": r["fake_prob"],
                "real_probability": r["real_prob"],
            })

        # Temporal analysis
        temporal_analysis = self._temporal_analysis(frame_results)

        num_frames = len(frame_results)
        mean_fake = float(np.mean([r["fake_prob"] for r in frame_results]))
        mean_real = float(np.mean([r["real_prob"] for r in frame_results]))

        prediction = VideoPrediction(
            video_path=video_path,
            is_fake=is_fake,
            confidence=confidence,
            fake_probability=mean_fake,
            real_probability=mean_real,
            num_frames=num_frames,
            aggregation_method=self.aggregation_method,
            frame_predictions=frame_predictions,
            temporal_analysis=temporal_analysis,
        )

        logger.info(
            f"Video {video_path}: {'FAKE' if is_fake else 'REAL'} "
            f"(confidence={confidence:.4f}, frames={num_frames})"
        )

        return prediction

    def _temporal_analysis(self, frame_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze temporal patterns in frame predictions.

        Args:
            frame_results: List of per-frame result dicts.

        Returns:
            Dictionary with temporal analysis metrics.
        """
        fake_probs = np.array([r["fake_prob"] for r in frame_results])

        mean_prob = float(np.mean(fake_probs))
        std_prob = float(np.std(fake_probs))
        max_prob = float(np.max(fake_probs))
        min_prob = float(np.min(fake_probs))

        manipulation_segments = []
        in_segment = False
        segment_start = 0

        for i, prob in enumerate(fake_probs):
            if prob >= self.threshold and not in_segment:
                in_segment = True
                segment_start = i
            elif prob < self.threshold and in_segment:
                in_segment = False
                manipulation_segments.append({
                    "start_frame": segment_start,
                    "end_frame": i - 1,
                    "length": i - segment_start,
                    "mean_probability": float(np.mean(fake_probs[segment_start:i])),
                })

        if in_segment:
            manipulation_segments.append({
                "start_frame": segment_start,
                "end_frame": len(fake_probs) - 1,
                "length": len(fake_probs) - segment_start,
                "mean_probability": float(np.mean(fake_probs[segment_start:])),
            })

        consistency = 1.0 - std_prob

        return {
            "mean_probability": mean_prob,
            "std_probability": std_prob,
            "max_probability": max_prob,
            "min_probability": min_prob,
            "consistency_score": float(consistency),
            "num_manipulation_segments": len(manipulation_segments),
            "manipulation_segments": manipulation_segments,
            "total_manipulated_frames": sum(s["length"] for s in manipulation_segments),
            "manipulation_ratio": (
                sum(s["length"] for s in manipulation_segments) / len(fake_probs)
                if len(fake_probs) > 0
                else 0.0
            ),
        }

    def to_dict(self, prediction: VideoPrediction) -> dict[str, Any]:
        """Convert VideoPrediction to dictionary.

        Args:
            prediction: VideoPrediction to convert.

        Returns:
            Dictionary representation.
        """
        return {
            "video_path": prediction.video_path,
            "is_fake": prediction.is_fake,
            "classification": "FAKE" if prediction.is_fake else "REAL",
            "confidence": prediction.confidence,
            "fake_probability": prediction.fake_probability,
            "real_probability": prediction.real_probability,
            "num_frames": prediction.num_frames,
            "aggregation_method": prediction.aggregation_method,
            "frame_predictions": prediction.frame_predictions,
            "temporal_analysis": prediction.temporal_analysis,
        }
