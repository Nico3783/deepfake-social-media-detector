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
        device: torch.device | None = None,
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
        self.model = model
        self.device = device or get_device()
        self.threshold = threshold
        self.aggregation_method = aggregation_method

        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(
            f"VideoClassifier initialized: "
            f"method={aggregation_method}, threshold={threshold}"
        )

    @torch.no_grad()
    def classify_frames(
        self,
        frames: torch.Tensor,
    ) -> dict[str, Any]:
        """Classify a batch of frames.

        Args:
            frames: Tensor of shape (N, C, H, W) where N is number of frames.

        Returns:
            Dictionary with frame-level predictions and probabilities.
        """
        frames = frames.to(self.device)

        # Forward pass
        outputs = self.model(frames)
        probs = torch.softmax(outputs, dim=1)

        # Extract fake class probabilities
        fake_probs = probs[:, 1].cpu().numpy()
        real_probs = probs[:, 0].cpu().numpy()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        return {
            "predictions": preds.tolist(),
            "fake_probabilities": fake_probs.tolist(),
            "real_probabilities": real_probs.tolist(),
            "mean_fake_probability": float(np.mean(fake_probs)),
            "mean_real_probability": float(np.mean(real_probs)),
            "num_frames": len(frames),
        }

    def aggregate_predictions(
        self,
        frame_results: dict[str, Any],
    ) -> tuple[bool, float]:
        """Aggregate frame predictions into video-level prediction.

        Args:
            frame_results: Results from classify_frames().

        Returns:
            Tuple of (is_fake, confidence).
        """
        fake_probs = np.array(frame_results["fake_probabilities"])

        if self.aggregation_method == "mean":
            # Simple mean of frame probabilities
            video_prob = float(np.mean(fake_probs))

        elif self.aggregation_method == "majority":
            # Majority voting
            preds = np.array(frame_results["predictions"])
            video_prob = float(np.mean(preds == 1))

        elif self.aggregation_method == "confidence_weighted":
            # Weight by confidence (distance from 0.5)
            confidences = np.abs(fake_probs - 0.5) * 2  # Scale to [0, 1]
            weights = confidences / confidences.sum() if confidences.sum() > 0 else np.ones_like(fake_probs) / len(fake_probs)
            video_prob = float(np.sum(fake_probs * weights))

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        # Classify
        is_fake = video_prob >= self.threshold
        confidence = video_prob if is_fake else 1 - video_prob

        return is_fake, confidence

    def classify_video(
        self,
        frames: torch.Tensor,
        video_path: str = "unknown",
    ) -> VideoPrediction:
        """Classify a complete video from its frames.

        Args:
            frames: Tensor of extracted frames (N, C, H, W).
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
        for i in range(frame_results["num_frames"]):
            frame_predictions.append({
                "frame_index": i,
                "prediction": "fake" if frame_results["predictions"][i] == 1 else "real",
                "fake_probability": frame_results["fake_probabilities"][i],
                "real_probability": frame_results["real_probabilities"][i],
            })

        # Temporal analysis
        temporal_analysis = self._temporal_analysis(frame_results)

        prediction = VideoPrediction(
            video_path=video_path,
            is_fake=is_fake,
            confidence=confidence,
            fake_probability=frame_results["mean_fake_probability"],
            real_probability=frame_results["mean_real_probability"],
            num_frames=frame_results["num_frames"],
            aggregation_method=self.aggregation_method,
            frame_predictions=frame_predictions,
            temporal_analysis=temporal_analysis,
        )

        logger.info(
            f"Video {video_path}: {'FAKE' if is_fake else 'REAL'} "
            f"(confidence={confidence:.4f}, frames={frame_results['num_frames']})"
        )

        return prediction

    def _temporal_analysis(self, frame_results: dict[str, Any]) -> dict[str, Any]:
        """Analyze temporal patterns in frame predictions.

        Args:
            frame_results: Frame-level classification results.

        Returns:
            Dictionary with temporal analysis metrics.
        """
        fake_probs = np.array(frame_results["fake_probabilities"])

        # Compute temporal statistics
        mean_prob = float(np.mean(fake_probs))
        std_prob = float(np.std(fake_probs))
        max_prob = float(np.max(fake_probs))
        min_prob = float(np.min(fake_probs))

        # Detect manipulation segments (consecutive frames with high fake probability)
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

        # Close last segment if still open
        if in_segment:
            manipulation_segments.append({
                "start_frame": segment_start,
                "end_frame": len(fake_probs) - 1,
                "length": len(fake_probs) - segment_start,
                "mean_probability": float(np.mean(fake_probs[segment_start:])),
            })

        # Compute consistency score (how consistent are the predictions)
        consistency = 1.0 - std_prob  # Higher consistency = lower std

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
