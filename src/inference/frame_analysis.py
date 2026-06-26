"""
Frame-level analysis.

Purpose: Detailed frame-by-frame analysis for interpretability.
Responsibilities: Per-frame scoring, manipulation localization, attention mapping.
Dependencies: torch, numpy, src.models, src.preprocessing

Research Traceability:
    Research Objective: Explainable deepfake detection
    Methodology: Frame-level probability analysis and manipulation localization
    Implementation: src/inference/frame_analysis.py
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
class FrameAnalysis:
    """Detailed frame-level analysis result."""

    frame_index: int
    timestamp: float
    fake_probability: float
    real_probability: float
    prediction: str
    confidence: float
    is_manipulated: bool


@dataclass
class VideoAnalysis:
    """Complete video analysis with frame-level details."""

    video_path: str
    total_frames: int
    analyzed_frames: int
    frames: list[FrameAnalysis] = field(default_factory=list)
    manipulation_timeline: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


class FrameAnalyzer:
    """Analyze individual frames for deepfake detection.

    Provides:
    - Per-frame fake/real probability scores
    - Manipulation timeline reconstruction
    - Confidence distribution analysis
    - Frame ranking by manipulation likelihood
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str | None = None,
        threshold: float = 0.5,
    ) -> None:
        """Initialize frame analyzer.

        Args:
            model: Trained frame-level classifier.
            device: Device for inference.
            threshold: Classification threshold.
        """
        if isinstance(device, str):
            device = torch.device(device)
        self.model = model
        self.device = device or get_device()
        self.threshold = threshold

        self.model = self.model.to(self.device)
        self.model.eval()

        self._preprocess = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ])

        logger.info(f"FrameAnalyzer initialized: threshold={threshold}")

    def _load_frames(
        self,
        paths: list[Path],
        indices: list[int],
    ) -> torch.Tensor:
        """Load and preprocess images from disk.

        Args:
            paths: List of image file paths.
            indices: Frame indices corresponding to paths.

        Returns:
            Tensor of shape (N, C, H, W).
        """
        images = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            images.append(self._preprocess(img))
        return torch.stack(images, dim=0)

    @torch.no_grad()
    def analyze_video(
        self,
        paths: list[Path],
        indices: list[int],
        timestamps: list[float],
        video_path: str = "unknown",
    ) -> VideoAnalysis:
        """Analyze video frames loaded from disk.

        Args:
            paths: List of image file paths.
            indices: Frame index for each path.
            timestamps: Timestamp in seconds for each frame.
            video_path: Path to the source video.

        Returns:
            VideoAnalysis with per-frame details.
        """
        frames = self._load_frames(paths, indices)
        return self.analyze_frames(
            frames,
            timestamps=timestamps,
            video_path=video_path,
        )

    @torch.no_grad()
    def analyze_frames(
        self,
        frames: torch.Tensor,
        timestamps: list[float] | None = None,
        video_path: str = "unknown",
    ) -> VideoAnalysis:
        """Analyze a sequence of frames.

        Args:
            frames: Tensor of shape (N, C, H, W).
            timestamps: Optional timestamps for each frame (in seconds).
            video_path: Path to the source video.

        Returns:
            VideoAnalysis with per-frame details.
        """
        frames = frames.to(self.device)
        num_frames = len(frames)

        if timestamps is None:
            timestamps = list(range(num_frames))

        # Forward pass
        outputs = self.model(frames)
        probs = torch.softmax(outputs, dim=1)

        fake_probs = probs[:, 1].cpu().numpy()
        real_probs = probs[:, 0].cpu().numpy()

        # Build per-frame analysis
        frame_analyses = []
        for i in range(num_frames):
            is_fake = fake_probs[i] >= self.threshold
            confidence = fake_probs[i] if is_fake else real_probs[i]

            frame_analyses.append(FrameAnalysis(
                frame_index=i,
                timestamp=timestamps[i],
                fake_probability=float(fake_probs[i]),
                real_probability=float(real_probs[i]),
                prediction="fake" if is_fake else "real",
                confidence=float(confidence),
                is_manipulated=is_fake,
            ))

        # Build manipulation timeline
        timeline = self._build_timeline(frame_analyses)

        # Compute summary statistics
        summary = self._compute_summary(frame_analyses)

        analysis = VideoAnalysis(
            video_path=video_path,
            total_frames=num_frames,
            analyzed_frames=num_frames,
            frames=frame_analyses,
            manipulation_timeline=timeline,
            summary=summary,
        )

        logger.info(
            f"Analyzed {num_frames} frames: "
            f"{summary['manipulated_frames']} manipulated "
            f"({summary['manipulation_ratio']:.1%})"
        )

        return analysis

    def _build_timeline(
        self,
        frame_analyses: list[FrameAnalysis],
    ) -> list[dict[str, Any]]:
        """Build manipulation timeline from frame analyses.

        Args:
            frame_analyses: List of per-frame analyses.

        Returns:
            List of manipulation segments.
        """
        timeline = []
        in_segment = False
        segment_start = 0

        for i, frame in enumerate(frame_analyses):
            if frame.is_manipulated and not in_segment:
                in_segment = True
                segment_start = i
            elif not frame.is_manipulated and in_segment:
                in_segment = False
                timeline.append(self._create_segment(
                    frame_analyses, segment_start, i - 1
                ))

        # Close last segment
        if in_segment:
            timeline.append(self._create_segment(
                frame_analyses, segment_start, len(frame_analyses) - 1
            ))

        return timeline

    def _create_segment(
        self,
        frame_analyses: list[FrameAnalysis],
        start: int,
        end: int,
    ) -> dict[str, Any]:
        """Create a manipulation segment dictionary.

        Args:
            frame_analyses: List of frame analyses.
            start: Start frame index.
            end: End frame index.

        Returns:
            Segment dictionary.
        """
        segment_frames = frame_analyses[start:end + 1]
        probs = [f.fake_probability for f in segment_frames]

        return {
            "start_frame": start,
            "end_frame": end,
            "start_time": frame_analyses[start].timestamp,
            "end_time": frame_analyses[end].timestamp,
            "length_frames": end - start + 1,
            "mean_probability": float(np.mean(probs)),
            "max_probability": float(np.max(probs)),
            "min_probability": float(np.min(probs)),
            "std_probability": float(np.std(probs)),
        }

    def _compute_summary(
        self,
        frame_analyses: list[FrameAnalysis],
    ) -> dict[str, Any]:
        """Compute summary statistics.

        Args:
            frame_analyses: List of frame analyses.

        Returns:
            Summary dictionary.
        """
        fake_probs = [f.fake_probability for f in frame_analyses]
        manipulated = [f for f in frame_analyses if f.is_manipulated]

        # Confidence distribution
        confidences = [f.confidence for f in frame_analyses]

        # Sort frames by manipulation probability
        sorted_frames = sorted(
            frame_analyses,
            key=lambda x: x.fake_probability,
            reverse=True,
        )

        return {
            "total_frames": len(frame_analyses),
            "manipulated_frames": len(manipulated),
            "real_frames": len(frame_analyses) - len(manipulated),
            "manipulation_ratio": len(manipulated) / len(frame_analyses) if frame_analyses else 0.0,
            "mean_fake_probability": float(np.mean(fake_probs)),
            "std_fake_probability": float(np.std(fake_probs)),
            "max_fake_probability": float(np.max(fake_probs)) if fake_probs else 0.0,
            "min_fake_probability": float(np.min(fake_probs)) if fake_probs else 0.0,
            "mean_confidence": float(np.mean(confidences)),
            "std_confidence": float(np.std(confidences)),
            "top_manipulated_frames": [
                {
                    "frame_index": f.frame_index,
                    "timestamp": f.timestamp,
                    "fake_probability": f.fake_probability,
                }
                for f in sorted_frames[:10]
            ],
        }

    def rank_frames(
        self,
        analysis: VideoAnalysis,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Rank frames by manipulation likelihood.

        Args:
            analysis: VideoAnalysis result.
            top_k: Number of top frames to return.

        Returns:
            List of ranked frames.
        """
        sorted_frames = sorted(
            analysis.frames,
            key=lambda x: x.fake_probability,
            reverse=True,
        )

        return [
            {
                "rank": i + 1,
                "frame_index": f.frame_index,
                "timestamp": f.timestamp,
                "fake_probability": f.fake_probability,
                "confidence": f.confidence,
            }
            for i, f in enumerate(sorted_frames[:top_k])
        ]

    def to_dict(self, analysis: VideoAnalysis) -> dict[str, Any]:
        """Convert VideoAnalysis to dictionary.

        Args:
            analysis: VideoAnalysis to convert.

        Returns:
            Dictionary representation.
        """
        return {
            "video_path": analysis.video_path,
            "total_frames": analysis.total_frames,
            "analyzed_frames": analysis.analyzed_frames,
            "frames": [
                {
                    "frame_index": f.frame_index,
                    "timestamp": f.timestamp,
                    "fake_probability": f.fake_probability,
                    "real_probability": f.real_probability,
                    "prediction": f.prediction,
                    "confidence": f.confidence,
                    "is_manipulated": f.is_manipulated,
                }
                for f in analysis.frames
            ],
            "manipulation_timeline": analysis.manipulation_timeline,
            "summary": analysis.summary,
        }
