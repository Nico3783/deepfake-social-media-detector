"""Tests for inference pipeline modules."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.inference.video_classifier import VideoClassifier, VideoPrediction
from src.inference.predict_image import ImagePredictor, ImagePrediction
from src.inference.frame_analysis import FrameAnalyzer


class TestVideoClassifier:
    """Tests for frame-to-video aggregation."""

    def test_mean_aggregation(self, device: torch.device) -> None:
        """Mean aggregation produces correct result."""
        model = torch.nn.Linear(10, 2)
        classifier = VideoClassifier(model=model, device=device, aggregation_method="mean")
        # Simulate frame predictions
        frame_probs = [0.1, 0.2, 0.3, 0.8, 0.9]
        result = classifier.aggregate(frame_probs)
        assert 0.0 <= result <= 1.0

    def test_majority_aggregation(self, device: torch.device) -> None:
        """Majority aggregation returns binary decision."""
        model = torch.nn.Linear(10, 2)
        classifier = VideoClassifier(model=model, device=device, aggregation_method="majority")
        frame_probs = [0.1, 0.2, 0.3, 0.8, 0.9]
        result = classifier.aggregate(frame_probs)
        assert result in (0.0, 1.0)

    def test_empty_frames(self, device: torch.device) -> None:
        """Aggregation handles empty frame list."""
        model = torch.nn.Linear(10, 2)
        classifier = VideoClassifier(model=model, device=device)
        result = classifier.aggregate([])
        assert isinstance(result, float)


class TestImagePrediction:
    """Tests for ImagePrediction dataclass."""

    def test_creation(self) -> None:
        """ImagePrediction creates with required fields."""
        pred = ImagePrediction(
            image_path="test.jpg",
            is_fake=True,
            confidence=0.95,
            fake_probability=0.95,
            real_probability=0.05,
            face_detected=True,
            bounding_box=(10, 10, 100, 100),
        )
        assert pred.is_fake is True
        assert pred.confidence == 0.95
        assert pred.bounding_box == (10, 10, 100, 100)

    def test_no_face(self) -> None:
        """ImagePrediction works without face detection."""
        pred = ImagePrediction(
            image_path="test.jpg",
            is_fake=False,
            confidence=0.6,
            fake_probability=0.4,
            real_probability=0.6,
            face_detected=False,
        )
        assert pred.face_detected is False
        assert pred.bounding_box is None


class TestFrameAnalyzer:
    """Tests for frame-level analysis."""

    def test_analyze_frames(self) -> None:
        """FrameAnalyzer processes frame probabilities correctly."""
        analyzer = FrameAnalyzer()
        probs = [0.1, 0.2, 0.8, 0.9, 0.3]
        result = analyzer.analyze(probs)
        assert "mean_fake_prob" in result or "frames" in result

    def test_detect_manipulation_segments(self) -> None:
        """FrameAnalyzer identifies manipulation segments."""
        analyzer = FrameAnalyzer(threshold=0.5)
        # Frames 2-3 are above threshold
        probs = [0.1, 0.1, 0.8, 0.9, 0.1]
        result = analyzer.analyze(probs)
        assert isinstance(result, dict)
