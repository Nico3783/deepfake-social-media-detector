"""Tests for inference modules: VideoClassifier, FrameAnalyzer, ImagePredictor."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.inference.video_classifier import VideoClassifier, VideoPrediction
from src.inference.frame_analysis import FrameAnalyzer, FrameAnalysis, VideoAnalysis
from src.inference.predict_image import ImagePredictor, ImagePrediction


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_model() -> nn.Module:
    """Minimal CNN for inference tests."""
    from src.models.xception import XceptionNet
    return XceptionNet(num_classes=2, pretrained=False)


@pytest.fixture
def fake_frame_paths(tmp_path: Path) -> list[Path]:
    """Create a few small test images and return their paths."""
    paths = []
    for i in range(5):
        p = tmp_path / f"frame_{i}.jpg"
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(p)
        paths.append(p)
    return paths


@pytest.fixture
def fake_image(tmp_path: Path) -> Path:
    """Create a single test image."""
    p = tmp_path / "test.jpg"
    img = Image.fromarray(np.random.randint(0, 255, (299, 299, 3), dtype=np.uint8))
    img.save(p)
    return p


# ---------------------------------------------------------------------------
# Dataclass structure
# ---------------------------------------------------------------------------


class TestVideoPredictionDataclass:
    """Verify VideoPrediction dataclass fields."""

    def test_has_required_fields(self) -> None:
        names = {f.name for f in fields(VideoPrediction)}
        expected = {
            "video_path", "is_fake", "confidence", "fake_probability",
            "real_probability", "num_frames", "aggregation_method",
            "frame_predictions", "temporal_analysis",
        }
        assert expected.issubset(names)


class TestFrameAnalysisDataclass:
    """Verify FrameAnalysis dataclass fields."""

    def test_has_required_fields(self) -> None:
        names = {f.name for f in fields(FrameAnalysis)}
        expected = {
            "frame_index", "timestamp", "fake_probability", "real_probability",
            "prediction", "confidence", "is_manipulated",
        }
        assert expected.issubset(names)


class TestVideoAnalysisDataclass:
    """Verify VideoAnalysis dataclass fields."""

    def test_has_required_fields(self) -> None:
        names = {f.name for f in fields(VideoAnalysis)}
        expected = {
            "video_path", "total_frames", "analyzed_frames", "frames",
            "manipulation_timeline", "summary",
        }
        assert expected.issubset(names)


class TestImagePredictionDataclass:
    """Verify ImagePrediction dataclass fields."""

    def test_has_required_fields(self) -> None:
        names = {f.name for f in fields(ImagePrediction)}
        expected = {
            "image_path", "is_fake", "confidence", "fake_probability",
            "real_probability", "face_detected", "bounding_box",
        }
        assert expected.issubset(names)


# ---------------------------------------------------------------------------
# VideoClassifier
# ---------------------------------------------------------------------------


class TestVideoClassifier:
    """Tests for VideoClassifier."""

    def test_aggregate_predictions_fake_majority(self, dummy_model: nn.Module) -> None:
        vc = VideoClassifier(model=dummy_model, device="cpu", threshold=0.5)
        # All frames predict fake (high fake_prob)
        frame_results = [
            {"fake_prob": 0.9, "real_prob": 0.1, "label": "fake"},
            {"fake_prob": 0.8, "real_prob": 0.2, "label": "fake"},
            {"fake_prob": 0.85, "real_prob": 0.15, "label": "fake"},
        ]
        is_fake, confidence = vc.aggregate_predictions(frame_results)
        assert is_fake is True
        assert 0.0 <= confidence <= 1.0

    def test_aggregate_predictions_real_majority(self, dummy_model: nn.Module) -> None:
        vc = VideoClassifier(model=dummy_model, device="cpu", threshold=0.5)
        frame_results = [
            {"fake_prob": 0.1, "real_prob": 0.9, "label": "real"},
            {"fake_prob": 0.2, "real_prob": 0.8, "label": "real"},
        ]
        is_fake, confidence = vc.aggregate_predictions(frame_results)
        assert is_fake is False

    def test_classify_frames_returns_list(self, dummy_model: nn.Module, fake_frame_paths: list[Path]) -> None:
        vc = VideoClassifier(model=dummy_model, device="cpu")
        results = vc.classify_frames(fake_frame_paths)
        assert isinstance(results, list)
        assert len(results) == len(fake_frame_paths)
        for r in results:
            assert "fake_prob" in r
            assert "real_prob" in r

    def test_classify_video_returns_prediction(self, dummy_model: nn.Module, fake_frame_paths: list[Path]) -> None:
        vc = VideoClassifier(model=dummy_model, device="cpu")
        prediction = vc.classify_video(fake_frame_paths)
        assert isinstance(prediction, VideoPrediction)
        assert prediction.num_frames == len(fake_frame_paths)

    def test_to_dict_round_trip(self, dummy_model: nn.Module, fake_frame_paths: list[Path]) -> None:
        vc = VideoClassifier(model=dummy_model, device="cpu")
        prediction = vc.classify_video(fake_frame_paths)
        d = vc.to_dict(prediction)
        assert isinstance(d, dict)
        assert d["num_frames"] == len(fake_frame_paths)


# ---------------------------------------------------------------------------
# FrameAnalyzer
# ---------------------------------------------------------------------------


class TestFrameAnalyzer:
    """Tests for FrameAnalyzer."""

    def test_analyze_video_returns_analysis(self, dummy_model: nn.Module, fake_frame_paths: list[Path]) -> None:
        fa = FrameAnalyzer(model=dummy_model, device="cpu")
        indices = list(range(len(fake_frame_paths)))
        timestamps = [i * 0.5 for i in indices]
        analysis = fa.analyze_video(fake_frame_paths, indices, timestamps)
        assert isinstance(analysis, VideoAnalysis)
        assert analysis.total_frames == len(fake_frame_paths)

    def test_rank_frames(self, dummy_model: nn.Module, fake_frame_paths: list[Path]) -> None:
        fa = FrameAnalyzer(model=dummy_model, device="cpu")
        indices = list(range(len(fake_frame_paths)))
        timestamps = [i * 0.5 for i in indices]
        analysis = fa.analyze_video(fake_frame_paths, indices, timestamps)
        top = fa.rank_frames(analysis, top_k=2)
        assert len(top) <= 2

    def test_to_dict(self, dummy_model: nn.Module, fake_frame_paths: list[Path]) -> None:
        fa = FrameAnalyzer(model=dummy_model, device="cpu")
        indices = list(range(len(fake_frame_paths)))
        timestamps = [i * 0.5 for i in indices]
        analysis = fa.analyze_video(fake_frame_paths, indices, timestamps)
        d = fa.to_dict(analysis)
        assert isinstance(d, dict)
        assert "frames" in d


# ---------------------------------------------------------------------------
# ImagePredictor
# ---------------------------------------------------------------------------


class TestImagePredictor:
    """Tests for ImagePredictor."""

    def test_predict_returns_prediction(self, dummy_model: nn.Module, fake_image: Path) -> None:
        ip = ImagePredictor(model=dummy_model, device="cpu")
        pred = ip.predict(fake_image)
        assert isinstance(pred, ImagePrediction)
        assert pred.image_path == str(fake_image)

    def test_predict_values_in_range(self, dummy_model: nn.Module, fake_image: Path) -> None:
        ip = ImagePredictor(model=dummy_model, device="cpu")
        pred = ip.predict(fake_image)
        assert 0.0 <= pred.fake_probability <= 1.0
        assert 0.0 <= pred.real_probability <= 1.0
        assert 0.0 <= pred.confidence <= 1.0

    def test_to_dict(self, dummy_model: nn.Module, fake_image: Path) -> None:
        ip = ImagePredictor(model=dummy_model, device="cpu")
        pred = ip.predict(fake_image)
        d = ip.to_dict(pred)
        assert isinstance(d, dict)
        assert d["image_path"] == str(fake_image)
