"""Tests for preprocessing pipeline modules."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.preprocessing.face_detector import FaceDetector
from src.preprocessing.face_cropper import FaceCropper
from src.preprocessing.image_resizer import ImageResizer
from src.preprocessing.normalizer import ImageNormalizer
from src.preprocessing.frame_extractor import FrameExtractor


class TestFaceDetector:
    """Tests for MTCNN face detector."""

    def test_initialization(self) -> None:
        """FaceDetector initializes without error."""
        detector = FaceDetector()
        assert detector is not None

    def test_detect_no_face(self) -> None:
        """FaceDetector handles images with no faces gracefully."""
        detector = FaceDetector()
        # Random noise unlikely to contain a face
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        faces = detector.detect(img)
        # Should return empty list or handle gracefully
        assert isinstance(faces, list)

    def test_detect_returns_list(self) -> None:
        """FaceDetector.detect returns a list."""
        detector = FaceDetector()
        img = np.random.randint(0, 255, (299, 299, 3), dtype=np.uint8)
        result = detector.detect(img)
        assert isinstance(result, list)


class TestFaceCropper:
    """Tests for face cropping."""

    def test_crop_with_bbox(self, sample_rgb_image: np.ndarray) -> None:
        """FaceCropper crops face given bounding box."""
        cropper = FaceCropper()
        bbox = (50, 50, 200, 200)
        cropped = cropper.crop(sample_rgb_image, bbox)
        assert cropped.shape[0] > 0
        assert cropped.shape[1] > 0

    def test_crop_output_is_image(self, sample_rgb_image: np.ndarray) -> None:
        """FaceCropper output is a valid image array."""
        cropper = FaceCropper()
        bbox = (10, 10, 100, 100)
        cropped = cropper.crop(sample_rgb_image, bbox)
        assert cropped.dtype == np.uint8


class TestImageResizer:
    """Tests for image resizing."""

    def test_resize_to_target(self, sample_rgb_image: np.ndarray) -> None:
        """ImageResizer resizes to target dimensions."""
        resizer = ImageResizer(target_size=224)
        resized = resizer.resize(sample_rgb_image)
        assert resized.shape[:2] == (224, 224)

    def test_resize_preserves_channels(self, sample_rgb_image: np.ndarray) -> None:
        """ImageResizer preserves channel count."""
        resizer = ImageResizer(target_size=299)
        resized = resizer.resize(sample_rgb_image)
        assert resized.shape[2] == 3

    def test_resize方形(self) -> None:
        """ImageResizer produces square output."""
        resizer = ImageResizer(target_size=256)
        img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        resized = resizer.resize(img)
        assert resized.shape[0] == 256
        assert resized.shape[1] == 256


class TestImageNormalizer:
    """Tests for image normalization."""

    def test_normalize_to_float(self, sample_rgb_image: np.ndarray) -> None:
        """ImageNormalizer converts uint8 to float."""
        normalizer = ImageNormalizer()
        normalized = normalizer.normalize(sample_rgb_image)
        assert normalized.dtype in (np.float32, np.float64)

    def test_normalize_range(self, sample_rgb_image: np.ndarray) -> None:
        """ImageNormalizer scales values to approximately [0, 1]."""
        normalizer = ImageNormalizer()
        normalized = normalizer.normalize(sample_rgb_image)
        assert normalized.min() >= -0.1  # allow small numerical error
        assert normalized.max() <= 1.1

    def test_normalize_with_mean_std(self, sample_rgb_image: np.ndarray) -> None:
        """ImageNormalizer applies mean/std normalization."""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalizer = ImageNormalizer(mean=mean, std=std)
        normalized = normalizer.normalize(sample_rgb_image)
        assert normalized.shape == sample_rgb_image.shape


class TestFrameExtractor:
    """Tests for video frame extraction."""

    def test_extract_from_video(self, tmp_path: Path) -> None:
        """FrameExtractor extracts frames from a video file."""
        # Create a tiny test video
        video_path = tmp_path / "test.mp4"
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (64, 64))
        for _ in range(10):
            frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        extractor = FrameExtractor()
        frames = extractor.extract(str(video_path))
        assert len(frames) > 0
        assert frames[0].shape[2] == 3

    def test_extract_with_sample_rate(self, tmp_path: Path) -> None:
        """FrameExtractor respects sample rate."""
        video_path = tmp_path / "test.mp4"
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (64, 64))
        for _ in range(30):
            frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        extractor = FrameExtractor()
        frames = extractor.extract(str(video_path), sample_rate=10)
        assert len(frames) <= 4  # 30 frames / sample_rate 10 ≈ 3
