"""Shared test fixtures for the deepfake detection test suite."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image


@pytest.fixture
def device() -> torch.device:
    """Return CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def dummy_model() -> nn.Module:
    """Return a minimal CNN model for testing."""
    from src.models.xception import XceptionNet
    return XceptionNet(num_classes=2, pretrained=False)


@pytest.fixture
def dummy_tensor() -> torch.Tensor:
    """Return a dummy input tensor (1, 3, 299, 299) for XceptionNet."""
    return torch.randn(1, 3, 299, 299)


@pytest.fixture
def dummy_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a batch of dummy data (images, labels)."""
    images = torch.randn(4, 3, 299, 299)
    labels = torch.tensor([0, 1, 0, 1])
    return images, labels


@pytest.fixture
def sample_rgb_image() -> np.ndarray:
    """Return a random 299x299 RGB image as numpy array."""
    return np.random.randint(0, 255, (299, 299, 3), dtype=np.uint8)


@pytest.fixture
def temp_image_file(tmp_path: Path) -> Path:
    """Create a temporary image file and return its path."""
    img = Image.fromarray(np.random.randint(0, 255, (299, 299, 3), dtype=np.uint8))
    path = tmp_path / "test_image.jpg"
    img.save(path)
    return path


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_config() -> dict:
    """Return a sample training configuration dictionary."""
    return {
        "model": {"name": "xception", "num_classes": 2, "pretrained": False},
        "training": {
            "batch_size": 4,
            "learning_rate": 0.001,
            "epochs": 2,
            "optimizer": "adam",
            "weight_decay": 0.0001,
        },
        "data": {
            "image_size": 299,
            "num_workers": 0,
        },
    }
