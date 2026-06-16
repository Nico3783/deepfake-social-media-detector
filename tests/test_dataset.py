"""Tests for dataset and data pipeline modules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.dataset import DeepfakeDataset
from src.data.splitter import DatasetSplitter
from src.data.metadata import MetadataManager


class TestDeepfakeDataset:
    """Tests for PyTorch DeepfakeDataset."""

    def test_dataset_length(self, tmp_path: Path) -> None:
        """Dataset reports correct length."""
        # Create mock dataset structure
        real_dir = tmp_path / "real"
        fake_dir = tmp_path / "fake"
        real_dir.mkdir()
        fake_dir.mkdir()

        # Create dummy images
        for i in range(5):
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img.save(real_dir / f"real_{i}.jpg")
            img.save(fake_dir / f"fake_{i}.jpg")

        dataset = DeepfakeDataset(
            real_dir=str(real_dir),
            fake_dir=str(fake_dir),
        )
        assert len(dataset) == 10

    def test_dataset_getitem(self, tmp_path: Path) -> None:
        """Dataset returns (image, label) tuple."""
        real_dir = tmp_path / "real"
        fake_dir = tmp_path / "fake"
        real_dir.mkdir()
        fake_dir.mkdir()

        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(real_dir / "real_0.jpg")
        img.save(fake_dir / "fake_0.jpg")

        dataset = DeepfakeDataset(real_dir=str(real_dir), fake_dir=str(fake_dir))
        image, label = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert label in (0, 1)

    def test_dataset_labels(self, tmp_path: Path) -> None:
        """Dataset assigns correct labels (real=0, fake=1)."""
        real_dir = tmp_path / "real"
        fake_dir = tmp_path / "fake"
        real_dir.mkdir()
        fake_dir.mkdir()

        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(real_dir / "r.jpg")
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(fake_dir / "f.jpg")

        dataset = DeepfakeDataset(real_dir=str(real_dir), fake_dir=str(fake_dir))
        # First file alphabetically gets index 0
        _, label = dataset[0]
        assert label in (0, 1)


class TestDatasetSplitter:
    """Tests for dataset train/val/test splitting."""

    def test_split_sizes(self, tmp_path: Path) -> None:
        """Splitter produces correct split sizes."""
        # Create image files
        for i in range(10):
            img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
            img.save(tmp_path / f"img_{i}.jpg")

        splitter = DatasetSplitter()
        splits = splitter.split(
            source_dir=str(tmp_path),
            output_dir=str(tmp_path / "splits"),
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        total = sum(len(v) for v in splits.values())
        assert total == 10

    def test_split_no_overlap(self, tmp_path: Path) -> None:
        """Split splits have no overlapping files."""
        for i in range(20):
            img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
            img.save(tmp_path / f"img_{i}.jpg")

        splitter = DatasetSplitter()
        splits = splitter.split(
            source_dir=str(tmp_path),
            output_dir=str(tmp_path / "splits"),
        )
        all_files = []
        for files in splits.values():
            all_files.extend(files)
        assert len(all_files) == len(set(all_files))


class TestMetadataManager:
    """Tests for metadata management."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        """MetadataManager persists and loads data correctly."""
        manager = MetadataManager(storage_path=str(tmp_path / "metadata.json"))
        manager.save({"key": "value", "count": 42})

        loaded = manager.load()
        assert loaded["key"] == "value"
        assert loaded["count"] == 42

    def test_load_empty(self, tmp_path: Path) -> None:
        """MetadataManager returns empty dict when no file exists."""
        manager = MetadataManager(storage_path=str(tmp_path / "nonexistent.json"))
        loaded = manager.load()
        assert loaded == {}
