"""Tests for dataset and data pipeline modules."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.dataset import DeepfakeDataset
from src.data.splitter import DatasetSplitter
from src.data.metadata import MetadataManager
from src.config.settings import Settings


def _create_metadata_csv(path: Path, samples: list[dict]) -> Path:
    """Helper to write a metadata CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path", "label", "label_name"])
        writer.writeheader()
        writer.writerows(samples)
    return path


class TestDeepfakeDataset:
    """Tests for PyTorch DeepfakeDataset."""

    def test_dataset_length(self, tmp_path: Path) -> None:
        """Dataset reports correct length."""
        root_dir = tmp_path / "frames"
        root_dir.mkdir()

        # Create dummy image files
        samples = []
        for i in range(5):
            for label, label_name in [(0, "real"), (1, "fake")]:
                fname = f"{label_name}_{i}.jpg"
                img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
                img.save(root_dir / fname)
                samples.append({"video_path": fname, "label": str(label), "label_name": label_name})

        csv_path = _create_metadata_csv(tmp_path / "metadata.csv", samples)

        dataset = DeepfakeDataset(metadata_path=csv_path, root_dir=root_dir)
        assert len(dataset) == 10

    def test_dataset_getitem(self, tmp_path: Path) -> None:
        """Dataset returns (image, label) tuple."""
        root_dir = tmp_path / "frames"
        root_dir.mkdir()

        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(root_dir / "real_0.jpg")
        img.save(root_dir / "fake_0.jpg")

        samples = [
            {"video_path": "real_0.jpg", "label": "0", "label_name": "real"},
            {"video_path": "fake_0.jpg", "label": "1", "label_name": "fake"},
        ]
        csv_path = _create_metadata_csv(tmp_path / "metadata.csv", samples)

        dataset = DeepfakeDataset(metadata_path=csv_path, root_dir=root_dir)
        image, label = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert label in (0, 1)

    def test_dataset_labels(self, tmp_path: Path) -> None:
        """Dataset assigns correct labels (real=0, fake=1)."""
        root_dir = tmp_path / "frames"
        root_dir.mkdir()

        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(root_dir / "real_0.jpg")
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(root_dir / "fake_0.jpg")

        samples = [
            {"video_path": "real_0.jpg", "label": "0", "label_name": "real"},
            {"video_path": "fake_0.jpg", "label": "1", "label_name": "fake"},
        ]
        csv_path = _create_metadata_csv(tmp_path / "metadata.csv", samples)

        dataset = DeepfakeDataset(metadata_path=csv_path, root_dir=root_dir)
        _, label_0 = dataset[0]
        _, label_1 = dataset[1]
        assert {label_0, label_1} == {0, 1}

    def test_dataset_get_class_weights(self, tmp_path: Path) -> None:
        """Dataset computes class weights."""
        root_dir = tmp_path / "frames"
        root_dir.mkdir()

        for i in range(3):
            Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(root_dir / f"r{i}.jpg")
        for i in range(1):
            Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(root_dir / f"f{i}.jpg")

        samples = [
            {"video_path": f"r{i}.jpg", "label": "0", "label_name": "real"} for i in range(3)
        ] + [
            {"video_path": "f0.jpg", "label": "1", "label_name": "fake"},
        ]
        csv_path = _create_metadata_csv(tmp_path / "metadata.csv", samples)

        dataset = DeepfakeDataset(metadata_path=csv_path, root_dir=root_dir)
        weights = dataset.get_class_weights()
        assert isinstance(weights, torch.Tensor)
        assert len(weights) == 2


class TestDatasetSplitter:
    """Tests for dataset train/val/test splitting."""

    def _make_settings(self, tmp_path: Path) -> Settings:
        """Create a minimal Settings instance for testing."""
        settings = Settings()
        settings.paths.project_root = tmp_path
        settings.paths.data_dir = tmp_path / "data"
        settings.paths.data_dir.mkdir(parents=True, exist_ok=True)
        return settings

    def test_split_sizes(self, tmp_path: Path) -> None:
        """Splitter produces correct split sizes."""
        settings = self._make_settings(tmp_path)
        splitter = DatasetSplitter(settings)

        metadata = [{"video_path": f"img_{i}.jpg", "label": str(i % 2), "label_name": "real" if i % 2 == 0 else "fake"} for i in range(20)]
        splits = splitter.split(metadata, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

        total = sum(len(v) for v in splits.values())
        assert total == 20
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

    def test_split_no_overlap(self, tmp_path: Path) -> None:
        """Split splits have no overlapping files."""
        settings = self._make_settings(tmp_path)
        splitter = DatasetSplitter(settings)

        metadata = [{"video_path": f"img_{i}.jpg", "label": str(i % 2), "label_name": "real"} for i in range(20)]
        splits = splitter.split(metadata)

        all_paths = []
        for split_data in splits.values():
            all_paths.extend(row["video_path"] for row in split_data)
        assert len(all_paths) == len(set(all_paths))

    def test_split_ratios(self, tmp_path: Path) -> None:
        """Splitter respects approximate ratios."""
        settings = self._make_settings(tmp_path)
        splitter = DatasetSplitter(settings)

        metadata = [{"video_path": f"img_{i}.jpg", "label": str(i % 2), "label_name": "real"} for i in range(100)]
        splits = splitter.split(metadata, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

        assert len(splits["train"]) == 70
        assert len(splits["val"]) == 15
        assert len(splits["test"]) == 15

    def test_split_ratios_must_sum_to_one(self, tmp_path: Path) -> None:
        """Splitter raises ValueError when ratios don't sum to 1."""
        settings = self._make_settings(tmp_path)
        splitter = DatasetSplitter(settings)

        metadata = [{"video_path": "a.jpg", "label": "0", "label_name": "real"}]
        with pytest.raises(ValueError, match="ratios must sum"):
            splitter.split(metadata, train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

    def test_save_splits(self, tmp_path: Path) -> None:
        """Splitter saves splits to CSV files."""
        settings = self._make_settings(tmp_path)
        splitter = DatasetSplitter(settings)

        metadata = [{"video_path": f"img_{i}.jpg", "label": str(i % 2), "label_name": "real"} for i in range(10)]
        splits = splitter.split(metadata, seed=42)
        output_dir = tmp_path / "splits"
        paths = splitter.save_splits(splits, output_dir)

        for split_name in ("train", "val", "test"):
            assert paths[split_name].exists()


class TestMetadataManager:
    """Tests for metadata management."""

    def test_create_and_read_csv(self, tmp_path: Path) -> None:
        """MetadataManager creates and reads CSV metadata."""
        manager = MetadataManager(data_dir=tmp_path)

        samples = [
            {"video_path": "real_0.jpg", "label": "0", "label_name": "real"},
            {"video_path": "fake_0.jpg", "label": "1", "label_name": "fake"},
        ]
        csv_path = manager.create_metadata_csv(samples)
        assert csv_path.exists()

        loaded = manager.read_metadata_csv(csv_path)
        assert len(loaded) == 2
        assert loaded[0]["video_path"] == "real_0.jpg"
        assert loaded[1]["label"] == "1"

    def test_create_metadata_empty_raises(self, tmp_path: Path) -> None:
        """MetadataManager raises ValueError for empty samples."""
        manager = MetadataManager(data_dir=tmp_path)
        with pytest.raises(ValueError, match="empty"):
            manager.create_metadata_csv([])

    def test_read_nonexistent_raises(self, tmp_path: Path) -> None:
        """MetadataManager raises FileNotFoundError for missing file."""
        manager = MetadataManager(data_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            manager.read_metadata_csv(tmp_path / "nonexistent.csv")

    def test_create_and_read_json(self, tmp_path: Path) -> None:
        """MetadataManager creates and reads JSON metadata."""
        manager = MetadataManager(data_dir=tmp_path)

        samples = [
            {"video_path": "real_0.jpg", "label": "0", "label_name": "real"},
        ]
        json_path = manager.create_metadata_json(samples)
        assert json_path.exists()

    def test_get_statistics(self, tmp_path: Path) -> None:
        """MetadataManager returns statistics from metadata."""
        manager = MetadataManager(data_dir=tmp_path)

        samples = [
            {"video_path": "r0.jpg", "label": "0", "label_name": "real"},
            {"video_path": "r1.jpg", "label": "0", "label_name": "real"},
            {"video_path": "f0.jpg", "label": "1", "label_name": "fake"},
        ]
        csv_path = manager.create_metadata_csv(samples)
        stats = manager.get_statistics(csv_path)

        assert stats["total_samples"] == 3
        assert stats["class_distribution"]["0"] == 2
        assert stats["class_distribution"]["1"] == 1
