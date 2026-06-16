"""
PyTorch Dataset for deepfake detection.

Purpose: Load and preprocess video frames for training/inference.
Responsibilities: Load images, apply transforms, return batches.
Dependencies: torch, torchvision, PIL, numpy

Research Traceability:
    Research Objective: Efficient data loading for deep learning
    Methodology: Standard PyTorch Dataset with augmentation
    Implementation: src/data/dataset.py
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DeepfakeDataset(Dataset):
    """Dataset for deepfake detection.

    Loads video frames and their corresponding labels.
    Supports data augmentation for training.
    """

    def __init__(
        self,
        metadata_path: Path,
        root_dir: Path,
        transform: Callable | None = None,
        mode: str = "train",
    ) -> None:
        """Initialize dataset.

        Args:
            metadata_path: Path to CSV file with video_path and label columns.
            root_dir: Root directory for video frames.
            transform: Optional transform to apply to images.
            mode: Dataset mode ('train', 'val', 'test').
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mode = mode

        self.samples = []
        self.labels = []

        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row["video_path"])
                self.labels.append(int(row["label"]))

        logger.info(f"Loaded {len(self.samples)} samples for {mode} mode")

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get a sample from the dataset.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image tensor, label).
        """
        sample_path = self.samples[idx]
        label = self.labels[idx]

        image_path = self.root_dir / sample_path
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets.

        Returns:
            Tensor of class weights.
        """
        class_counts = np.bincount(self.labels, minlength=2)
        total = len(self.labels)
        weights = total / (2 * class_counts)
        return torch.FloatTensor(weights)


class VideoFrameDataset(Dataset):
    """Dataset for video frames organized in directories.

    Expects directory structure:
    root/
        real/
            video1/
                frame_001.jpg
                frame_002.jpg
        fake/
            video1/
                frame_001.jpg
                frame_002.jpg
    """

    def __init__(
        self,
        root_dir: Path,
        transform: Callable | None = None,
        mode: str = "train",
        max_frames_per_video: int | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            root_dir: Root directory containing real/fake subdirectories.
            transform: Optional transform to apply to images.
            mode: Dataset mode ('train', 'val', 'test').
            max_frames_per_video: Maximum frames to load per video.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mode = mode
        self.max_frames = max_frames_per_video

        self.samples = []
        self.labels = []

        for label_name, label in [("real", 0), ("fake", 1)]:
            label_dir = self.root_dir / label_name
            if not label_dir.exists():
                continue

            for video_dir in sorted(label_dir.iterdir()):
                if not video_dir.is_dir():
                    continue

                frames = sorted(video_dir.glob("*.jpg"))
                if self.max_frames:
                    frames = frames[:self.max_frames]

                for frame_path in frames:
                    self.samples.append(frame_path)
                    self.labels.append(label)

        logger.info(f"Loaded {len(self.samples)} frames for {mode} mode")

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get a sample from the dataset.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image tensor, label).
        """
        image_path = self.samples[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets.

        Returns:
            Tensor of class weights.
        """
        class_counts = np.bincount(self.labels, minlength=2)
        total = len(self.labels)
        weights = total / (2 * class_counts)
        return torch.FloatTensor(weights)
