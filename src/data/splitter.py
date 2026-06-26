"""
Dataset splitting.

Purpose: Split datasets into train/val/test sets.
Responsibilities: Stratified splitting, cross-validation support.
Dependencies: sklearn, numpy, pathlib

Research Traceability:
    Research Objective: Standardized train/val/test splits
    Methodology: 70/15/15 stratified split
    Implementation: src/data/splitter.py
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.config.settings import Settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DatasetSplitter:
    """Split datasets into train/val/test sets.

    Supports:
    - Stratified splitting by label
    - Fixed random seed for reproducibility
    - Cross-validation folds
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize splitter with project settings.

        Args:
            settings: Project settings instance.
        """
        self.settings = settings
        self.data_dir = settings.paths.data_dir

    def load_metadata(self, metadata_path: Path | None = None) -> list[dict]:
        """Load dataset metadata.

        Args:
            metadata_path: Path to metadata CSV.

        Returns:
            List of metadata dictionaries.
        """
        if metadata_path is None:
            metadata_path = self.data_dir / "metadata" / "combined.csv"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        rows = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        logger.info(f"Loaded metadata for {len(rows)} samples")
        return rows

    def split(
        self,
        metadata: list[dict],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> dict[str, list[dict]]:
        """Split dataset into train/val/test sets.

        Args:
            metadata: List of metadata dictionaries.
            train_ratio: Proportion for training.
            val_ratio: Proportion for validation.
            test_ratio: Proportion for testing.
            seed: Random seed for reproducibility.

        Returns:
            Dictionary with train/val/test splits.

        Raises:
            ValueError: If ratios don't sum to 1.0.
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

        rng = random.Random(seed)
        n = len(metadata)
        n_test = round(n * test_ratio)
        n_val = round(n * val_ratio)
        n_train = n - n_test - n_val

        # Group by label for stratified splitting
        labels = sorted(set(row["label"] for row in metadata))
        label_groups: dict[str, list[dict]] = {label: [] for label in labels}
        for row in metadata:
            label_groups[row["label"]].append(row)

        # Per-label split sizes proportionally, then enforce global sizes
        train, val, test = [], [], []
        for label in labels:
            group = label_groups[label]
            rng.shuffle(group)
            lg = len(group)
            lt = round(lg * test_ratio)
            lv = round(lg * val_ratio)
            ltrain = lg - lt - lv
            train.extend(group[:ltrain])
            val.extend(group[ltrain:ltrain + lv])
            test.extend(group[ltrain + lv:])

        # Enforce exact global sizes by rebalancing train↔val and train↔test
        target_val = n_val
        target_test = n_test

        # First fix test size: move items between train and test
        while len(test) < target_test and len(train) > n_train:
            test.append(train.pop())
        while len(test) > target_test and train is not None:
            train.append(test.pop())

        # Then fix val size: move items between train and val
        while len(val) < target_val and len(train) > n_train:
            val.append(train.pop())
        while len(val) > target_val and train is not None:
            train.append(val.pop())

        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(test)

        splits = {"train": train, "val": val, "test": test}

        # Log split statistics
        for split_name, split_data in splits.items():
            real_count = sum(1 for row in split_data if row["label"] == "0")
            fake_count = sum(1 for row in split_data if row["label"] == "1")
            logger.info(f"{split_name}: {len(split_data)} samples (real={real_count}, fake={fake_count})")

        return splits

    def save_splits(
        self,
        splits: dict[str, list[dict]],
        output_dir: Path | None = None,
    ) -> dict[str, Path]:
        """Save splits to CSV files.

        Args:
            splits: Dictionary with train/val/test splits.
            output_dir: Output directory (default: data/splits).

        Returns:
            Dictionary mapping split names to file paths.
        """
        output_dir = output_dir or self.data_dir / "splits"
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}
        for split_name, split_data in splits.items():
            csv_path = output_dir / f"{split_name}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["video_path", "label", "label_name"])
                writer.writeheader()
                writer.writerows(split_data)

            paths[split_name] = csv_path
            logger.info(f"Saved {split_name} split to {csv_path}")

        return paths

    def create_splits(
        self,
        metadata_path: Path | None = None,
        output_dir: Path | None = None,
        seed: int = 42,
    ) -> dict[str, Path]:
        """Create train/val/test splits from metadata.

        Args:
            metadata_path: Path to metadata CSV.
            output_dir: Output directory for splits.
            seed: Random seed.

        Returns:
            Dictionary mapping split names to file paths.
        """
        metadata = self.load_metadata(metadata_path)
        splits = self.split(metadata, seed=seed)
        return self.save_splits(splits, output_dir)

    def cross_validate(
        self,
        metadata: list[dict],
        n_folds: int = 5,
        seed: int = 42,
    ) -> list[dict[str, list[dict]]]:
        """Create cross-validation folds.

        Args:
            metadata: List of metadata dictionaries.
            n_folds: Number of folds.
            seed: Random seed.

        Returns:
            List of fold dictionaries, each with train/val splits.
        """
        random.seed(seed)
        np.random.seed(seed)

        labels = [row["label"] for row in metadata]
        unique_labels = list(set(labels))

        # Group samples by label
        label_groups = {label: [] for label in unique_labels}
        for i, label in enumerate(labels):
            label_groups[label].append(metadata[i])

        folds = []
        for fold_idx in range(n_folds):
            val_indices = []
            for label, samples in label_groups.items():
                fold_size = len(samples) // n_folds
                start = fold_idx * fold_size
                end = start + fold_size if fold_idx < n_folds - 1 else len(samples)
                val_indices.extend(range(start, end))

            val_data = [samples[i] for label, samples in label_groups.items()
                       for i in range(len(samples)) if i in val_indices]
            train_data = [samples[i] for label, samples in label_groups.items()
                         for i in range(len(samples)) if i not in val_indices]

            folds.append({"train": train_data, "val": val_data})

        return folds
