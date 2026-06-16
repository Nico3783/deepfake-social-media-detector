"""
Dataset organization.

Purpose: Organize raw datasets into a standardized structure.
Responsibilities: Create directory structure, move/copy files, generate metadata.
Dependencies: pathlib, shutil

Research Traceability:
    Research Objective: Standardized dataset format for reproducibility
    Methodology: Consistent directory structure across datasets
    Implementation: src/data/organize.py
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from src.config.settings import Settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DatasetOrganizer:
    """Organize raw datasets into standardized structure.

    Creates a consistent directory layout:
    data/
        raw/
            faceforensics++/
                real/
                fake/
            celeb-df/
                real/
                fake/
        splits/
            train/
            val/
            test/
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize organizer with project settings.

        Args:
            settings: Project settings instance.
        """
        self.settings = settings
        self.data_dir = settings.paths.data_dir

    def create_standard_structure(self) -> dict[str, Path]:
        """Create the standard directory structure.

        Returns:
            Dictionary of created directories.
        """
        dirs = {
            "raw": self.data_dir / "raw",
            "processed": self.data_dir / "processed",
            "splits": self.data_dir / "splits",
            "train": self.data_dir / "splits" / "train",
            "val": self.data_dir / "splits" / "val",
            "test": self.data_dir / "splits" / "test",
        }

        for name, path in dirs.items():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")

        return dirs

    def organize_faceforensics(
        self,
        source_dir: Path,
        target_dir: Path | None = None,
        compression_level: str = "c23",
    ) -> Path:
        """Organize FaceForensics++ dataset.

        Args:
            source_dir: Source directory with raw FaceForensics++ data.
            target_dir: Target directory (default: data/raw/faceforensics++).
            compression_level: Video compression level (c0, c23, c40).

        Returns:
            Path to organized dataset.
        """
        target_dir = target_dir or self.data_dir / "raw" / "faceforensics++"
        target_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for each manipulation method
        manipulation_methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
        for method in manipulation_methods:
            (target_dir / "real" / method).mkdir(parents=True, exist_ok=True)
            (target_dir / "manipulated" / method).mkdir(parents=True, exist_ok=True)

        logger.info(f"Organizing FaceForensics++ from {source_dir} to {target_dir}")

        # Copy real videos (original)
        real_source = source_dir / "original_sequences" / "youtube" / compression_level
        if real_source.exists():
            self._copy_videos(real_source, target_dir / "real", label="real")
        else:
            logger.warning(f"Real videos not found at {real_source}")

        # Copy manipulated videos
        for method in manipulation_methods:
            method_source = source_dir / "manipulated_sequences" / method / compression_level
            if method_source.exists():
                self._copy_videos(method_source, target_dir / "manipulated" / method, label="fake")
            else:
                logger.warning(f"Method {method} not found at {method_source}")

        logger.info("FaceForensics++ organization complete")
        return target_dir

    def organize_celebdf(
        self,
        source_dir: Path,
        target_dir: Path | None = None,
    ) -> Path:
        """Organize Celeb-DF dataset.

        Args:
            source_dir: Source directory with raw Celeb-DF data.
            target_dir: Target directory (default: data/raw/celeb-df).

        Returns:
            Path to organized dataset.
        """
        target_dir = target_dir or self.data_dir / "raw" / "celeb-df"
        target_dir.mkdir(parents=True, exist_ok=True)

        (target_dir / "real").mkdir(parents=True, exist_ok=True)
        (target_dir / "fake").mkdir(parents=True, exist_ok=True)

        logger.info(f"Organizing Celeb-DF from {source_dir} to {target_dir}")

        # Copy real videos
        real_source = source_dir / "Celeb-real"
        if real_source.exists():
            self._copy_videos(real_source, target_dir / "real", label="real")
        else:
            logger.warning(f"Real videos not found at {real_source}")

        # Copy fake videos
        fake_source = source_dir / "Celeb-synthesis"
        if fake_source.exists():
            self._copy_videos(fake_source, target_dir / "fake", label="fake")
        else:
            logger.warning(f"Fake videos not found at {fake_source}")

        logger.info("Celeb-DF organization complete")
        return target_dir

    def _copy_videos(self, source: Path, target: Path, label: str) -> None:
        """Copy video files from source to target.

        Args:
            source: Source directory with videos.
            target: Target directory.
            label: Label for the videos (real/fake).
        """
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        copied = 0

        for video_file in source.rglob("*"):
            if video_file.suffix.lower() in video_extensions:
                dest = target / video_file.name
                if not dest.exists():
                    shutil.copy2(video_file, dest)
                    copied += 1

        logger.info(f"Copied {copied} videos to {target}")

    def generate_metadata(self, dataset_name: str = "combined") -> Path:
        """Generate metadata CSV for the organized dataset.

        Args:
            dataset_name: Name of the dataset.

        Returns:
            Path to the generated metadata file.
        """
        metadata_path = self.data_dir / "metadata" / f"{dataset_name}.csv"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

        for video_path in self.data_dir.rglob("*"):
            if video_path.suffix.lower() in video_extensions:
                label = 1 if "fake" in str(video_path).lower() or "manipulated" in str(video_path).lower() else 0
                rows.append({
                    "video_path": str(video_path.relative_to(self.data_dir)),
                    "label": label,
                    "label_name": "fake" if label == 1 else "real",
                })

        with open(metadata_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["video_path", "label", "label_name"])
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Generated metadata for {len(rows)} videos at {metadata_path}")
        return metadata_path
