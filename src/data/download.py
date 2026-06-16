"""
Dataset download management.

Purpose: Download and manage FaceForensics++ and Celeb-DF datasets.
Responsibilities: Download datasets, verify integrity, manage storage.
Dependencies: requests, pathlib, hashlib

Research Traceability:
    Research Objective: Access to benchmark deepfake datasets
    Methodology: FaceForensics++ and Celeb-DF as evaluation datasets
    Implementation: src/data/download.py
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import requests

from src.config.settings import Settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DatasetDownloader:
    """Download and verify datasets for deepfake detection.

    Supports FaceForensics++ and Celeb-DF datasets.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize downloader with project settings.

        Args:
            settings: Project settings instance.
        """
        self.settings = settings
        self.data_dir = settings.paths.data_dir

    def download_file(self, url: str, dest: Path, expected_md5: str | None = None) -> Path:
        """Download a file with progress tracking.

        Args:
            url: URL to download from.
            dest: Destination file path.
            expected_md5: Optional MD5 hash for verification.

        Returns:
            Path to downloaded file.

        Raises:
            requests.RequestException: If download fails.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists():
            logger.info(f"File already exists: {dest}")
            if expected_md5 and self._verify_md5(dest, expected_md5):
                logger.info(f"MD5 verification passed: {dest.name}")
                return dest
            logger.warning(f"MD5 mismatch, re-downloading: {dest.name}")

        logger.info(f"Downloading {url} -> {dest}")

        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    if downloaded % (1024 * 1024) == 0:  # Log every MB
                        logger.info(f"Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)")

        if expected_md5:
            if not self._verify_md5(dest, expected_md5):
                dest.unlink()
                raise ValueError(f"MD5 verification failed for {dest}")

        logger.info(f"Download complete: {dest}")
        return dest

    def _verify_md5(self, filepath: Path, expected_md5: str) -> bool:
        """Verify MD5 hash of a file.

        Args:
            filepath: Path to the file.
            expected_md5: Expected MD5 hash.

        Returns:
            True if hash matches, False otherwise.
        """
        md5_hash = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest() == expected_md5

    def download_faceforensics(self, output_dir: Path | None = None) -> Path:
        """Download FaceForensics++ dataset.

        Note: FaceForensics++ requires manual download from
        https://github.com/ondyari/FaceForensics

        Args:
            output_dir: Output directory (default: data/faceforensics++).

        Returns:
            Path to dataset directory.
        """
        output_dir = output_dir or self.data_dir / "faceforensics++"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("FaceForensics++ requires manual download.")
        logger.info("Please download from: https://github.com/ondyari/FaceForensics")
        logger.info(f"Place the dataset in: {output_dir}")

        # Check if dataset already exists
        real_dir = output_dir / "real"
        manipulated_dir = output_dir / "manipulated"

        if real_dir.exists() and any(real_dir.iterdir()):
            logger.info(f"FaceForensics++ dataset found at {output_dir}")
        else:
            logger.warning(f"FaceForensics++ dataset not found at {output_dir}")
            logger.warning("Please download and place the dataset manually.")

        return output_dir

    def download_celebdf(self, output_dir: Path | None = None) -> Path:
        """Download Celeb-DF dataset.

        Note: Celeb-DF requires manual download from
        https://github.com/yuezunli/celeb-deepfakeforensics

        Args:
            output_dir: Output directory (default: data/celeb-df).

        Returns:
            Path to dataset directory.
        """
        output_dir = output_dir or self.data_dir / "celeb-df"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Celeb-DF requires manual download.")
        logger.info("Please download from: https://github.com/yuezunli/celeb-deepfakeforensics")
        logger.info(f"Place the dataset in: {output_dir}")

        # Check if dataset already exists
        real_dir = output_dir / "Celeb-real"
        fake_dir = output_dir / "Celeb-synthesis"

        if real_dir.exists() and any(real_dir.iterdir()):
            logger.info(f"Celeb-DF dataset found at {output_dir}")
        else:
            logger.warning(f"Celeb-DF dataset not found at {output_dir}")
            logger.warning("Please download and place the dataset manually.")

        return output_dir

    def download_all(self) -> dict[str, Path]:
        """Download all approved datasets.

        Returns:
            Dictionary mapping dataset names to their paths.
        """
        logger.info("Downloading all approved datasets...")

        paths = {
            "faceforensics": self.download_faceforensics(),
            "celebdf": self.download_celebdf(),
        }

        logger.info("Dataset download process complete.")
        logger.info("Note: Some datasets may require manual download.")
        return paths
