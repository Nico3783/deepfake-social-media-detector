"""
Metadata management.

Purpose: Manage dataset metadata and annotations.
Responsibilities: Create, read, update, and validate metadata.
Dependencies: csv, json, pathlib

Research Traceability:
    Research Objective: Structured metadata for experiment tracking
    Methodology: CSV-based metadata with versioning
    Implementation: src/data/metadata.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MetadataManager:
    """Manage dataset metadata.

    Supports:
    - CSV metadata files
    - JSON metadata files
    - Metadata validation
    - Metadata statistics
    """

    def __init__(self, data_dir: Path) -> None:
        """Initialize metadata manager.

        Args:
            data_dir: Path to the data directory.
        """
        self.data_dir = Path(data_dir)
        self.metadata_dir = self.data_dir / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def create_metadata_csv(
        self,
        samples: list[dict[str, Any]],
        output_path: Path | None = None,
    ) -> Path:
        """Create a metadata CSV file.

        Args:
            samples: List of sample dictionaries.
            output_path: Output path (default: metadata/combined.csv).

        Returns:
            Path to the created CSV file.
        """
        output_path = output_path or self.metadata_dir / "combined.csv"

        if not samples:
            raise ValueError("Cannot create metadata from empty samples")

        fieldnames = list(samples[0].keys())

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(samples)

        logger.info(f"Created metadata CSV with {len(samples)} samples at {output_path}")
        return output_path

    def read_metadata_csv(self, metadata_path: Path) -> list[dict[str, str]]:
        """Read a metadata CSV file.

        Args:
            metadata_path: Path to the CSV file.

        Returns:
            List of metadata dictionaries.
        """
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def create_metadata_json(
        self,
        samples: list[dict[str, Any]],
        output_path: Path | None = None,
    ) -> Path:
        """Create a metadata JSON file.

        Args:
            samples: List of sample dictionaries.
            output_path: Output path (default: metadata/combined.json).

        Returns:
            Path to the created JSON file.
        """
        output_path = output_path or self.metadata_dir / "combined.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        logger.info(f"Created metadata JSON with {len(samples)} samples at {output_path}")
        return output_path

    def validate_metadata(self, metadata_path: Path) -> dict[str, Any]:
        """Validate metadata file.

        Args:
            metadata_path: Path to the metadata file.

        Returns:
            Dictionary with validation results.
        """
        results = {
            "valid": True,
            "total_samples": 0,
            "class_distribution": {},
            "missing_values": [],
            "errors": [],
        }

        try:
            if metadata_path.suffix == ".csv":
                samples = self.read_metadata_csv(metadata_path)
            elif metadata_path.suffix == ".json":
                with open(metadata_path, "r", encoding="utf-8") as f:
                    samples = json.load(f)
            else:
                results["valid"] = False
                results["errors"].append(f"Unsupported format: {metadata_path.suffix}")
                return results

            results["total_samples"] = len(samples)

            # Check class distribution
            for sample in samples:
                label = sample.get("label", "unknown")
                results["class_distribution"][label] = results["class_distribution"].get(label, 0) + 1

            # Check for missing values
            for i, sample in enumerate(samples):
                for key, value in sample.items():
                    if not value and value != 0:
                        results["missing_values"].append(f"Sample {i}: {key}")

            if results["missing_values"]:
                results["valid"] = False
                results["errors"].append(f"Found {len(results['missing_values'])} missing values")

        except Exception as e:
            results["valid"] = False
            results["errors"].append(str(e))

        return results

    def get_statistics(self, metadata_path: Path) -> dict[str, Any]:
        """Get statistics from metadata file.

        Args:
            metadata_path: Path to the metadata file.

        Returns:
            Dictionary with metadata statistics.
        """
        samples = self.read_metadata_csv(metadata_path)

        stats = {
            "total_samples": len(samples),
            "class_distribution": {},
            "label_names": {},
        }

        for sample in samples:
            label = sample.get("label", "unknown")
            label_name = sample.get("label_name", "unknown")

            stats["class_distribution"][label] = stats["class_distribution"].get(label, 0) + 1
            stats["label_names"][label] = label_name

        return stats
