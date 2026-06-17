"""
CLI entry point for dataset splitting.

Purpose: Provide command-line interface for splitting datasets into train/val/test.
Responsibilities: Parse arguments, invoke DatasetSplitter, report statistics.
Dependencies: argparse, src.data.splitter

Research Traceability:
    Research Objective: Standardized dataset splitting
    Methodology: 70/15/15 stratified split with reproducible seeds
    Implementation: src/data/split_data.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data.splitter import DatasetSplitter
from src.config.settings import Settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Split dataset into train/val/test sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Split FaceForensics++ dataset
    python -m src.data.split_data --input datasets/processed/faceforensics

    # Split with custom ratios
    python -m src.data.split_data --input datasets/processed/celebdf \\
        --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1

    # Split with custom seed
    python -m src.data.split_data --input datasets/processed/faceforensics \\
        --seed 123 --output datasets/splits/faceforensics
        """,
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to processed dataset directory",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for splits (default: <input>/splits)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Path to metadata CSV (default: <input>/metadata/combined.csv)",
    )

    return parser.parse_args()


def main() -> None:
    """Run dataset splitting pipeline."""
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input directory does not exist: %s", input_path)
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path / "splits"

    # Validate ratios
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-6:
        logger.error(
            "Ratios must sum to 1.0, got %.4f (train=%.2f, val=%.2f, test=%.2f)",
            total, args.train_ratio, args.val_ratio, args.test_ratio,
        )
        sys.exit(1)

    # Initialize settings and splitter
    settings = Settings()
    splitter = DatasetSplitter(settings)

    # Resolve metadata path
    metadata_path = Path(args.metadata) if args.metadata else input_path / "metadata" / "combined.csv"

    if not metadata_path.exists():
        logger.error("Metadata file not found: %s", metadata_path)
        logger.info("Run 'python -m src.data.organize' first to prepare the dataset")
        sys.exit(1)

    # Load metadata
    logger.info("Loading metadata from %s", metadata_path)
    metadata = splitter.load_metadata(metadata_path)

    # Perform split
    logger.info(
        "Splitting %d samples: train=%.2f, val=%.2f, test=%.2f (seed=%d)",
        len(metadata), args.train_ratio, args.val_ratio, args.test_ratio, args.seed,
    )

    splits = splitter.split(
        metadata,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # Save splits
    paths = splitter.save_splits(splits, output_path)

    # Report results
    logger.info("=" * 60)
    logger.info("Split Summary")
    logger.info("=" * 60)
    for split_name, split_path in paths.items():
        count = len(splits[split_name])
        real = sum(1 for r in splits[split_name] if r.get("label") == "0")
        fake = sum(1 for r in splits[split_name] if r.get("label") == "1")
        logger.info(
            "  %s: %d samples (real=%d, fake=%d) → %s",
            split_name.upper(), count, real, fake, split_path,
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
