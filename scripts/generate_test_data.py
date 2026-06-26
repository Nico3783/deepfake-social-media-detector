#!/usr/bin/env python3
"""
Generate synthetic test data for pipeline validation.

Purpose: Create small synthetic dataset to validate experiment pipeline before using real data.
Usage: python -m scripts.generate_test_data [--num_samples 100]

Note: This generates synthetic images for testing the pipeline only.
      For actual experiments, download FaceForensics++ and Celeb-DF datasets.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image


def create_synthetic_image(
    output_path: Path,
    size: int = 299,
    label: str = "real",
    seed: int | None = None,
) -> None:
    """Create a synthetic test image.

    Args:
        output_path: Path to save image.
        size: Image size.
        label: 'real' or 'fake' for visual distinction.
        seed: Random seed.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Create base image with face-like structure
    img_array = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)

    # Add face-like oval (lighter region)
    center = size // 2
    y, x = np.ogrid[:size, :size]
    mask = ((x - center) ** 2 + (y - center) ** 2) < (size // 3) ** 2
    img_array[mask] = np.clip(img_array[mask].astype(int) + 50, 0, 255).astype(np.uint8)

    # Add eye-like dark spots for 'real' images
    if label == "real":
        eye_y = center - size // 6
        left_eye_x = center - size // 6
        right_eye_x = center + size // 6
        img_array[eye_y - 5:eye_y + 5, left_eye_x - 5:left_eye_x + 5] = [40, 40, 40]
        img_array[eye_y - 5:eye_y + 5, right_eye_x - 5:right_eye_x + 5] = [40, 40, 40]
    else:
        # 'fake' images have slightly different patterns
        img_array = np.clip(img_array.astype(int) + 20, 0, 255).astype(np.uint8)

    img = Image.fromarray(img_array)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def generate_dataset(
    output_dir: Path,
    num_samples: int = 100,
    split: str = "test",
    seed: int = 42,
) -> None:
    """Generate synthetic dataset with metadata.

    Args:
        output_dir: Output directory.
        num_samples: Number of samples per class.
        split: Dataset split ('train', 'val', 'test').
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)

    images_dir = output_dir / "images"
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    sample_id = 0

    # Generate real samples
    for i in range(num_samples):
        img_name = f"{split}_real_{i:04d}.jpg"
        img_path = images_dir / img_name
        create_synthetic_image(img_path, label="real", seed=seed + i)

        rows.append({
            "video_path": f"images/{img_name}",
            "label": 0,
            "split": split,
            "class": "real",
        })
        sample_id += 1

    # Generate fake samples
    for i in range(num_samples):
        img_name = f"{split}_fake_{i:04d}.jpg"
        img_path = images_dir / img_name
        create_synthetic_image(img_path, label="fake", seed=seed + num_samples + i)

        rows.append({
            "video_path": f"images/{img_name}",
            "label": 1,
            "split": split,
            "class": "fake",
        })
        sample_id += 1

    # Write metadata CSV
    csv_path = metadata_dir / f"{split}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path", "label", "split", "class"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} samples in {output_dir}")
    print(f"Metadata saved to {csv_path}")


def main() -> None:
    """Generate synthetic test data."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic test data for pipeline validation"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples per class per split",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/test_synthetic",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Generate train, val, test splits
    for split, n in [("train", args.num_samples), ("val", args.num_samples // 2), ("test", args.num_samples // 2)]:
        generate_dataset(output_dir, num_samples=n, split=split, seed=args.seed)

    print("\nSynthetic dataset generated successfully!")
    print(f"Use with: python -m scripts.run_experiments --output_dir {output_dir}")


if __name__ == "__main__":
    main()
