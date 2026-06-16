"""
General helper utilities.

Purpose: Provide common utility functions used across the project.
Responsibilities: Directory management, device detection, file operations.
Dependencies: pathlib, torch

Research Traceability:
    Research Objective: Cross-platform compatibility
    Methodology: Utility functions for consistent behavior
    Implementation: src/utils/helpers.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import torch


def ensure_directory(path: str | Path) -> Path:
    """Create directory if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        The path as a Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device(preference: str = "auto") -> torch.device:
    """Determine the best available compute device.

    Args:
        preference: Device preference - 'auto', 'cuda', 'mps', or 'cpu'.

    Returns:
        torch.device instance for the selected device.

    Example:
        device = get_device()
        model = model.to(device)
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(preference)


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Count total and trainable parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary with 'total' and 'trainable' parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def get_file_size_mb(path: str | Path) -> float:
    """Get file size in megabytes.

    Args:
        path: Path to the file.

    Returns:
        File size in MB.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.stat().st_size / (1024 * 1024)


def safe_copy(src: str | Path, dst: str | Path) -> Path:
    """Copy a file safely, creating parent directories as needed.

    Args:
        src: Source file path.
        dst: Destination file path.

    Returns:
        Path to the copied file.
    """
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst
