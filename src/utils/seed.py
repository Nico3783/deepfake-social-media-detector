"""
Random seed management for reproducible experiments.

Purpose: Ensure all random processes produce identical results across runs.
Responsibilities: Set seeds for Python, NumPy, and PyTorch.
Dependencies: random, numpy, torch

Research Traceability:
    Research Objective: Experiment reproducibility
    Methodology: Deterministic random seed control
    Implementation: src/utils/seed.py
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for all random number generators.

    Ensures reproducibility across Python, NumPy, and PyTorch.
    Also configures PyTorch for deterministic operations.

    Args:
        seed: Integer seed value (default: 42).

    Example:
        set_seed(42)
        # All subsequent random operations are now deterministic
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_seed_for_dataloader(seed: int = 42) -> torch.Generator:
    """Create a Generator for DataLoader workers.

    Args:
        seed: Integer seed value.

    Returns:
        A torch.Generator with the specified seed.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
