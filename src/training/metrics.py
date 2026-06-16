"""
Training metrics.

Purpose: Calculate and track metrics during training.
Responsibilities: Accuracy, precision, recall, F1, confusion matrix.
Dependencies: torch, numpy

Research Traceability:
    Research Objective: Comprehensive performance evaluation
    Methodology: Standard classification metrics
    Implementation: src/training/metrics.py
"""

from __future__ import annotations

import numpy as np
import torch
from torchmetrics import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    ConfusionMatrix,
    AUROC,
)

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MetricsTracker:
    """Track and compute training metrics.

    Supports:
    - Accuracy, Precision, Recall, F1-Score
    - Confusion Matrix
    - ROC-AUC
    """

    def __init__(
        self,
        num_classes: int = 2,
        task: str = "binary",
        device: torch.device | None = None,
    ) -> None:
        """Initialize metrics tracker.

        Args:
            num_classes: Number of classes.
            task: Task type ('binary', 'multiclass').
            device: Device for computation.
        """
        self.num_classes = num_classes
        self.task = task

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device

        # Initialize metrics
        self.accuracy = Accuracy(task=task, num_classes=num_classes).to(device)
        self.precision = Precision(task=task, num_classes=num_classes).to(device)
        self.recall = Recall(task=task, num_classes=num_classes).to(device)
        self.f1 = F1Score(task=task, num_classes=num_classes).to(device)
        self.confusion_matrix = ConfusionMatrix(task=task, num_classes=num_classes).to(device)
        self.auroc = AUROC(task=task, num_classes=num_classes).to(device)

        # Storage for epoch metrics
        self.reset()

    def reset(self) -> None:
        """Reset all metrics."""
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.confusion_matrix.reset()
        self.auroc.reset()

        self.all_preds = []
        self.all_targets = []
        self.all_probs = []

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        probs: torch.Tensor | None = None,
    ) -> None:
        """Update metrics with batch predictions.

        Args:
            preds: Predicted labels.
            targets: Ground truth labels.
            probs: Prediction probabilities (for ROC-AUC).
        """
        self.accuracy.update(preds, targets)
        self.precision.update(preds, targets)
        self.recall.update(preds, targets)
        self.f1.update(preds, targets)
        self.confusion_matrix.update(preds, targets)

        if probs is not None:
            self.auroc.update(probs, targets)
            self.all_probs.append(probs.cpu())

        self.all_preds.append(preds.cpu())
        self.all_targets.append(targets.cpu())

    def compute(self) -> dict[str, float]:
        """Compute all metrics.

        Returns:
            Dictionary of computed metrics.
        """
        metrics = {
            "accuracy": self.accuracy.compute().item(),
            "precision": self.precision.compute().item(),
            "recall": self.recall.compute().item(),
            "f1": self.f1.compute().item(),
        }

        # Compute ROC-AUC if probabilities available
        if self.all_probs:
            try:
                metrics["auroc"] = self.auroc.compute().item()
            except Exception as e:
                logger.warning(f"Could not compute AUROC: {e}")
                metrics["auroc"] = 0.0

        # Compute confusion matrix
        cm = self.confusion_matrix.compute()
        metrics["confusion_matrix"] = cm.cpu().numpy().tolist()

        return metrics

    def get_epoch_metrics(self) -> dict[str, float]:
        """Get metrics for the current epoch.

        Returns:
            Dictionary of epoch metrics.
        """
        return self.compute()

    def log_metrics(self, epoch: int, prefix: str = "") -> None:
        """Log metrics to logger.

        Args:
            epoch: Current epoch number.
            prefix: Prefix for metric names.
        """
        metrics = self.compute()
        prefix_str = f"{prefix}_" if prefix else ""

        logger.info(
            f"Epoch {epoch} - "
            f"{prefix_str}acc={metrics['accuracy']:.4f}, "
            f"{prefix_str}prec={metrics['precision']:.4f}, "
            f"{prefix_str}rec={metrics['recall']:.4f}, "
            f"{prefix_str}f1={metrics['f1']:.4f}"
        )

        if "auroc" in metrics:
            logger.info(f"Epoch {epoch} - {prefix_str}auroc={metrics['auroc']:.4f}")
