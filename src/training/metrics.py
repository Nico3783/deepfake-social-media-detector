"""
Training metrics.

Purpose: Calculate and track metrics during training.
Responsibilities: Accuracy, precision, recall, F1, confusion matrix, ROC-AUC.
Dependencies: torch, numpy

Research Traceability:
    Research Objective: Comprehensive performance evaluation
    Methodology: Standard classification metrics
    Implementation: src/training/metrics.py
"""

from __future__ import annotations

import numpy as np
import torch

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MetricsTracker:
    """Track and compute training metrics.

    Supports:
    - Accuracy, Precision, Recall, F1-Score
    - Confusion Matrix
    - ROC-AUC

    Uses pure torch/numpy implementations — no torchmetrics dependency.
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

        # Storage for epoch metrics
        self.reset()

    def reset(self) -> None:
        """Reset all metrics."""
        self.all_preds: list[torch.Tensor] = []
        self.all_targets: list[torch.Tensor] = []
        self.all_probs: list[torch.Tensor] = []

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
        self.all_preds.append(preds.cpu())
        self.all_targets.append(targets.cpu())

        if probs is not None:
            self.all_probs.append(probs.cpu())

    def compute(self) -> dict[str, float]:
        """Compute all metrics.

        Returns:
            Dictionary of computed metrics.
        """
        all_preds = torch.cat(self.all_preds) if self.all_preds else torch.tensor([], dtype=torch.long)
        all_targets = torch.cat(self.all_targets) if self.all_targets else torch.tensor([], dtype=torch.long)

        if len(all_preds) == 0:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "auroc": 0.0,
                "confusion_matrix": [[0, 0], [0, 0]],
            }

        # Accuracy
        correct = (all_preds == all_targets).sum().item()
        total = all_targets.numel()
        accuracy = correct / total if total > 0 else 0.0

        # Precision, Recall, F1 per class, then average
        precision, recall, f1 = self._compute_precision_recall_f1(all_preds, all_targets)

        # ROC-AUC
        auroc = 0.0
        if self.all_probs:
            all_probs = torch.cat(self.all_probs)
            try:
                auroc = self._compute_auroc(all_probs, all_targets)
            except Exception as e:
                logger.warning(f"Could not compute AUROC: {e}")
                auroc = 0.0

        # Confusion matrix
        cm = self._compute_confusion_matrix(all_preds, all_targets)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auroc": auroc,
            "confusion_matrix": cm,
        }

    def _compute_precision_recall_f1(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> tuple[float, float, float]:
        """Compute precision, recall, and F1-score.

        Args:
            preds: Predicted labels.
            targets: Ground truth labels.

        Returns:
            Tuple of (precision, recall, f1).
        """
        if self.task == "binary":
            tp = ((preds == 1) & (targets == 1)).sum().item()
            fp = ((preds == 1) & (targets == 0)).sum().item()
            fn = ((preds == 0) & (targets == 1)).sum().item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            return precision, recall, f1
        else:
            # Multiclass: macro average
            precisions = []
            recalls = []
            f1s = []
            for c in range(self.num_classes):
                tp = ((preds == c) & (targets == c)).sum().item()
                fp = ((preds == c) & (targets != c)).sum().item()
                fn = ((preds != c) & (targets == c)).sum().item()

                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

                precisions.append(p)
                recalls.append(r)
                f1s.append(f)

            return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))

    def _compute_auroc(self, probs: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute AUROC using numpy trapezoidal integration.

        Args:
            probs: Prediction probabilities.
            targets: Ground truth labels.

        Returns:
            AUROC score.
        """
        if self.task == "binary":
            # probs should be probabilities for the positive class
            if probs.dim() == 2 and probs.shape[1] == 2:
                scores = probs[:, 1].numpy()
            else:
                scores = probs.numpy()

            labels = targets.numpy()
        else:
            # Multiclass: one-vs-rest
            scores = probs.numpy()
            labels = targets.numpy()

        # Compute ROC curve points
        if self.task == "binary":
            thresholds = np.sort(np.unique(scores))[::-1]
            tpr_list = [0.0]
            fpr_list = [0.0]

            pos_total = (labels == 1).sum()
            neg_total = (labels == 0).sum()

            if pos_total == 0 or neg_total == 0:
                return 0.5

            for thresh in thresholds:
                preds = (scores >= thresh).astype(int)
                tp = ((preds == 1) & (labels == 1)).sum()
                fp = ((preds == 1) & (labels == 0)).sum()
                tpr = tp / pos_total
                fpr = fp / neg_total
                tpr_list.append(tpr)
                fpr_list.append(fpr)

            tpr_list.append(1.0)
            fpr_list.append(1.0)

            # Trapezoidal AUC
            auroc = float(np.trapz(tpr_list, fpr_list))
            # If the curve is inverted (below diagonal), flip it
            if auroc < 0.5:
                auroc = 1.0 - auroc
            return auroc
        else:
            # Multiclass macro AUROC
            aurocs = []
            for c in range(self.num_classes):
                binary_labels = (labels == c).astype(int)
                if binary_labels.sum() == 0 or (1 - binary_labels).sum() == 0:
                    continue
                class_probs = scores[:, c] if scores.ndim == 2 else scores

                thresholds = np.sort(np.unique(class_probs))[::-1]
                tpr_list = [0.0]
                fpr_list = [0.0]

                pos_total = binary_labels.sum()
                neg_total = (1 - binary_labels).sum()

                for thresh in thresholds:
                    preds = (class_probs >= thresh).astype(int)
                    tp = ((preds == 1) & (binary_labels == 1)).sum()
                    fp = ((preds == 1) & (binary_labels == 0)).sum()
                    tpr = tp / pos_total
                    fpr = fp / neg_total
                    tpr_list.append(tpr)
                    fpr_list.append(fpr)

                tpr_list.append(1.0)
                fpr_list.append(1.0)
                auc = float(np.trapz(tpr_list, fpr_list))
                if auc < 0.5:
                    auc = 1.0 - auc
                aurocs.append(auc)

            return float(np.mean(aurocs)) if aurocs else 0.5

    def _compute_confusion_matrix(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> list[list[int]]:
        """Compute confusion matrix.

        Args:
            preds: Predicted labels.
            targets: Ground truth labels.

        Returns:
            Confusion matrix as nested list.
        """
        cm = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.long
        )
        for t, p in zip(targets, preds):
            cm[t.long()][p.long()] += 1
        return cm.cpu().numpy().tolist()

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
