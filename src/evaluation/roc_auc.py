"""
ROC-AUC analysis.

Purpose: Compute ROC curves and AUC scores for deepfake detection models.
Responsibilities: ROC curve computation, AUC scoring, threshold analysis.
Dependencies: numpy, torch, torchmetrics, sklearn

Research Traceability:
    Research Objective: Threshold-independent model evaluation
    Methodology: ROC curve analysis and AUC scoring
    Implementation: src/evaluation/roc_auc.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchmetrics import AUROC

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ROCCurveAnalyzer:
    """Compute and analyze ROC curves for binary classification.

    Provides:
    - ROC-AUC score computation
    - ROC curve data (FPR, TPR at various thresholds)
    - Optimal threshold selection (Youden's J)
    - Per-threshold analysis
    """

    def __init__(self, num_classes: int = 2) -> None:
        """Initialize ROC curve analyzer.

        Args:
            num_classes: Number of classes.
        """
        self.num_classes = num_classes
        self.auroc = AUROC(
            task="binary" if num_classes == 2 else "multiclass",
            num_classes=num_classes,
        )

        logger.info("ROCCurveAnalyzer initialized")

    def compute(
        self,
        probs: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
        num_thresholds: int = 100,
    ) -> dict[str, Any]:
        """Compute ROC curve and AUC score.

        Args:
            probs: Prediction probabilities for the positive class.
                    For binary: shape (N,) or (N, 2) where [:, 1] is positive class.
                    For multiclass: shape (N, num_classes).
            targets: Ground truth labels. Shape (N,).
            num_thresholds: Number of threshold points for the ROC curve.

        Returns:
            Dictionary with ROC-AUC results.
        """
        # Convert to tensors
        if isinstance(probs, np.ndarray):
            probs = torch.from_numpy(probs).float()
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).long()

        # Handle 2D probability input for binary case
        if self.num_classes == 2 and probs.dim() == 2:
            probs = probs[:, 1]

        # Ensure 1D for binary
        if probs.dim() > 1:
            probs = probs.squeeze()

        # Compute AUC
        try:
            auc_score = self.auroc(probs, targets).item()
        except Exception as e:
            logger.warning(f"Could not compute AUC: {e}")
            auc_score = 0.0

        # Compute ROC curve manually for full curve data
        fpr, tpr, thresholds = self._compute_roc_curve(probs, targets, num_thresholds)

        # Find optimal threshold using Youden's J statistic
        optimal_idx = int(np.argmax(tpr - fpr))
        optimal_threshold = float(thresholds[optimal_idx]) if len(thresholds) > 0 else 0.5

        # Compute metrics at optimal threshold
        preds_at_optimal = (probs >= optimal_threshold).long()
        metrics_at_optimal = self._compute_metrics_at_threshold(
            preds_at_optimal, targets
        )

        result: dict[str, Any] = {
            "auc": auc_score,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "optimal_threshold": optimal_threshold,
            "optimal_j_statistic": float(tpr[optimal_idx] - fpr[optimal_idx]) if len(tpr) > 0 else 0.0,
            "metrics_at_optimal": metrics_at_optimal,
            "num_samples": len(targets),
        }

        logger.info(f"ROC-AUC: {auc_score:.4f}")
        logger.info(f"Optimal threshold: {optimal_threshold:.4f}")

        return result

    def _compute_roc_curve(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        num_thresholds: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute ROC curve using manual threshold sweeping.

        Args:
            probs: Prediction probabilities.
            targets: Ground truth labels.
            num_thresholds: Number of thresholds to evaluate.

        Returns:
            Tuple of (fpr, tpr, thresholds).
        """
        probs_np = probs.cpu().numpy()
        targets_np = targets.cpu().numpy()

        # Sort by probability
        sorted_indices = np.argsort(probs_np)[::-1]
        sorted_probs = probs_np[sorted_indices]
        sorted_targets = targets_np[sorted_indices]

        # Total positives and negatives
        total_positives = np.sum(targets_np == 1)
        total_negatives = np.sum(targets_np == 0)

        if total_positives == 0 or total_negatives == 0:
            logger.warning("Cannot compute ROC: only one class present")
            return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0])

        # Generate thresholds
        thresholds = np.linspace(0.0, 1.0, num_thresholds)

        fpr_list = []
        tpr_list = []

        for threshold in thresholds:
            preds = (sorted_probs >= threshold).astype(int)

            # True positives and false positives
            tp = np.sum((preds == 1) & (sorted_targets == 1))
            fp = np.sum((preds == 1) & (sorted_targets == 0))

            tpr_list.append(tp / total_positives)
            fpr_list.append(fp / total_negatives)

        return np.array(fpr_list), np.array(tpr_list), thresholds

    def _compute_metrics_at_threshold(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, float]:
        """Compute classification metrics at a specific threshold.

        Args:
            preds: Binary predictions at threshold.
            targets: Ground truth labels.

        Returns:
            Dictionary of metrics.
        """
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()

        tp = int(np.sum((preds_np == 1) & (targets_np == 1)))
        fp = int(np.sum((preds_np == 1) & (targets_np == 0)))
        tn = int(np.sum((preds_np == 0) & (targets_np == 0)))
        fn = int(np.sum((preds_np == 0) & (targets_np == 1)))

        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        }

    def compute_multiple_thresholds(
        self,
        probs: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
        thresholds: list[float] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Compute metrics at multiple specific thresholds.

        Args:
            probs: Prediction probabilities.
            targets: Ground truth labels.
            thresholds: List of thresholds to evaluate.

        Returns:
            Dictionary mapping threshold to metrics.
        """
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        if isinstance(probs, np.ndarray):
            probs = torch.from_numpy(probs).float()
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).long()

        if self.num_classes == 2 and probs.dim() == 2:
            probs = probs[:, 1]

        results = {}
        for threshold in thresholds:
            preds = (probs >= threshold).long()
            results[f"{threshold:.1f}"] = self._compute_metrics_at_threshold(
                preds, targets
            )

        return results

    def save(self, result: dict[str, Any], output_path: Path) -> None:
        """Save ROC-AUC analysis to JSON.

        Args:
            result: Result dictionary from compute().
            output_path: Path to save JSON file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved ROC-AUC analysis to {output_path}")
