"""
Confusion matrix analysis.

Purpose: Compute, analyze, and visualize confusion matrices for deepfake detection.
Responsibilities: Confusion matrix computation, error analysis, per-class breakdown.
Dependencies: numpy, torch, torchmetrics

Research Traceability:
    Research Objective: Detailed classification error analysis
    Methodology: Confusion matrix decomposition into TP, FP, TN, FN
    Implementation: src/evaluation/confusion_matrix.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchmetrics import ConfusionMatrix as TorchConfusionMatrix

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ConfusionMatrixAnalyzer:
    """Analyze classification results using confusion matrices.

    Provides:
    - Raw confusion matrix computation
    - Per-class TP, FP, TN, FN breakdown
    - Error analysis (false positives vs false negatives)
    - Normalized confusion matrix
    """

    def __init__(
        self,
        num_classes: int = 2,
        class_names: list[str] | None = None,
    ) -> None:
        """Initialize confusion matrix analyzer.

        Args:
            num_classes: Number of classes.
            class_names: Names for each class index.
        """
        self.num_classes = num_classes
        self.class_names = class_names or ["real", "fake"]

        self.confusion_matrix = TorchConfusionMatrix(
            task="binary" if num_classes == 2 else "multiclass",
            num_classes=num_classes,
        )

        logger.info(
            f"ConfusionMatrixAnalyzer initialized: "
            f"{num_classes} classes ({', '.join(self.class_names)})"
        )

    def compute(
        self,
        preds: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
    ) -> dict[str, Any]:
        """Compute confusion matrix and derived metrics.

        Args:
            preds: Predicted labels.
            targets: Ground truth labels.

        Returns:
            Dictionary with confusion matrix data and analysis.
        """
        # Convert to tensors if needed
        if isinstance(preds, np.ndarray):
            preds = torch.from_numpy(preds)
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)

        # Compute confusion matrix
        cm = self.confusion_matrix(preds, targets)
        cm_np = cm.cpu().numpy()

        # Build result
        result: dict[str, Any] = {
            "confusion_matrix": cm_np.tolist(),
            "num_classes": self.num_classes,
            "class_names": self.class_names,
        }

        # Per-class breakdown
        result["per_class"] = self._per_class_breakdown(cm_np)

        # Error analysis
        result["error_analysis"] = self._error_analysis(cm_np)

        # Normalized matrix (row-normalized = recall per class)
        row_sums = cm_np.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        result["normalized_by_row"] = (cm_np / row_sums).tolist()

        # Normalized matrix (column-normalized = precision per class)
        col_sums = cm_np.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        result["normalized_by_col"] = (cm_np / col_sums).tolist()

        # Overall metrics from matrix
        result["overall"] = self._overall_metrics(cm_np)

        return result

    def _per_class_breakdown(self, cm: np.ndarray) -> dict[str, dict[str, int]]:
        """Break down confusion matrix into per-class TP, FP, TN, FN.

        Args:
            cm: Confusion matrix as numpy array.

        Returns:
            Per-class breakdown dictionary.
        """
        breakdown = {}

        for i, class_name in enumerate(self.class_names):
            tp = int(cm[i, i])
            fp = int(cm[:, i].sum() - cm[i, i])
            fn = int(cm[i, :].sum() - cm[i, i])
            tn = int(cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i])

            breakdown[class_name] = {
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
                "support": int(cm[i, :].sum()),
            }

        return breakdown

    def _error_analysis(self, cm: np.ndarray) -> dict[str, Any]:
        """Analyze classification errors.

        Args:
            cm: Confusion matrix as numpy array.

        Returns:
            Error analysis dictionary.
        """
        total_samples = int(cm.sum())
        total_errors = int(cm.sum() - np.trace(cm))

        # False positives (fake predicted as real)
        false_positives = 0
        false_negatives = 0

        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j:
                    if j == 0:  # Predicted as real (class 0)
                        false_positives += int(cm[i, j])
                    else:  # Predicted as fake (class 1)
                        false_negatives += int(cm[i, j])

        # For binary case, simplify
        if self.num_classes == 2:
            false_positives = int(cm[0, 1])  # Real predicted as fake
            false_negatives = int(cm[1, 0])  # Fake predicted as real

        return {
            "total_samples": total_samples,
            "total_errors": total_errors,
            "error_rate": total_errors / total_samples if total_samples > 0 else 0.0,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "fp_rate": false_positives / total_samples if total_samples > 0 else 0.0,
            "fn_rate": false_negatives / total_samples if total_samples > 0 else 0.0,
        }

    def _overall_metrics(self, cm: np.ndarray) -> dict[str, float]:
        """Compute overall metrics from confusion matrix.

        Args:
            cm: Confusion matrix as numpy array.

        Returns:
            Overall metrics dictionary.
        """
        # For binary classification
        if self.num_classes == 2:
            tp = cm[1, 1]
            fp = cm[0, 1]
            tn = cm[0, 0]
            fn = cm[1, 0]

            accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            return {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }

        # For multiclass: macro-averaged
        accuracy = float(np.trace(cm) / cm.sum()) if cm.sum() > 0 else 0.0

        precisions = []
        recalls = []
        for i in range(self.num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp

            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)

        macro_precision = float(np.mean(precisions))
        macro_recall = float(np.mean(recalls))
        macro_f1 = (
            2 * macro_precision * macro_recall / (macro_precision + macro_recall)
            if (macro_precision + macro_recall) > 0
            else 0.0
        )

        return {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
        }

    def save(self, result: dict[str, Any], output_path: Path) -> None:
        """Save confusion matrix analysis to JSON.

        Args:
            result: Result dictionary from compute().
            output_path: Path to save JSON file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved confusion matrix analysis to {output_path}")
