"""
Publication-quality static plots for thesis assets.

Purpose: Generate training curves, ROC curves, confusion matrices, and comparison plots.
Responsibilities: matplotlib/seaborn figure generation with consistent styling.
Dependencies: matplotlib, seaborn, numpy, pathlib

Research Traceability:
    Research Objective: Visual evaluation of deepfake detection performance
    Methodology: Standard ML visualization (training dynamics, classification analysis)
    Implementation: src/visualization/plots.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

from src.utils.logger import setup_logger
from src.utils.paths import PathConfig

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Thesis-consistent style
# ---------------------------------------------------------------------------
THESIS_STYLE: dict[str, Any] = {
    "figure.figsize": (8, 6),
    "figure.dpi": 150,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.constrained_layout.use": True,
}

CLASS_COLORS = {
    "real": "#2196F3",
    "fake": "#F44336",
    "xception": "#1976D2",
    "efficientnet": "#388E3C",
}


class PlotGenerator:
    """Generate publication-quality static plots for thesis evaluation.

    Provides methods for:
    - Training loss/accuracy curves
    - ROC curves with AUC scores
    - Confusion matrix heatmaps
    - Model comparison bar charts
    """

    def __init__(self, output_dir: str | Path | None = None) -> None:
        """Initialize plot generator.

        Args:
            output_dir: Directory for saved figures. Uses PathConfig.OUTPUT_REPORTS if None.
        """
        if output_dir is None:
            paths = PathConfig()
            self.output_dir = paths.output_reports / "figures"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._apply_style()

    def _apply_style(self) -> None:
        """Apply thesis-consistent matplotlib style."""
        plt.rcParams.update(THESIS_STYLE)
        sns.set_palette("husl")

    def _save(self, fig: plt.Figure, name: str) -> Path:
        """Save figure to output directory.

        Args:
            fig: Matplotlib figure.
            name: Base filename (without extension).

        Returns:
            Path to saved figure.
        """
        path = self.output_dir / f"{name}.pdf"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        logger.info("Saved plot: %s", path)
        return path

    # ------------------------------------------------------------------
    # Training curves
    # ------------------------------------------------------------------
    def plot_training_curves(
        self,
        train_losses: list[float],
        val_losses: list[float],
        train_accs: list[float],
        val_accs: list[float],
        title: str = "Training Curves",
        filename: str = "training_curves",
    ) -> Path:
        """Plot training and validation loss/accuracy curves.

        Args:
            train_losses: Training loss per epoch.
            val_losses: Validation loss per epoch.
            train_accs: Training accuracy per epoch.
            val_accs: Validation accuracy per epoch.
            title: Plot title.
            filename: Output filename (without extension).

        Returns:
            Path to saved figure.
        """
        epochs = list(range(1, len(train_losses) + 1))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curve
        ax1.plot(epochs, train_losses, "o-", label="Train Loss", color=CLASS_COLORS["real"])
        ax1.plot(epochs, val_losses, "s--", label="Val Loss", color=CLASS_COLORS["fake"])
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss")
        ax1.legend()

        # Accuracy curve
        ax2.plot(epochs, train_accs, "o-", label="Train Acc", color=CLASS_COLORS["real"])
        ax2.plot(epochs, val_accs, "s--", label="Val Acc", color=CLASS_COLORS["fake"])
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy")
        ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        ax2.legend()

        fig.suptitle(title, fontsize=16, fontweight="bold")
        return self._save(fig, filename)

    # ------------------------------------------------------------------
    # ROC curve
    # ------------------------------------------------------------------
    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc_score: float,
        model_name: str = "Model",
        filename: str = "roc_curve",
    ) -> Path:
        """Plot ROC curve with AUC annotation.

        Args:
            fpr: False positive rates.
            tpr: True positive rates.
            auc_score: Area under the ROC curve.
            model_name: Model label for legend.
            filename: Output filename.

        Returns:
            Path to saved figure.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(
            fpr, tpr,
            color=CLASS_COLORS["xception"],
            lw=2,
            label=f"{model_name} (AUC = {auc_score:.4f})",
        )
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic")
        ax.legend(loc="lower right")
        return self._save(fig, filename)

    def plot_multi_roc(
        self,
        curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
        filename: str = "roc_comparison",
    ) -> Path:
        """Plot multiple ROC curves for model comparison.

        Args:
            curves: Dict mapping model_name → (fpr, tpr, auc_score).
            filename: Output filename.

        Returns:
            Path to saved figure.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = list(CLASS_COLORS.values())

        for i, (name, (fpr, tpr, auc_score)) in enumerate(curves.items()):
            color = colors[i % len(colors)]
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {auc_score:.4f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve Comparison")
        ax.legend(loc="lower right")
        return self._save(fig, filename)

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: list[str] | None = None,
        title: str = "Confusion Matrix",
        filename: str = "confusion_matrix",
        normalize: bool = False,
    ) -> Path:
        """Plot confusion matrix heatmap.

        Args:
            cm: 2D confusion matrix (num_classes x num_classes).
            class_names: Labels for each class.
            title: Plot title.
            filename: Output filename.
            normalize: If True, show row-normalized values.

        Returns:
            Path to saved figure.
        """
        if class_names is None:
            class_names = ["Real", "Fake"]

        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm_display = cm.astype(float) / row_sums
            fmt = ".2%"
            vmin, vmax = 0.0, 1.0
        else:
            cm_display = cm.astype(float)
            fmt = "d"
            vmin, vmax = 0, cm.max()

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            cm_display,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(title)
        return self._save(fig, filename)

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------
    def plot_model_comparison(
        self,
        results: dict[str, dict[str, float]],
        metrics: list[str] | None = None,
        filename: str = "model_comparison",
    ) -> Path:
        """Plot grouped bar chart comparing models across metrics.

        Args:
            results: Dict mapping model_name → {metric_name: value}.
            metrics: Subset of metrics to plot (default: accuracy, precision, recall, f1).
            filename: Output filename.

        Returns:
            Path to saved figure.
        """
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1"]

        model_names = list(results.keys())
        n_models = len(model_names)
        n_metrics = len(metrics)
        bar_width = 0.8 / n_models
        x = np.arange(n_metrics)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = list(CLASS_COLORS.values())

        for i, model_name in enumerate(model_names):
            values = [results[model_name].get(m, 0.0) for m in metrics]
            offset = (i - n_models / 2 + 0.5) * bar_width
            bars = ax.bar(
                x + offset, values, bar_width,
                label=model_name,
                color=colors[i % len(colors)],
                edgecolor="white",
                linewidth=0.5,
            )
            # Value labels on bars
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.15)
        ax.set_title("Model Comparison")
        ax.legend(loc="upper right")
        return self._save(fig, filename)

    # ------------------------------------------------------------------
    # Per-class bar chart
    # ------------------------------------------------------------------
    def plot_per_class_metrics(
        self,
        class_metrics: dict[str, dict[str, float]],
        class_names: list[str] | None = None,
        filename: str = "per_class_metrics",
    ) -> Path:
        """Plot per-class precision, recall, F1 as grouped bars.

        Args:
            class_metrics: Dict mapping class_name → {precision, recall, f1}.
            class_names: Class labels (default: use dict keys).
            filename: Output filename.

        Returns:
            Path to saved figure.
        """
        if class_names is None:
            class_names = list(class_metrics.keys())

        metric_names = ["precision", "recall", "f1"]
        n_classes = len(class_names)
        n_metrics = len(metric_names)
        bar_width = 0.25
        x = np.arange(n_classes)

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ["#1976D2", "#388E3C", "#F57C00"]

        for i, metric in enumerate(metric_names):
            values = [class_metrics[c].get(metric, 0.0) for c in class_names]
            ax.bar(x + i * bar_width, values, bar_width, label=metric.capitalize(), color=colors[i])

        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(class_names)
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.15)
        ax.set_title("Per-Class Classification Metrics")
        ax.legend()
        return self._save(fig, filename)

    # ------------------------------------------------------------------
    # Frame-level probability timeline
    # ------------------------------------------------------------------
    def plot_frame_probabilities(
        self,
        probabilities: list[float],
        ground_truth: int | None = None,
        threshold: float = 0.5,
        title: str = "Frame-Level Fake Probability",
        filename: str = "frame_probabilities",
    ) -> Path:
        """Plot per-frame fake probability with threshold line.

        Args:
            probabilities: Fake probability per frame.
            ground_truth: Ground truth label (0=real, 1=fake) for coloring.
            threshold: Decision threshold line.
            title: Plot title.
            filename: Output filename.

        Returns:
            Path to saved figure.
        """
        frames = list(range(1, len(probabilities) + 1))
        fig, ax = plt.subplots(figsize=(12, 4))

        bg_color = "#FFEBEE" if ground_truth == 1 else "#E3F2FD"
        ax.axhspan(threshold, 1.0, alpha=0.1, color="#F44336")
        ax.axhspan(0.0, threshold, alpha=0.1, color="#2196F3")

        ax.plot(frames, probabilities, "o-", color="#1976D2", markersize=3, lw=1.5)
        ax.axhline(y=threshold, color="#F44336", linestyle="--", lw=1, label=f"Threshold = {threshold}")

        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Fake Probability")
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        return self._save(fig, filename)

    # ------------------------------------------------------------------
    # Learning rate schedule
    # ------------------------------------------------------------------
    def plot_learning_rate(
        self,
        lrs: list[float],
        filename: str = "learning_rate_schedule",
    ) -> Path:
        """Plot learning rate schedule over epochs.

        Args:
            lrs: Learning rate at each epoch.
            filename: Output filename.

        Returns:
            Path to saved figure.
        """
        epochs = list(range(1, len(lrs) + 1))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, lrs, "o-", color="#7B1FA2", markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale("log")
        return self._save(fig, filename)
