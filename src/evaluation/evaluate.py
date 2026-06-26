"""
Model evaluation pipeline.

Purpose: Comprehensive evaluation of trained deepfake detection models.
Responsibilities: Load models, run inference on test sets, compute metrics, generate reports.
Dependencies: torch, numpy, pathlib, src.models, src.training.metrics

Research Traceability:
    Research Objective: Evaluate model performance on deepfake detection
    Methodology: Standard classification metrics (accuracy, precision, recall, F1, ROC-AUC)
    Implementation: src/evaluation/evaluate.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.metrics import MetricsTracker
from src.utils.logger import setup_logger
from src.utils.helpers import get_device

logger = setup_logger(__name__)


class Evaluator:
    """Comprehensive model evaluation pipeline.

    Handles:
    - Loading trained models
    - Running inference on test datasets
    - Computing classification metrics
    - Generating evaluation reports
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | None = None,
        num_classes: int = 2,
    ) -> None:
        """Initialize evaluator.

        Args:
            model: Trained model to evaluate.
            device: Device for inference.
            num_classes: Number of classes.
        """
        self.model = model
        self.device = device or get_device()
        self.num_classes = num_classes

        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        # Initialize metrics tracker
        self.metrics = MetricsTracker(num_classes=num_classes, device=self.device)

        logger.info(f"Evaluator initialized on {self.device}")

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> dict[str, Any]:
        """Evaluate model on test dataset.

        Args:
            test_loader: Test data loader.

        Returns:
            Dictionary containing evaluation metrics and predictions.
        """
        logger.info("Starting evaluation...")
        start_time = time.time()

        self.metrics.reset()
        all_preds = []
        all_targets = []
        all_probs = []
        all_video_ids = []

        for batch_idx, batch in enumerate(test_loader):
            # Handle different batch formats
            if len(batch) == 3:
                inputs, targets, video_ids = batch
            else:
                inputs, targets = batch
                video_ids = None

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            # Update metrics
            self.metrics.update(preds, targets, probs)

            # Store predictions
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            if video_ids is not None:
                all_video_ids.extend(video_ids)

            if batch_idx % 10 == 0:
                logger.info(f"Evaluation batch {batch_idx}/{len(test_loader)}")

        # Compute metrics
        metrics = self.metrics.compute()

        # Concatenate all predictions
        metrics["predictions"] = np.concatenate(all_preds).tolist()
        metrics["targets"] = np.concatenate(all_targets).tolist()
        metrics["probabilities"] = np.concatenate(all_probs).tolist()

        if all_video_ids:
            metrics["video_ids"] = all_video_ids

        # Compute per-class metrics
        metrics["per_class"] = self._compute_per_class_metrics(
            np.concatenate(all_targets),
            np.concatenate(all_preds)
        )

        # Compute timing metrics
        total_time = time.time() - start_time
        metrics["total_time"] = total_time
        metrics["samples_per_second"] = len(metrics["predictions"]) / total_time

        logger.info(f"Evaluation completed in {total_time:.2f}s")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1']:.4f}")

        if "auroc" in metrics:
            logger.info(f"ROC-AUC: {metrics['auroc']:.4f}")

        return metrics

    def _compute_per_class_metrics(
        self,
        targets: np.ndarray,
        preds: np.ndarray,
    ) -> dict[str, dict[str, float]]:
        """Compute per-class metrics.

        Args:
            targets: Ground truth labels.
            preds: Predicted labels.

        Returns:
            Dictionary of per-class metrics.
        """
        per_class = {}

        for class_idx in range(self.num_classes):
            class_name = "real" if class_idx == 0 else "fake"

            # True positives, false positives, false negatives
            tp = np.sum((preds == class_idx) & (targets == class_idx))
            fp = np.sum((preds == class_idx) & (targets != class_idx))
            fn = np.sum((preds != class_idx) & (targets == class_idx))
            tn = np.sum((preds != class_idx) & (targets != class_idx))

            # Compute metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / len(targets) if len(targets) > 0 else 0.0

            per_class[class_name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "accuracy": float(accuracy),
                "support": int(np.sum(targets == class_idx)),
            }

        return per_class

    def load_model(self, checkpoint_path: Path) -> None:
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded model from {checkpoint_path}")

    def save_metrics(self, metrics: dict[str, Any], output_path: Path) -> None:
        """Save evaluation metrics to JSON file.

        Args:
            metrics: Evaluation metrics dictionary.
            output_path: Path to save metrics.
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_metrics[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                serializable_metrics[key] = value

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(serializable_metrics, f, indent=2)

        logger.info(f"Saved metrics to {output_path}")

    def compare_models(
        self,
        models: list[nn.Module],
        model_names: list[str],
        test_loader: DataLoader,
    ) -> dict[str, dict[str, Any]]:
        """Compare multiple models on the same test set.

        Args:
            models: List of models to evaluate.
            model_names: Names for each model.
            test_loader: Test data loader.

        Returns:
            Dictionary of results for each model.
        """
        if len(models) != len(model_names):
            raise ValueError("Number of models must match number of names")

        results = {}
        for model, name in zip(models, model_names):
            logger.info(f"Evaluating model: {name}")
            evaluator = Evaluator(model, self.device, self.num_classes)
            results[name] = evaluator.evaluate(test_loader)

        return results


def main() -> None:
    """CLI entry point for model evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate deepfake detection model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Config file")
    parser.add_argument("--output", type=str, default="outputs/eval_results.json", help="Output metrics file")
    args = parser.parse_args()

    from src.config.settings import Settings
    from src.models.model_factory import create_model
    from src.data.dataset import DeepfakeDataset
    from torch.utils.data import DataLoader

    settings = Settings.from_yaml(args.config)
    device = torch.device(settings.get_device())
    model = create_model(settings.config.model, device)

    evaluator = Evaluator(model, device, settings.config.model.num_classes)
    evaluator.load_model(Path(args.checkpoint))

    metadata_dir = Path(settings.config.data.root_dir or "datasets") / "metadata"
    test_dataset = DeepfakeDataset(metadata_path=metadata_dir / "test.csv", root_dir=Path(settings.config.data.root_dir or "datasets"), mode="test")
    test_loader = DataLoader(test_dataset, batch_size=settings.config.training.batch_size, shuffle=False)

    metrics = evaluator.evaluate(test_loader)
    evaluator.save_metrics(metrics, Path(args.output))
