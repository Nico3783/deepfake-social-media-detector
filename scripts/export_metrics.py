#!/usr/bin/env python3
"""
Experiment Metrics Export Script

Purpose: Export experiment metrics to CSV and JSON for Chapter 4 evidence package.
Responsibilities: Read experiment JSON files, extract metrics, generate exports.
Dependencies: json, pathlib, csv

Research Traceability:
    Research Objective: Generate Chapter 4 evidence package
    Methodology: Systematic export of experimental results
    Implementation: scripts/export_metrics.py

Usage:
    python scripts/export_metrics.py --experiments outputs/experiments/ --output thesis/evidence/
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def load_experiments(experiments_dir: Path) -> list[dict[str, Any]]:
    """Load all experiment JSON files from directory."""
    experiments = []
    for exp_file in experiments_dir.glob("*.json"):
        with open(exp_file) as f:
            data = json.load(f)
            experiments.append(data)
    return experiments


def export_metrics_csv(experiments: list[dict[str, Any]], output_path: Path) -> None:
    """Export experiment metrics to CSV."""
    rows = []
    for exp in experiments:
        if exp.get("status") != "completed":
            continue

        metrics = exp.get("metrics", {})
        config = exp.get("config", {})
        hardware = exp.get("hardware", {})

        rows.append({
            "experiment_id": exp.get("experiment_id", ""),
            "model_type": exp.get("model_type", ""),
            "dataset": exp.get("dataset", ""),
            "accuracy": metrics.get("accuracy", 0),
            "precision": metrics.get("precision", 0),
            "recall": metrics.get("recall", 0),
            "f1_score": metrics.get("f1_score", 0),
            "roc_auc": metrics.get("roc_auc", 0),
            "specificity": metrics.get("specificity", 0),
            "sensitivity": metrics.get("sensitivity", 0),
            "learning_rate": config.get("learning_rate", 0),
            "batch_size": config.get("batch_size", 0),
            "optimizer": config.get("optimizer", ""),
            "epochs_completed": exp.get("epochs_completed", 0),
            "training_time": exp.get("training_time", ""),
            "gpu": hardware.get("gpu", ""),
            "created_at": exp.get("created_at", ""),
            "completed_at": exp.get("completed_at", ""),
        })

    if not rows:
        print("No completed experiments to export.")
        return

    fieldnames = rows[0].keys()
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} experiments to: {output_path}")


def export_confusion_matrices(experiments: list[dict[str, Any]], output_path: Path) -> None:
    """Export confusion matrices to CSV."""
    rows = []
    for exp in experiments:
        if exp.get("status") != "completed":
            continue

        cm = exp.get("confusion_matrix", {})
        if cm:
            rows.append({
                "experiment_id": exp.get("experiment_id", ""),
                "model_type": exp.get("model_type", ""),
                "true_positive": cm.get("true_positive", 0),
                "false_positive": cm.get("false_positive", 0),
                "true_negative": cm.get("true_negative", 0),
                "false_negative": cm.get("false_negative", 0),
            })

    if not rows:
        print("No confusion matrices to export.")
        return

    fieldnames = rows[0].keys()
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} confusion matrices to: {output_path}")


def export_training_history(experiments: list[dict[str, Any]], output_dir: Path) -> None:
    """Export training history to individual CSV files."""
    for exp in experiments:
        if exp.get("status") != "completed":
            continue

        history = exp.get("training_history", {})
        if not history:
            continue

        exp_id = exp.get("experiment_id", "unknown")
        output_path = output_dir / f"{exp_id}_training_history.csv"

        # Find max epochs
        max_epochs = max(
            len(history.get("train_loss", [])),
            len(history.get("val_loss", [])),
            len(history.get("train_acc", [])),
            len(history.get("val_acc", [])),
        )

        rows = []
        for i in range(max_epochs):
            rows.append({
                "epoch": i + 1,
                "train_loss": history.get("train_loss", [])[i] if i < len(history.get("train_loss", [])) else "",
                "val_loss": history.get("val_loss", [])[i] if i < len(history.get("val_loss", [])) else "",
                "train_acc": history.get("train_acc", [])[i] if i < len(history.get("train_acc", [])) else "",
                "val_acc": history.get("val_acc", [])[i] if i < len(history.get("val_acc", [])) else "",
                "learning_rate": history.get("learning_rate", [])[i] if i < len(history.get("learning_rate", [])) else "",
            })

        fieldnames = ["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "learning_rate"]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Exported training history to: {output_path}")


def generate_summary_report(experiments: list[dict[str, Any]], output_path: Path) -> None:
    """Generate a summary report of all experiments."""
    completed = [e for e in experiments if e.get("status") == "completed"]

    report = {
        "generated_at": datetime.now().isoformat(),
        "total_experiments": len(experiments),
        "completed_experiments": len(completed),
        "experiments": [],
    }

    for exp in completed:
        metrics = exp.get("metrics", {})
        report["experiments"].append({
            "experiment_id": exp.get("experiment_id", ""),
            "model_type": exp.get("model_type", ""),
            "dataset": exp.get("dataset", ""),
            "accuracy": metrics.get("accuracy", 0),
            "f1_score": metrics.get("f1_score", 0),
            "roc_auc": metrics.get("roc_auc", 0),
            "training_time": exp.get("training_time", ""),
        })

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Generated summary report: {output_path}")


def main() -> None:
    """Main function to export experiment metrics."""
    parser = argparse.ArgumentParser(description="Export experiment metrics")
    parser.add_argument(
        "--experiments",
        type=Path,
        default=Path("outputs/experiments"),
        help="Directory containing experiment JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("thesis/evidence"),
        help="Output directory for exports",
    )
    args = parser.parse_args()

    # Create output directories
    args.output.mkdir(parents=True, exist_ok=True)

    # Load experiments
    print(f"Loading experiments from: {args.experiments}")
    experiments = load_experiments(args.experiments)
    print(f"Found {len(experiments)} experiment(s)")

    if not experiments:
        print("No experiments found. Generate experiments first.")
        return

    # Export metrics
    print("\nExporting metrics...")
    export_metrics_csv(experiments, args.output / "experiment_metrics.csv")

    # Export confusion matrices
    print("\nExporting confusion matrices...")
    export_confusion_matrices(experiments, args.output / "confusion_matrices.csv")

    # Export training history
    print("\nExporting training history...")
    export_training_history(experiments, args.output)

    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(experiments, args.output / "experiment_summary.json")

    print(f"\nMetrics export completed successfully in: {args.output}")
    print("Files generated:")
    for file in sorted(args.output.glob("*")):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()