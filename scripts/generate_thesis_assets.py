#!/usr/bin/env python3
"""
Thesis Asset Generation Script

Purpose: Generate tables, graphs, and other assets for Chapter 4 of the thesis.
Responsibilities: Load experiment results, generate comparison tables, create visualizations.
Dependencies: json, pathlib, matplotlib, pandas

Research Traceability:
    Research Objective: Generate thesis-ready assets for Chapter 4
    Methodology: Automated generation of experimental result summaries
    Implementation: scripts/generate_thesis_assets.py

Usage:
    python scripts/generate_thesis_assets.py --experiments outputs/experiments/ --output thesis/assets/
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def load_experiments(experiments_dir: Path) -> list[dict[str, Any]]:
    """Load all experiment JSON files from directory."""
    experiments = []
    for exp_file in experiments_dir.glob("*.json"):
        with open(exp_file) as f:
            data = json.load(f)
            experiments.append(data)
    return experiments


def generate_model_comparison_table(experiments: list[dict[str, Any]]) -> pd.DataFrame:
    """Generate model comparison table for Chapter 4."""
    rows = []
    for exp in experiments:
        if exp.get("status") != "completed":
            continue
        rows.append({
            "Model": exp.get("model_type", "Unknown"),
            "Dataset": exp.get("dataset", "Unknown"),
            "Accuracy": exp.get("metrics", {}).get("accuracy", 0),
            "Precision": exp.get("metrics", {}).get("precision", 0),
            "Recall": exp.get("metrics", {}).get("recall", 0),
            "F1 Score": exp.get("metrics", {}).get("f1_score", 0),
            "AUC-ROC": exp.get("metrics", {}).get("roc_auc", 0),
            "Epochs": exp.get("epochs_completed", 0),
            "Training Time": exp.get("training_time", "N/A"),
        })
    return pd.DataFrame(rows)


def generate_hyperparameter_table(experiments: list[dict[str, Any]]) -> pd.DataFrame:
    """Generate hyperparameter configuration table."""
    rows = []
    for exp in experiments:
        config = exp.get("config", {})
        rows.append({
            "Experiment": exp.get("experiment_id", "Unknown"),
            "Model": exp.get("model_type", "Unknown"),
            "Learning Rate": config.get("learning_rate", 0),
            "Batch Size": config.get("batch_size", 0),
            "Optimizer": config.get("optimizer", "Unknown"),
            "Weight Decay": config.get("weight_decay", 0),
            "Image Size": config.get("image_size", 0),
            "Early Stopping": config.get("early_stopping_patience", 0),
        })
    return pd.DataFrame(rows)


def plot_training_curves(experiments: list[dict[str, Any]], output_dir: Path) -> None:
    """Generate training curve plots for each experiment."""
    for exp in experiments:
        if exp.get("status") != "completed":
            continue

        history = exp.get("training_history", {})
        if not history:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss curve
        if "train_loss" in history and "val_loss" in history:
            epochs = range(1, len(history["train_loss"]) + 1)
            axes[0].plot(epochs, history["train_loss"], label="Train Loss")
            axes[0].plot(epochs, history["val_loss"], label="Val Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Training and Validation Loss")
            axes[0].legend()
            axes[0].grid(True)

        # Accuracy curve
        if "train_acc" in history and "val_acc" in history:
            epochs = range(1, len(history["train_acc"]) + 1)
            axes[1].plot(epochs, history["train_acc"], label="Train Accuracy")
            axes[1].plot(epochs, history["val_acc"], label="Val Accuracy")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Accuracy")
            axes[1].set_title("Training and Validation Accuracy")
            axes[1].legend()
            axes[1].grid(True)

        plt.tight_layout()
        exp_id = exp.get("experiment_id", "unknown")
        plt.savefig(output_dir / f"{exp_id}_training_curves.png", dpi=150)
        plt.close()


def plot_model_comparison(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate model comparison bar chart."""
    if results_df.empty:
        return

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]
    models = results_df["Model"].unique()

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(models))
    width = 0.15

    for i, metric in enumerate(metrics):
        values = results_df[metric].values
        ax.bar([xi + i * width for xi in x], values, width, label=metric)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks([xi + width * 2 for xi in x])
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=150)
    plt.close()


def generate_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Generate LaTeX table from DataFrame."""
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{{label}}}\n"
    latex += df.to_latex(index=False, float_format="%.4f")
    latex += "\\end{table}\n"
    return latex


def main() -> None:
    """Main function to generate thesis assets."""
    parser = argparse.ArgumentParser(description="Generate thesis assets")
    parser.add_argument(
        "--experiments",
        type=Path,
        default=Path("outputs/experiments"),
        help="Directory containing experiment JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("thesis/assets"),
        help="Output directory for thesis assets",
    )
    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load experiments
    print(f"Loading experiments from: {args.experiments}")
    experiments = load_experiments(args.experiments)
    print(f"Found {len(experiments)} experiment(s)")

    if not experiments:
        print("No experiments found. Generate experiments first.")
        return

    # Generate comparison table
    print("Generating model comparison table...")
    comparison_df = generate_model_comparison_table(experiments)
    comparison_df.to_csv(args.output / "model_comparison.csv", index=False)

    # Generate hyperparameter table
    print("Generating hyperparameter table...")
    hyperparam_df = generate_hyperparameter_table(experiments)
    hyperparam_df.to_csv(args.output / "hyperparameters.csv", index=False)

    # Generate LaTeX tables
    print("Generating LaTeX tables...")
    latex_comparison = generate_latex_table(
        comparison_df,
        caption="Model Performance Comparison",
        label="tab:model_comparison",
    )
    with open(args.output / "model_comparison.tex", "w") as f:
        f.write(latex_comparison)

    latex_hyperparam = generate_latex_table(
        hyperparam_df,
        caption="Hyperparameter Configurations",
        label="tab:hyperparameters",
    )
    with open(args.output / "hyperparameters.tex", "w") as f:
        f.write(latex_hyperparam)

    # Generate plots
    print("Generating training curves...")
    plot_training_curves(experiments, args.output)

    print("Generating model comparison plot...")
    plot_model_comparison(comparison_df, args.output)

    # Generate summary report
    print("Generating summary report...")
    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_experiments": len(experiments),
        "completed_experiments": sum(1 for e in experiments if e.get("status") == "completed"),
        "best_model": comparison_df.loc[comparison_df["Accuracy"].idxmax()]["Model"] if not comparison_df.empty else "N/A",
        "best_accuracy": float(comparison_df["Accuracy"].max()) if not comparison_df.empty else 0,
    }
    with open(args.output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nThesis assets generated successfully in: {args.output}")
    print("Files generated:")
    for file in sorted(args.output.glob("*")):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()