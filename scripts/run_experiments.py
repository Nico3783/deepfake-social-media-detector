#!/usr/bin/env python3
"""
Experiment runner for thesis contributions.

Purpose: Execute all 4 contribution experiments systematically.
Usage: python -m scripts.run_experiments [--experiment all|compression|cross_dataset|comparison|deployment]

Research Traceability:
    Research Objective: Evaluate deepfake detection under realistic social media conditions
    Methodology: Systematic experiments across compression levels, datasets, and architectures
    Implementation: scripts/run_experiments.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config.settings import Settings, ModelConfig, DataConfig
from src.models.model_factory import create_model, get_model_summary
from src.evaluation.evaluate import Evaluator
from src.visualization.plots import PlotGenerator
from src.utils.logger import setup_logger
from src.utils.helpers import get_device

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------

COMPRESSION_LEVELS = {
    "c0": "raw",
    "c23": "light_compression",
    "c40": "heavy_compression",
}

MODELS = {
    "xception": {"name": "XceptionNet", "input_size": 299},
    "efficientnet": {"name": "EfficientNet-B0", "input_size": 224},
}

DATASETS = {
    "faceforensics": "FaceForensics++",
    "celebdf": "Celeb-DF",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_test_data(
    dataset_name: str,
    compression: str | None = None,
    batch_size: int = 32,
    image_size: int = 299,
) -> DataLoader | None:
    """Load test dataset.

    Args:
        dataset_name: Dataset identifier ('faceforensics' or 'celebdf').
        compression: Compression level ('c0', 'c23', 'c40') or None.
        batch_size: Batch size for data loader.
        image_size: Target image size.

    Returns:
        DataLoader if data exists, None otherwise.
    """
    metadata_dir = Path("datasets/metadata")
    test_csv = metadata_dir / f"{dataset_name}_test.csv"

    if not test_csv.exists():
        logger.warning(f"Test data not found: {test_csv}")
        return None

    from src.data.dataset import DeepfakeDataset

    dataset = DeepfakeDataset(
        metadata_path=test_csv,
        root_dir=Path("datasets/processed"),
        mode="test",
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )


def measure_inference_speed(
    model: nn.Module,
    input_size: int = 299,
    num_iterations: int = 100,
    batch_size: int = 1,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Measure model inference speed.

    Args:
        model: PyTorch model.
        input_size: Input image size.
        num_iterations: Number of forward passes.
        batch_size: Batch size.
        device: Compute device.

    Returns:
        Dictionary with timing metrics.
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, input_size, input_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Synchronize if CUDA
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_batch = total_time / num_iterations
    fps = batch_size / avg_time_per_batch

    # Memory usage
    if device.type == "cuda":
        memory_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        memory_allocated = 0.0

    return {
        "total_time_s": round(total_time, 4),
        "avg_time_per_batch_ms": round(avg_time_per_batch * 1000, 2),
        "fps": round(fps, 2),
        "memory_mb": round(memory_allocated, 2),
        "num_iterations": num_iterations,
    }


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_compression_experiment(
    models_dict: dict[str, nn.Module],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Contribution 1: Compression impact quantification.

    Tests each model across c0, c23, c40 compression levels.

    Args:
        models_dict: Dictionary of model_name -> model.
        output_dir: Output directory for results.
        device: Compute device.

    Returns:
        Results dictionary.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: Compression Impact Quantification")
    logger.info("=" * 60)

    results = {
        "experiment": "compression_impact",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "models": {},
    }

    for model_name, model in models_dict.items():
        logger.info(f"\nEvaluating {model_name} across compression levels...")
        model_results = {}

        for compression, description in COMPRESSION_LEVELS.items():
            logger.info(f"  Compression: {compression} ({description})")

            test_loader = load_test_data(
                dataset_name="faceforensics",
                compression=compression,
                batch_size=32,
                image_size=MODELS[model_name]["input_size"],
            )

            if test_loader is None:
                logger.warning(f"  Skipping {compression}: no test data")
                model_results[compression] = {"status": "no_data"}
                continue

            evaluator = Evaluator(model, device, num_classes=2)
            metrics = evaluator.evaluate(test_loader)

            model_results[compression] = {
                "accuracy": metrics.get("accuracy", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1": metrics.get("f1", 0.0),
                "auroc": metrics.get("auroc", 0.0),
                "samples_per_second": metrics.get("samples_per_second", 0.0),
            }

            logger.info(
                f"    Accuracy: {model_results[compression]['accuracy']:.4f}, "
                f"F1: {model_results[compression]['f1']:.4f}"
            )

        results["models"][model_name] = model_results

    # Save results
    output_path = output_dir / "compression_impact.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # Generate visualization
    plotter = PlotGenerator(output_dir / "plots")
    if all(
        isinstance(r, dict) and "accuracy" in r
        for model_r in results["models"].values()
        for r in model_r.values()
    ):
        plotter.plot_compression_comparison(results)

    return results


def run_cross_dataset_experiment(
    models_dict: dict[str, nn.Module],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Contribution 2: Cross-dataset generalization.

    Train on FF++, test on Celeb-DF (and vice versa).

    Args:
        models_dict: Dictionary of model_name -> model.
        output_dir: Output directory for results.
        device: Compute device.

    Returns:
        Results dictionary.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2: Cross-Dataset Generalization")
    logger.info("=" * 60)

    results = {
        "experiment": "cross_dataset_generalization",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "models": {},
    }

    for model_name, model in models_dict.items():
        logger.info(f"\nEvaluating {model_name} cross-dataset...")
        model_results = {}

        # Test direction 1: FF++ train -> Celeb-DF test
        logger.info("  Direction: FaceForensics++ -> Celeb-DF")
        test_loader = load_test_data(
            dataset_name="celebdf",
            batch_size=32,
            image_size=MODELS[model_name]["input_size"],
        )

        if test_loader is not None:
            evaluator = Evaluator(model, device, num_classes=2)
            metrics = evaluator.evaluate(test_loader)
            model_results["ff_to_celebdf"] = {
                "accuracy": metrics.get("accuracy", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1": metrics.get("f1", 0.0),
                "auroc": metrics.get("auroc", 0.0),
            }
        else:
            model_results["ff_to_celebdf"] = {"status": "no_data"}

        # Test direction 2: Celeb-DF train -> FF++ test
        logger.info("  Direction: Celeb-DF -> FaceForensics++")
        test_loader = load_test_data(
            dataset_name="faceforensics",
            batch_size=32,
            image_size=MODELS[model_name]["input_size"],
        )

        if test_loader is not None:
            evaluator = Evaluator(model, device, num_classes=2)
            metrics = evaluator.evaluate(test_loader)
            model_results["celebdf_to_ff"] = {
                "accuracy": metrics.get("accuracy", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1": metrics.get("f1", 0.0),
                "auroc": metrics.get("auroc", 0.0),
            }
        else:
            model_results["celebdf_to_ff"] = {"status": "no_data"}

        results["models"][model_name] = model_results

    # Save results
    output_path = output_dir / "cross_dataset_generalization.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    return results


def run_model_comparison_experiment(
    models_dict: dict[str, nn.Module],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Contribution 3: Head-to-head model comparison.

    Compare XceptionNet vs EfficientNet across all conditions.

    Args:
        models_dict: Dictionary of model_name -> model.
        output_dir: Output directory for results.
        device: Compute device.

    Returns:
        Results dictionary.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 3: Head-to-Head Model Comparison")
    logger.info("=" * 60)

    results = {
        "experiment": "model_comparison",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "conditions": {},
    }

    # Compare on FaceForensics++ c0
    logger.info("\nCondition: FaceForensics++ (c0)")
    condition_results = {}

    test_loader = load_test_data(
        dataset_name="faceforensics",
        compression="c0",
        batch_size=32,
        image_size=299,  # Use max size
    )

    if test_loader is not None:
        for model_name, model in models_dict.items():
            evaluator = Evaluator(model, device, num_classes=2)
            metrics = evaluator.evaluate(test_loader)
            condition_results[model_name] = {
                "accuracy": metrics.get("accuracy", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1": metrics.get("f1", 0.0),
                "auroc": metrics.get("auroc", 0.0),
                "samples_per_second": metrics.get("samples_per_second", 0.0),
            }
    else:
        for model_name in models_dict:
            condition_results[model_name] = {"status": "no_data"}

    results["conditions"]["faceforensics_c0"] = condition_results

    # Compare on Celeb-DF
    logger.info("\nCondition: Celeb-DF")
    condition_results = {}

    test_loader = load_test_data(
        dataset_name="celebdf",
        batch_size=32,
        image_size=299,
    )

    if test_loader is not None:
        for model_name, model in models_dict.items():
            evaluator = Evaluator(model, device, num_classes=2)
            metrics = evaluator.evaluate(test_loader)
            condition_results[model_name] = {
                "accuracy": metrics.get("accuracy", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1": metrics.get("f1", 0.0),
                "auroc": metrics.get("auroc", 0.0),
                "samples_per_second": metrics.get("samples_per_second", 0.0),
            }
    else:
        for model_name in models_dict:
            condition_results[model_name] = {"status": "no_data"}

    results["conditions"]["celebdf"] = condition_results

    # Save results
    output_path = output_dir / "model_comparison.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # Generate visualization
    plotter = PlotGenerator(output_dir / "plots")
    plotter.plot_model_comparison(results)

    return results


def run_deployment_experiment(
    models_dict: dict[str, nn.Module],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Contribution 4: Practical deployment assessment.

    Measure inference speed, model size, and memory footprint.

    Args:
        models_dict: Dictionary of model_name -> model.
        output_dir: Output directory for results.
        device: Compute device.

    Returns:
        Results dictionary.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 4: Practical Deployment Assessment")
    logger.info("=" * 60)

    results = {
        "experiment": "deployment_assessment",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "models": {},
    }

    for model_name, model in models_dict.items():
        logger.info(f"\nProfiling {model_name}...")

        # Model summary
        summary = get_model_summary(model)

        # Inference speed
        speed_metrics = measure_inference_speed(
            model,
            input_size=MODELS[model_name]["input_size"],
            num_iterations=100,
            batch_size=1,
            device=device,
        )

        # Model size (file size estimation)
        param_count = summary["total_parameters"]
        model_size_mb = param_count * 4 / (1024 ** 2)  # float32

        results["models"][model_name] = {
            "total_parameters": param_count,
            "trainable_parameters": summary["trainable_parameters"],
            "model_size_mb": round(model_size_mb, 2),
            "inference_speed": speed_metrics,
            "input_size": MODELS[model_name]["input_size"],
        }

        logger.info(f"  Parameters: {param_count:,}")
        logger.info(f"  Model size: {model_size_mb:.2f} MB")
        logger.info(f"  FPS: {speed_metrics['fps']:.2f}")
        logger.info(f"  Latency: {speed_metrics['avg_time_per_batch_ms']:.2f} ms")

    # Save results
    output_path = output_dir / "deployment_assessment.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run selected experiments."""
    parser = argparse.ArgumentParser(
        description="Run thesis contribution experiments"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "compression", "cross_dataset", "comparison", "deployment"],
        help="Which experiment to run",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/experiments",
        help="Output directory",
    )
    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Device: {device}")
    logger.info(f"Output: {output_dir}")

    # Create models
    models_dict = {}
    for model_name in MODELS:
        try:
            config = ModelConfig(
                name=model_name,
                num_classes=2,
                pretrained=True,
            )
            model = create_model(config, device)
            models_dict[model_name] = model
            logger.info(f"Created {model_name} model")
        except Exception as e:
            logger.error(f"Failed to create {model_name}: {e}")

    if not models_dict:
        logger.error("No models created. Exiting.")
        return

    # Run experiments
    all_results = {}

    if args.experiment in ("all", "compression"):
        all_results["compression"] = run_compression_experiment(
            models_dict, output_dir, device
        )

    if args.experiment in ("all", "cross_dataset"):
        all_results["cross_dataset"] = run_cross_dataset_experiment(
            models_dict, output_dir, device
        )

    if args.experiment in ("all", "comparison"):
        all_results["comparison"] = run_model_comparison_experiment(
            models_dict, output_dir, device
        )

    if args.experiment in ("all", "deployment"):
        all_results["deployment"] = run_deployment_experiment(
            models_dict, output_dir, device
        )

    # Save combined results
    combined_path = output_dir / "all_experiments.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info("Next step: python scripts/generate_thesis_assets.py")


if __name__ == "__main__":
    main()
