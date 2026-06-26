"""
Training script entry point.

Purpose: Main script for training deepfake detection models.
Dependencies: argparse, pathlib, torch

Research Traceability:
    Research Objective: End-to-end model training pipeline
    Methodology: Configuration-driven training
    Implementation: src/training/train.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config.settings import Settings, ExperimentConfig
from src.models.model_factory import create_model
from src.training.trainer import Trainer
from src.training.losses import get_loss_fn
from src.training.callbacks import EarlyStopping, ModelCheckpoint, LRScheduler
from src.data.dataset import DeepfakeDataset
from src.utils.seed import set_seed
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train deepfake detection model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to training config",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/xception.yaml",
        help="Path to model config",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="configs/dataset.yaml",
        help="Path to dataset config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Training device (cuda/cpu)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for logging",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    return parser.parse_args()


def create_data_loaders(
    config: ExperimentConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders.

    Args:
        config: Full experiment configuration.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    root_dir = Path(config.data.root_dir) if config.data.root_dir else Path("datasets")
    metadata_dir = root_dir / "metadata"

    train_dataset = DeepfakeDataset(
        metadata_path=metadata_dir / "train.csv",
        root_dir=root_dir,
        mode="train",
    )
    val_dataset = DeepfakeDataset(
        metadata_path=metadata_dir / "val.csv",
        root_dir=root_dir,
        mode="val",
    )
    test_dataset = DeepfakeDataset(
        metadata_path=metadata_dir / "test.csv",
        root_dir=root_dir,
        mode="test",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )

    logger.info(
        f"Data loaders: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Load settings
    settings = Settings.from_yaml(args.config)

    # Set seed for reproducibility
    set_seed(settings.config.seed)

    logger.info("Starting training pipeline")
    logger.info(f"Config: {args.config}")

    # Create model
    model_config = settings.config.model
    device = torch.device(args.device if args.device else settings.get_device())
    model = create_model(model_config, device)

    # Create data loaders
    train_loader, val_loader, _test_loader = create_data_loaders(settings.config)

    # Create loss function
    loss_fn = get_loss_fn(name="cross_entropy")

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings.config.training.learning_rate,
        weight_decay=settings.config.training.weight_decay,
    )

    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=5,
    )

    # Create callbacks
    checkpoint_dir = Path(settings.config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(
            patience=settings.config.training.early_stopping_patience,
            monitor="val_loss",
        ),
        ModelCheckpoint(
            save_dir=str(checkpoint_dir),
            monitor="val_loss",
            save_best=True,
            save_last=True,
            optimizer=optimizer,
            scheduler=scheduler,
        ),
        LRScheduler(scheduler, monitor="val_loss"),
    ]

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        num_classes=settings.config.model.num_classes,
        callbacks=callbacks,
    )

    logger.info("Trainer initialized")
    logger.info(f"Model: {settings.config.model.name}")
    logger.info(f"Device: {device}")
    logger.info(f"Max epochs: {settings.config.training.epochs}")

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            logger.info(f"Resuming from checkpoint: {resume_path}")
            checkpoint = trainer.load_checkpoint(resume_path)
            start_epoch = checkpoint["epoch"] + 1
            logger.info(f"Resuming from epoch {start_epoch}")
        else:
            logger.error(f"Checkpoint not found: {resume_path}")
            return

    # Train
    history = trainer.train(
        num_epochs=settings.config.training.epochs,
        start_epoch=start_epoch,
    )

    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
