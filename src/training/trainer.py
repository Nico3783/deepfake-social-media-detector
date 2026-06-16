"""
Training loop.

Purpose: Main training loop for deepfake detection models.
Responsibilities: Train, validate, save checkpoints, early stopping.
Dependencies: torch, pathlib

Research Traceability:
    Research Objective: Reproducible model training
    Methodology: Standard PyTorch training loop with callbacks
    Implementation: src/training/trainer.py
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.metrics import MetricsTracker
from src.training.callbacks import EarlyStopping, ModelCheckpoint, LRScheduler
from src.training.losses import get_loss_fn
from src.utils.logger import setup_logger
from src.utils.helpers import get_device

logger = setup_logger(__name__)


class Trainer:
    """Main training loop for deepfake detection models.

    Handles:
    - Training and validation loops
    - Metrics tracking
    - Early stopping
    - Model checkpointing
    - Learning rate scheduling
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None = None,
        loss_fn: nn.Module | None = None,
        device: torch.device | None = None,
        num_classes: int = 2,
        callbacks: list | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: Model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler.
            loss_fn: Loss function (default: CrossEntropyLoss).
            device: Training device.
            num_classes: Number of classes.
            callbacks: List of callbacks.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or get_device()
        self.num_classes = num_classes

        # Move model to device
        self.model = self.model.to(self.device)

        # Initialize loss function
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

        # Initialize metrics tracker
        self.train_metrics = MetricsTracker(num_classes=num_classes, device=self.device)
        self.val_metrics = MetricsTracker(num_classes=num_classes, device=self.device)

        # Initialize callbacks
        self.callbacks = callbacks or []

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        logger.info(f"Trainer initialized on {self.device}")

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Train for one epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        self.train_metrics.reset()

        running_loss = 0.0
        num_batches = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if hasattr(self.optimizer, "param_groups"):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update metrics
            preds = torch.argmax(outputs, dim=1)
            probs = torch.softmax(outputs, dim=1)
            self.train_metrics.update(preds, targets, probs)

            running_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        # Compute epoch metrics
        epoch_metrics = self.train_metrics.compute()
        epoch_metrics["loss"] = running_loss / num_batches

        return epoch_metrics

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        """Validate the model.

        Args:
            epoch: Current epoch number.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        self.val_metrics.reset()

        running_loss = 0.0
        num_batches = 0

        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)

            # Update metrics
            preds = torch.argmax(outputs, dim=1)
            probs = torch.softmax(outputs, dim=1)
            self.val_metrics.update(preds, targets, probs)

            running_loss += loss.item()
            num_batches += 1

        # Compute epoch metrics
        epoch_metrics = self.val_metrics.compute()
        epoch_metrics["loss"] = running_loss / num_batches

        return epoch_metrics

    def train(self, num_epochs: int = 50) -> dict[str, list]:
        """Run the full training loop.

        Args:
            num_epochs: Number of epochs to train.

        Returns:
            Training history dictionary.
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # Training
            train_metrics = self.train_epoch(epoch)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])

            # Validation
            val_metrics = self.validate(epoch)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])

            epoch_time = time.time() - epoch_start

            # Log epoch results
            logger.info(
                f"Epoch {epoch}/{num_epochs} ({epoch_time:.1f}s) - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )

            # Combine metrics for callbacks
            all_metrics = {
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }

            # Run callbacks
            continue_training = True
            for callback in self.callbacks:
                if hasattr(callback, "on_epoch_end"):
                    if not callback.on_epoch_end(epoch, all_metrics, self.model):
                        continue_training = False
                        break

            if not continue_training:
                logger.info(f"Training stopped at epoch {epoch}")
                break

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f}s")

        # Run end callbacks
        for callback in self.callbacks:
            if hasattr(callback, "on_train_end"):
                callback.on_train_end(self.model)

        return self.history

    def save_checkpoint(self, path: Path, epoch: int, metrics: dict) -> None:
        """Save training checkpoint.

        Args:
            path: Path to save checkpoint.
            epoch: Current epoch.
            metrics: Current metrics.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "history": self.history,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path) -> dict:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint.

        Returns:
            Checkpoint dictionary.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint
