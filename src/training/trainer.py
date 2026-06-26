"""
Training loop.

Purpose: Main training loop for deepfake detection models.
Responsibilities: Train, validate, save checkpoints, early stopping, resume.
Dependencies: torch, pathlib, time

Research Traceability:
    Research Objective: Reproducible model training with resume capability
    Methodology: Standard PyTorch training loop with callbacks and checkpointing
    Implementation: src/training/trainer.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.metrics import MetricsTracker
from src.training.callbacks import EarlyStopping, ModelCheckpoint, LRScheduler
from src.training.losses import get_loss_fn
from src.utils.logger import setup_logger
from src.utils.helpers import get_device

logger = setup_logger(__name__)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted string like '2h 15m 30s' or '5m 12s' or '45s'.
    """
    if seconds < 0:
        return "unknown"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


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

    def train(self, num_epochs: int = 50, start_epoch: int = 1) -> dict[str, list]:
        """Run the full training loop.

        Args:
            num_epochs: Number of epochs to train.
            start_epoch: Epoch to start from (1-indexed). Used for resuming.

        Returns:
            Training history dictionary.
        """
        if start_epoch > 1:
            logger.info(
                f"Resuming training from epoch {start_epoch} to {num_epochs}"
            )
        else:
            logger.info(f"Starting training for {num_epochs} epochs")

        training_start_time = time.time()
        epoch_times: list[float] = []

        for epoch in range(start_epoch, num_epochs + 1):
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
            epoch_times.append(epoch_time)

            # Calculate ETA
            elapsed_total = time.time() - training_start_time
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            remaining_epochs = num_epochs - epoch
            eta_seconds = avg_epoch_time * remaining_epochs

            # Log epoch results with ETA
            logger.info(
                f"Epoch {epoch}/{num_epochs} ({format_time(epoch_time)}) - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Elapsed: {format_time(elapsed_total)}, "
                f"ETA: {format_time(eta_seconds)}"
            )

            # Combine metrics for callbacks
            all_metrics = {
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }

            # Run callbacks
            for callback in self.callbacks:
                if hasattr(callback, "on_epoch_end"):
                    callback.on_epoch_end(epoch, all_metrics, self.model)

            # Check early stopping
            early_stop = False
            for callback in self.callbacks:
                if isinstance(callback, EarlyStopping) and callback.early_stop:
                    early_stop = True
                    break

            if early_stop:
                logger.info(f"Training stopped at epoch {epoch}")
                break

        total_time = time.time() - training_start_time
        logger.info(f"Training completed in {format_time(total_time)}")

        # Run end callbacks
        for callback in self.callbacks:
            if hasattr(callback, "on_train_end"):
                callback.on_train_end(self.model)

        return self.history

    def save_checkpoint(self, path: Path, epoch: int, metrics: dict) -> None:
        """Save training checkpoint for resume.

        Saves model weights, optimizer state, scheduler state, epoch number,
        metrics, and full history so training can be resumed exactly.

        Args:
            path: Path to save checkpoint.
            epoch: Current epoch.
            metrics: Current metrics.
        """
        scheduler_state = None
        if self.scheduler is not None:
            if hasattr(self.scheduler, "state_dict"):
                scheduler_state = self.scheduler.state_dict()

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": scheduler_state,
            "metrics": metrics,
            "history": self.history,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path, load_optimizer: bool = True, load_scheduler: bool = True) -> dict:
        """Load training checkpoint for resume.

        Restores model weights, optimizer state, scheduler state, epoch number,
        and training history.

        Args:
            path: Path to checkpoint.
            load_optimizer: Whether to restore optimizer state.
            load_scheduler: Whether to restore scheduler state.

        Returns:
            Checkpoint dictionary with 'epoch', 'metrics', 'history' keys.
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Restore model weights
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Restore optimizer state
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Restored optimizer state")

        # Restore scheduler state
        if (
            load_scheduler
            and self.scheduler is not None
            and checkpoint.get("scheduler_state_dict") is not None
            and hasattr(self.scheduler, "load_state_dict")
        ):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.info("Restored scheduler state")

        # Restore history
        if "history" in checkpoint:
            self.history = checkpoint["history"]
            logger.info(
                f"Restored history: {len(self.history.get('train_loss', []))} epochs"
            )

        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
        return checkpoint
