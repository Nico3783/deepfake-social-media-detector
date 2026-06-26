"""
Training callbacks.

Purpose: Implement training callbacks for checkpointing, early stopping, etc.
Responsibilities: Save checkpoints, early stopping, learning rate scheduling.
Dependencies: torch, pathlib

Research Traceability:
    Research Objective: Robust training with automatic stopping
    Methodology: Early stopping and model checkpointing
    Implementation: src/training/callbacks.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class Callback:
    """Base class for training callbacks."""

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], model: nn.Module) -> bool:
        """Called at the end of each epoch.

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of computed metrics.
            model: The model being trained.

        Returns:
            True if training should continue, False to stop.
        """
        return True

    def on_train_end(self, model: nn.Module) -> None:
        """Called at the end of training.

        Args:
            model: The trained model.
        """
        pass


class EarlyStopping(Callback):
    """Early stopping callback.

    Stops training when validation loss stops improving.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        monitor: str = "val_loss",
        mode: str = "min",
    ) -> None:
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change to qualify as improvement.
            monitor: Metric to monitor.
            mode: 'min' for loss, 'max' for accuracy.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], model: nn.Module) -> bool:
        """Check if training should stop.

        Args:
            epoch: Current epoch.
            metrics: Current metrics.
            model: Current model.

        Returns:
            True to continue, False to stop.
        """
        score = metrics.get(self.monitor)

        if score is None:
            logger.warning(f"Metric '{self.monitor}' not found, continuing training")
            return True

        if self.best_score is None:
            self.best_score = score
            return True

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            logger.info(f"Improved {self.monitor}: {score:.4f}")
        else:
            self.counter += 1
            logger.info(
                f"No improvement in {self.monitor} for {self.counter}/{self.patience} epochs"
            )

            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                self.early_stop = True
                return False

        return True


class ModelCheckpoint(Callback):
    """Model checkpoint callback.

    Saves model checkpoints based on validation performance.
    Supports full resume by saving optimizer and scheduler state when provided.
    """

    def __init__(
        self,
        save_dir: str | Path,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best: bool = True,
        save_last: bool = True,
        save_top_k: int = 3,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
    ) -> None:
        """Initialize model checkpoint.

        Args:
            save_dir: Directory to save checkpoints.
            monitor: Metric to monitor.
            mode: 'min' or 'max'.
            save_best: Save the best model.
            save_last: Save the last model.
            save_top_k: Number of top models to keep.
            optimizer: Optimizer to save state for resume.
            scheduler: LR scheduler to save state for resume.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.save_last = save_last
        self.save_top_k = save_top_k
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.best_score = None
        self.checkpoints = []

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], model: nn.Module) -> bool:
        """Save checkpoint if improved.

        Args:
            epoch: Current epoch.
            metrics: Current metrics.
            model: Current model.

        Returns:
            True to continue training, False to stop.
        """
        score = metrics.get(self.monitor)

        if score is None:
            return not self.save_best

        improved = False

        # Save last model
        if self.save_last:
            last_path = self.save_dir / "last_model.pth"
            self._save_checkpoint(model, epoch, metrics, last_path)

        # Save best model
        if self.save_best:
            if self.best_score is None:
                self.best_score = score
                improved = True
            elif self.mode == "min" and score < self.best_score:
                self.best_score = score
                improved = True
            elif self.mode == "max" and score > self.best_score:
                self.best_score = score
                improved = True

            if improved:
                best_path = self.save_dir / f"best_model_epoch_{epoch}.pt"
                self._save_checkpoint(model, epoch, metrics, best_path)

        return False

    def _save_checkpoint(
        self,
        model: nn.Module,
        epoch: int,
        metrics: dict[str, float],
        path: Path,
    ) -> None:
        """Save model checkpoint with optional optimizer/scheduler state.

        Args:
            model: Model to save.
            epoch: Current epoch.
            metrics: Current metrics.
            path: Save path.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
        }

        # Save optimizer state for resume
        if self.optimizer is not None and hasattr(self.optimizer, "state_dict"):
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()

        # Save scheduler state for resume
        if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")


class LRScheduler(Callback):
    """Learning rate scheduler callback."""

    def __init__(
        self,
        scheduler: Any,
        monitor: str = "val_loss",
    ) -> None:
        """Initialize LR scheduler.

        Args:
            scheduler: PyTorch LR scheduler.
            monitor: Metric to monitor.
        """
        self.scheduler = scheduler
        self.monitor = monitor

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], model: nn.Module | None = None) -> bool:
        """Step the scheduler.

        Args:
            epoch: Current epoch.
            metrics: Current metrics.
            model: Current model.

        Returns:
            True to continue training.
        """
        if hasattr(self.scheduler, "step"):
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                score = metrics.get(self.monitor)
                if score is not None:
                    self.scheduler.step(score)
            else:
                self.scheduler.step()

        return True


class TensorBoardCallback(Callback):
    """TensorBoard logging callback.

    Logs training and validation metrics to TensorBoard.
    """

    def __init__(self, log_dir: str | Path = "outputs/logs") -> None:
        """Initialize TensorBoard callback.

        Args:
            log_dir: Directory for TensorBoard logs.
        """
        self.log_dir = Path(log_dir)
        self.writer = None

    def _ensure_writer(self) -> None:
        """Lazily initialize TensorBoard writer."""
        if self.writer is None:
            from torch.utils.tensorboard import SummaryWriter

            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], model: nn.Module) -> bool:
        """Log metrics to TensorBoard.

        Args:
            epoch: Current epoch.
            metrics: Current metrics.
            model: Current model.

        Returns:
            True to continue training.
        """
        self._ensure_writer()

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, epoch)

        # Log learning rate
        for param_group in model.parameters():
            if hasattr(param_group, "lr"):
                self.writer.add_scalar("learning_rate", param_group.lr, epoch)
                break

        self.writer.flush()
        return True

    def on_train_end(self, model: nn.Module) -> None:
        """Close TensorBoard writer.

        Args:
            model: The trained model.
        """
        if self.writer is not None:
            self.writer.close()


class MixedPrecisionManager:
    """Mixed precision training manager.

    Handles automatic mixed precision (AMP) for faster training.
    """

    def __init__(self, enabled: bool = True) -> None:
        """Initialize mixed precision manager.

        Args:
            enabled: Whether to enable AMP.
        """
        self.enabled = enabled
        self.scaler = torch.amp.GradScaler("cuda") if enabled else None

    def forward(self, model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """Run forward pass with optional AMP.

        Args:
            model: Model to run.
            inputs: Input tensor.

        Returns:
            Model output.
        """
        if self.enabled:
            with torch.amp.autocast("cuda"):
                return model(inputs)
        return model(inputs)

    def backward(self, loss: torch.Tensor) -> None:
        """Run backward pass with optional AMP scaling.

        Args:
            loss: Loss tensor.
        """
        if self.enabled and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Step optimizer with optional AMP unscaling.

        Args:
            optimizer: Optimizer to step.
        """
        if self.enabled and self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
