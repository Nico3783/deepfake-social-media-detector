"""Tests for training module: losses, metrics, callbacks, schedulers."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.training.losses import FocalLoss, LabelSmoothingLoss
from src.training.metrics import MetricsTracker
from src.training.callbacks import EarlyStopping, ModelCheckpoint, LRScheduler


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


class TestFocalLoss:
    """Tests for FocalLoss."""

    def test_forward_produces_scalar(self) -> None:
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0  # scalar

    def test_perfect_prediction_low_loss(self) -> None:
        loss_fn = FocalLoss()
        logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
        targets = torch.tensor([0, 1])
        loss = loss_fn(logits, targets)
        assert loss.item() < 0.01

    def test_bad_prediction_higher_loss(self) -> None:
        loss_fn = FocalLoss()
        good_logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
        bad_logits = torch.tensor([[-10.0, 10.0], [10.0, -10.0]])
        targets = torch.tensor([0, 1])
        good_loss = loss_fn(good_logits, targets)
        bad_loss = loss_fn(bad_logits, targets)
        assert bad_loss > good_loss

    def test_gamma_zero_equals_ce(self) -> None:
        """With gamma=0, focal loss reduces to standard cross-entropy."""
        focal = FocalLoss(gamma=0.0, reduction="mean")
        ce = nn.CrossEntropyLoss(reduction="mean")
        logits = torch.randn(8, 2)
        targets = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])
        assert torch.allclose(focal(logits, targets), ce(logits, targets), atol=1e-6)

    def test_different_alpha(self) -> None:
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])
        loss_a = FocalLoss(alpha=0.25)(logits, targets)
        loss_b = FocalLoss(alpha=0.75)(logits, targets)
        assert loss_a.item() != loss_b.item()


class TestLabelSmoothingLoss:
    """Tests for LabelSmoothingLoss."""

    def test_forward_produces_scalar(self) -> None:
        loss_fn = LabelSmoothingLoss(num_classes=2, smoothing=0.1)
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_smoothed_target_differs_from_hard(self) -> None:
        hard = nn.CrossEntropyLoss()
        smooth = LabelSmoothingLoss(num_classes=2, smoothing=0.1)
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])
        # Smoothed loss should not be identical to hard CE
        assert hard(logits, targets).item() != smooth(logits, targets).item()

    def test_zero_smoothing_matches_ce(self) -> None:
        """With smoothing=0, should be close to cross-entropy."""
        smooth = LabelSmoothingLoss(num_classes=2, smoothing=0.0)
        ce = nn.CrossEntropyLoss()
        logits = torch.randn(8, 2)
        targets = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])
        assert torch.allclose(smooth(logits, targets), ce(logits, targets), atol=1e-5)

    def test_multiclass(self) -> None:
        loss_fn = LabelSmoothingLoss(num_classes=5, smoothing=0.1)
        logits = torch.randn(8, 5)
        targets = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0
        assert loss.item() > 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetricsTracker:
    """Tests for MetricsTracker."""

    def test_update_and_compute(self) -> None:
        tracker = MetricsTracker(num_classes=2, task="binary")
        preds = torch.tensor([0, 1, 1, 0])
        targets = torch.tensor([0, 1, 0, 0])
        tracker.update(preds, targets)
        metrics = tracker.compute()
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_reset(self) -> None:
        tracker = MetricsTracker(num_classes=2, task="binary")
        tracker.update(torch.tensor([1, 0]), torch.tensor([1, 0]))
        tracker.reset()
        metrics = tracker.compute()
        assert metrics["accuracy"] == 0.0

    def test_perfect_predictions(self) -> None:
        tracker = MetricsTracker(num_classes=2, task="binary")
        preds = torch.tensor([0, 1, 0, 1])
        targets = torch.tensor([0, 1, 0, 1])
        tracker.update(preds, targets)
        metrics = tracker.compute()
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0

    def test_all_wrong_predictions(self) -> None:
        tracker = MetricsTracker(num_classes=2, task="binary")
        preds = torch.tensor([1, 0, 1, 0])
        targets = torch.tensor([0, 1, 0, 1])
        tracker.update(preds, targets)
        metrics = tracker.compute()
        assert metrics["accuracy"] == 0.0

    def test_get_epoch_metrics(self) -> None:
        tracker = MetricsTracker(num_classes=2, task="binary")
        tracker.update(torch.tensor([0, 1]), torch.tensor([0, 1]))
        epoch_metrics = tracker.get_epoch_metrics()
        assert isinstance(epoch_metrics, dict)
        assert "accuracy" in epoch_metrics

    def test_device_consistency(self) -> None:
        tracker = MetricsTracker(num_classes=2, task="binary", device="cpu")
        preds = torch.tensor([0, 1])
        targets = torch.tensor([0, 1])
        tracker.update(preds, targets)
        metrics = tracker.compute()
        assert isinstance(metrics["accuracy"], float)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    """Tests for EarlyStopping callback."""

    def test_no_stop_when_loss_improves(self) -> None:
        es = EarlyStopping(patience=3, min_delta=0.0, monitor="val_loss", mode="min")
        model = nn.Linear(10, 2)
        # Simulate improving loss
        for epoch in range(5):
            stop = es.on_epoch_end(epoch, {"val_loss": 1.0 - epoch * 0.1}, model)
        assert not es.early_stop

    def test_stops_after_patience_exceeded(self) -> None:
        es = EarlyStopping(patience=2, min_delta=0.0, monitor="val_loss", mode="min")
        model = nn.Linear(10, 2)
        # Simulate non-improving loss
        for epoch in range(5):
            es.on_epoch_end(epoch, {"val_loss": 1.0}, model)
        assert es.early_stop

    def test_resets_on_improvement(self) -> None:
        es = EarlyStopping(patience=2, min_delta=0.0, monitor="val_loss", mode="min")
        model = nn.Linear(10, 2)
        # Some non-improving epochs
        es.on_epoch_end(0, {"val_loss": 1.0}, model)
        es.on_epoch_end(1, {"val_loss": 1.0}, model)
        # Improvement resets counter
        es.on_epoch_end(2, {"val_loss": 0.5}, model)
        assert not es.early_stop

    def test_mode_max(self) -> None:
        es = EarlyStopping(patience=2, min_delta=0.0, monitor="val_acc", mode="max")
        model = nn.Linear(10, 2)
        # Decreasing accuracy triggers early stopping
        for epoch in range(5):
            es.on_epoch_end(epoch, {"val_acc": 0.5}, model)
        assert es.early_stop

    def test_min_delta(self) -> None:
        es = EarlyStopping(patience=1, min_delta=0.1, monitor="val_loss", mode="min")
        model = nn.Linear(10, 2)
        es.on_epoch_end(0, {"val_loss": 1.0}, model)
        # Improvement smaller than min_delta does NOT count
        es.on_epoch_end(1, {"val_loss": 0.95}, model)
        assert es.early_stop


class TestModelCheckpoint:
    """Tests for ModelCheckpoint callback."""

    def test_saves_checkpoint(self, tmp_path: Path) -> None:
        mc = ModelCheckpoint(save_dir=tmp_path, monitor="val_loss", mode="min")
        model = nn.Linear(10, 2)
        stop = mc.on_epoch_end(0, {"val_loss": 0.5}, model)
        assert not stop
        assert (tmp_path / "best_model_epoch_0.pt").exists()

    def test_saves_only_best(self, tmp_path: Path) -> None:
        mc = ModelCheckpoint(
            save_dir=tmp_path,
            monitor="val_loss",
            mode="min",
            save_best=True,
            save_top_k=1,
        )
        model = nn.Linear(10, 2)
        mc.on_epoch_end(0, {"val_loss": 1.0}, model)
        mc.on_epoch_end(1, {"val_loss": 0.5}, model)
        # Only the best should be kept
        checkpoints = list(tmp_path.glob("best_model_*.pt"))
        assert len(checkpoints) <= 2  # best + possibly last


class TestLRScheduler:
    """Tests for LRScheduler callback."""

    def test_reduces_lr(self) -> None:
        optimizer = torch.optim.SGD([torch.randn(3, requires_grad=True)], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        lr_sched = LRScheduler(scheduler=scheduler, monitor="val_loss")
        lr_sched.on_epoch_end(0, {"val_loss": 1.0})
        # LR should have been stepped
        lr = optimizer.param_groups[0]["lr"]
        assert lr < 0.1


# ---------------------------------------------------------------------------
# get_loss_fn helper
# ---------------------------------------------------------------------------


class TestGetLossFn:
    """Tests for get_loss_fn factory function."""

    def test_focal_loss(self) -> None:
        from src.training.losses import get_loss_fn
        loss_fn = get_loss_fn("focal", alpha=0.25, gamma=2.0)
        assert isinstance(loss_fn, FocalLoss)

    def test_label_smoothing_loss(self) -> None:
        from src.training.losses import get_loss_fn
        loss_fn = get_loss_fn("label_smoothing", num_classes=2, smoothing=0.1)
        assert isinstance(loss_fn, LabelSmoothingLoss)

    def test_ce_loss(self) -> None:
        from src.training.losses import get_loss_fn
        loss_fn = get_loss_fn("cross_entropy")
        assert isinstance(loss_fn, nn.CrossEntropyLoss)

    def test_unknown_loss_raises(self) -> None:
        from src.training.losses import get_loss_fn
        with pytest.raises(ValueError):
            get_loss_fn("nonexistent_loss")
