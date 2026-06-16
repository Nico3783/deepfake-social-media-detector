"""Tests for training pipeline modules."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from src.training.losses import FocalLoss, LabelSmoothingLoss
from src.training.metrics import MetricsTracker
from src.training.callbacks import EarlyStopping, LRScheduler


class TestFocalLoss:
    """Tests for Focal Loss."""

    def test_focal_loss_output_scalar(self) -> None:
        """Focal loss returns a scalar tensor."""
        criterion = FocalLoss()
        logits = torch.randn(4, 2)
        labels = torch.tensor([0, 1, 0, 1])
        loss = criterion(logits, labels)
        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_focal_loss_perfect_prediction(self) -> None:
        """Focal loss is low when predictions are correct."""
        criterion = FocalLoss()
        logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
        labels = torch.tensor([0, 1])
        loss = criterion(logits, labels)
        assert loss.item() < 0.1

    def test_focal_loss_bad_prediction(self) -> None:
        """Focal loss is high when predictions are wrong."""
        criterion = FocalLoss()
        logits = torch.tensor([[-10.0, 10.0], [10.0, -10.0]])
        labels = torch.tensor([0, 1])
        loss = criterion(logits, labels)
        assert loss.item() > 1.0

    def test_focal_loss_with_alpha(self) -> None:
        """Focal loss accepts alpha parameter."""
        criterion = FocalLoss(alpha=0.75)
        logits = torch.randn(4, 2)
        labels = torch.tensor([0, 1, 0, 1])
        loss = criterion(logits, labels)
        assert loss.item() >= 0.0


class TestLabelSmoothingLoss:
    """Tests for Label Smoothing Loss."""

    def test_label_smoothing_loss(self) -> None:
        """Label smoothing loss returns scalar."""
        criterion = LabelSmoothingLoss()
        logits = torch.randn(4, 2)
        labels = torch.tensor([0, 1, 0, 1])
        loss = criterion(logits, labels)
        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_label_smoothing_smooths(self) -> None:
        """Label smoothing loss is lower than hard cross-entropy for overconfident predictions."""
        hard_criterion = nn.CrossEntropyLoss()
        smooth_criterion = LabelSmoothingLoss()
        logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
        labels = torch.tensor([0, 1])
        hard_loss = hard_criterion(logits, labels)
        smooth_loss = smooth_criterion(logits, labels)
        # Smoothed loss should be slightly lower due to regularization
        assert smooth_loss.item() <= hard_loss.item()


class TestMetricsTracker:
    """Tests for MetricsTracker."""

    def test_initialization(self) -> None:
        """MetricsTracker initializes with empty history."""
        tracker = MetricsTracker()
        assert tracker.history == {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def test_update(self) -> None:
        """MetricsTracker records values correctly."""
        tracker = MetricsTracker()
        tracker.update({"train_loss": 0.5, "train_acc": 0.8})
        assert len(tracker.history["train_loss"]) == 1
        assert tracker.history["train_loss"][0] == 0.5

    def test_get_best_metric(self) -> None:
        """MetricsTracker finds best metric value."""
        tracker = MetricsTracker()
        tracker.update({"val_loss": 0.5})
        tracker.update({"val_loss": 0.3})
        tracker.update({"val_loss": 0.4})
        best = tracker.get_best("val_loss", mode="min")
        assert best == 0.3

    def test_get_best_accuracy(self) -> None:
        """MetricsTracker finds best accuracy (max mode)."""
        tracker = MetricsTracker()
        tracker.update({"val_acc": 0.7})
        tracker.update({"val_acc": 0.9})
        tracker.update({"val_acc": 0.8})
        best = tracker.get_best("val_acc", mode="max")
        assert best == 0.9


class TestEarlyStopping:
    """Tests for EarlyStopping callback."""

    def test_no_stop_within_patience(self) -> None:
        """EarlyStopping does not trigger within patience window."""
        es = EarlyStopping(patience=3, min_delta=0.01)
        # Simulate improving loss
        es.step(1.0)
        es.step(0.9)
        es.step(0.8)
        assert not es.should_stop

    def test_stop_after_patience(self) -> None:
        """EarlyStopping triggers after patience exhausted."""
        es = EarlyStopping(patience=3, min_delta=0.01)
        es.step(1.0)
        es.step(1.0)  # no improvement
        es.step(1.0)  # no improvement
        es.step(1.0)  # no improvement → should stop
        assert es.should_stop

    def test_reset_on_improvement(self) -> None:
        """EarlyStopping patience resets on improvement."""
        es = EarlyStopping(patience=3, min_delta=0.01)
        es.step(1.0)
        es.step(1.0)
        es.step(0.5)  # improvement → reset counter
        es.step(0.5)
        es.step(0.5)
        assert not es.should_stop  # only 3 steps since last improvement
