"""
Loss functions for deepfake detection.

Purpose: Custom and standard loss functions for model training.
Responsibilities: Cross-entropy, focal loss, label smoothing.
Dependencies: torch

Research Traceability:
    Research Objective: Robust training with class imbalance
    Methodology: Focal loss for handling imbalanced datasets
    Implementation: src/training/losses.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    Reduces loss for well-classified examples, focusing on hard examples.
    Reference: Lin et al., "Focal Loss for Dense Object Detection", 2017.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        """Initialize Focal Loss.

        Args:
            alpha: Balancing factor for positive class.
            gamma: Focusing parameter (higher = more focus on hard examples).
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Model outputs (logits) with shape (N, C).
            targets: Ground truth labels with shape (N,).

        Returns:
            Computed focal loss.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Cross-entropy loss with label smoothing.

    Prevents overconfidence by smoothing one-hot encoded labels.
    """

    def __init__(
        self,
        num_classes: int = 2,
        smoothing: float = 0.1,
        reduction: str = "mean",
    ) -> None:
        """Initialize Label Smoothing Loss.

        Args:
            num_classes: Number of classes.
            smoothing: Label smoothing factor (0 = no smoothing).
            reduction: Reduction method.
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label smoothing loss.

        Args:
            inputs: Model outputs (logits) with shape (N, C).
            targets: Ground truth labels with shape (N,).

        Returns:
            Computed loss.
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_loss_fn(
    name: str = "cross_entropy",
    **kwargs,
) -> nn.Module:
    """Get loss function by name.

    Args:
        name: Loss function name.
        **kwargs: Additional arguments for the loss function.

    Returns:
        Loss function instance.
    """
    loss_fns = {
        "cross_entropy": nn.CrossEntropyLoss,
        "focal": FocalLoss,
        "label_smoothing": LabelSmoothingLoss,
        "nll": nn.NLLLoss,
        "bce": nn.BCEWithLogitsLoss,
    }

    if name not in loss_fns:
        raise ValueError(f"Unsupported loss: {name}. Supported: {list(loss_fns.keys())}")

    return loss_fns[name](**kwargs)
