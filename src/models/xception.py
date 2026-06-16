"""
XceptionNet model implementation.

Purpose: Primary CNN architecture for deepfake detection.
Responsibilities: XceptionNet with depthwise separable convolutions, transfer learning.
Dependencies: torch, torchvision

Research Traceability:
    Research Objective: CNN-based deepfake detection architecture
    Methodology: XceptionNet with depthwise separable convolutions (Chollet, 2017)
    Implementation: src/models/xception.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from src.utils.logger import setup_logger
from src.utils.helpers import count_parameters

logger = setup_logger(__name__)


class XceptionNet(nn.Module):
    """XceptionNet for deepfake detection.

    Architecture:
    - Entry flow (3 blocks): Initial feature extraction
    - Middle flow (8 blocks): Feature refinement with depthwise separable convolutions
    - Exit flow (2 blocks): Classification

    Uses transfer learning from ImageNet pre-trained weights.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        freeze_base: bool = False,
    ) -> None:
        """Initialize XceptionNet.

        Args:
            num_classes: Number of output classes (default: 2 for binary).
            pretrained: Use ImageNet pre-trained weights.
            dropout_rate: Dropout rate before final layer.
            freeze_base: Freeze base model parameters.
        """
        super().__init__()

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Load pre-trained XceptionNet
        if pretrained:
            self.base_model = models.xception(weights=models.Xception_Weights.IMAGENET1K_V1)
        else:
            self.base_model = models.xception(weights=None)

        # Get the number of features from the last layer
        num_features = self.base_model.fc.in_features

        # Replace the final fully connected layer
        self.base_model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, num_classes),
        )

        # Freeze base model if specified
        if freeze_base:
            self._freeze_base()

        # Log model information
        params = count_parameters(self)
        logger.info(
            f"XceptionNet initialized: {num_classes} classes, "
            f"pretrained={pretrained}, frozen={freeze_base}, "
            f"params={params['total']:,}"
        )

    def _freeze_base(self) -> None:
        """Freeze base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze the final classification layer
        for param in self.base_model.fc.parameters():
            param.requires_grad = True

        logger.info("Base model frozen, only classification layer trainable")

    def unfreeze_from(self, layer_idx: int) -> None:
        """Unfreeze model from specified layer index.

        Args:
            layer_idx: Layer index to unfreeze from (negative indexing).
        """
        layers = list(self.base_model.children())
        unfreeze_from = max(0, len(layers) + layer_idx)

        for i, layer in enumerate(layers):
            for param in layer.parameters():
                param.requires_grad = i >= unfreeze_from

        trainable = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        logger.info(f"Unfroze from layer {unfreeze_from}, trainable params: {trainable:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor with shape (N, 3, 299, 299).

        Returns:
            Output logits with shape (N, num_classes).
        """
        return self.base_model(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification layer.

        Args:
            x: Input tensor with shape (N, 3, 299, 299).

        Returns:
            Feature tensor with shape (N, num_features).
        """
        # Remove the final FC layer
        feature_extractor = nn.Sequential(*list(self.base_model.children())[:-1])
        return feature_extractor(x)

    def count_parameters(self) -> dict[str, int]:
        """Count model parameters.

        Returns:
            Dictionary with total and trainable parameter counts.
        """
        return count_parameters(self)
