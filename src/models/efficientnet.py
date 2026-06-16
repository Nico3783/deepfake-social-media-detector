"""
EfficientNet model implementation.

Purpose: Secondary CNN architecture for deepfake detection.
Responsibilities: EfficientNet with compound scaling, transfer learning.
Dependencies: torch, torchvision

Research Traceability:
    Research Objective: Lightweight CNN architecture for deepfake detection
    Methodology: EfficientNet compound scaling (Tan & Le, 2019)
    Implementation: src/models/efficientnet.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from src.utils.logger import setup_logger
from src.utils.helpers import count_parameters

logger = setup_logger(__name__)


class EfficientNetModel(nn.Module):
    """EfficientNet for deepfake detection.

    Architecture:
    - MBConv1: Inverted residual with squeeze-and-excitation
    - MBConv6: Expanded inverted residual with SE attention
    - Head: Global average pooling + FC

    Supports EfficientNet-B0 through B7 variants.
    """

    def __init__(
        self,
        variant: str = "B0",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        freeze_base: bool = False,
    ) -> None:
        """Initialize EfficientNet.

        Args:
            variant: EfficientNet variant ('B0', 'B1', 'B2', ..., 'B7').
            num_classes: Number of output classes.
            pretrained: Use ImageNet pre-trained weights.
            dropout_rate: Dropout rate before final layer.
            freeze_base: Freeze base model parameters.
        """
        super().__init__()

        self.variant = variant
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Select model variant
        model_fn = self._get_model_fn(variant)

        # Load pre-trained EfficientNet
        if pretrained:
            self.base_model = model_fn(weights=f"EfficientNet_{variant}_Weights.IMAGENET1K_V1")
        else:
            self.base_model = model_fn(weights=None)

        # Get the number of features from the classifier
        num_features = self.base_model.classifier[1].in_features

        # Replace the classifier
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, num_classes),
        )

        # Freeze base model if specified
        if freeze_base:
            self._freeze_base()

        # Log model information
        params = count_parameters(self)
        logger.info(
            f"EfficientNet-{variant} initialized: {num_classes} classes, "
            f"pretrained={pretrained}, frozen={freeze_base}, "
            f"params={params['total']:,}"
        )

    def _get_model_fn(self, variant: str):
        """Get the model function for the specified variant.

        Args:
            variant: EfficientNet variant.

        Returns:
            Model function.

        Raises:
            ValueError: If variant is not supported.
        """
        variants = {
            "B0": models.efficientnet_b0,
            "B1": models.efficientnet_b1,
            "B2": models.efficientnet_b2,
            "B3": models.efficientnet_b3,
            "B4": models.efficientnet_b4,
            "B5": models.efficientnet_b5,
            "B6": models.efficientnet_b6,
            "B7": models.efficientnet_b7,
        }

        if variant not in variants:
            raise ValueError(
                f"Unsupported variant: {variant}. "
                f"Supported: {list(variants.keys())}"
            )

        return variants[variant]

    def _freeze_base(self) -> None:
        """Freeze base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze the classifier
        for param in self.base_model.classifier.parameters():
            param.requires_grad = True

        logger.info("Base model frozen, only classifier trainable")

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
            x: Input tensor with shape (N, 3, 224, 224) for B0.

        Returns:
            Output logits with shape (N, num_classes).
        """
        return self.base_model(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification layer.

        Args:
            x: Input tensor.

        Returns:
            Feature tensor.
        """
        # Remove the classifier
        feature_extractor = nn.Sequential(*list(self.base_model.children())[:-1])
        return feature_extractor(x)

    def count_parameters(self) -> dict[str, int]:
        """Count model parameters.

        Returns:
            Dictionary with total and trainable parameter counts.
        """
        return count_parameters(self)

    @property
    def input_size(self) -> int:
        """Get input size for the model variant."""
        sizes = {
            "B0": 224, "B1": 240, "B2": 260,
            "B3": 280, "B4": 300, "B5": 380,
            "B6": 456, "B7": 528,
        }
        return sizes[self.variant]
