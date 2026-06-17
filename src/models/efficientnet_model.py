"""
EfficientNet model variant with custom classifier.

Purpose: Extended EfficientNet implementation with configurable classifier heads.
Responsibilities: EfficientNet-B0 with custom classification layers.
Dependencies: torch, torchvision

Research Traceability:
    Research Objective: Lightweight CNN architecture for deepfake detection
    Methodology: EfficientNet-B0 with compound scaling (Tan & Le, 2019)
    Implementation: src/models/efficientnet_model.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from src.utils.logger import setup_logger
from src.utils.helpers import count_parameters

logger = setup_logger(__name__)


class EfficientNetB0Custom(nn.Module):
    """EfficientNet-B0 with custom classifier head.

    Uses the EfficientNet-B0 backbone with a custom multi-layer
    classification head for improved deepfake detection.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        hidden_dim: int = 256,
        freeze_backbone: bool = False,
    ) -> None:
        """Initialize EfficientNet-B0 with custom classifier.

        Args:
            num_classes: Number of output classes.
            pretrained: Use ImageNet pre-trained weights.
            dropout_rate: Dropout rate.
            hidden_dim: Hidden layer dimension.
            freeze_backbone: Freeze backbone parameters.
        """
        super().__init__()

        self.num_classes = num_classes

        # Load pre-trained EfficientNet-B0
        if pretrained:
            self.backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1,
            )
        else:
            self.backbone = models.efficientnet_b0(weights=None)

        # Get feature dimension from backbone
        num_features = self.backbone.classifier[1].in_features

        # Replace classifier with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(hidden_dim, num_classes),
        )

        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()

        # Log model info
        params = count_parameters(self)
        logger.info(
            "EfficientNetB0Custom: classes=%d, pretrained=%s, params=%s",
            num_classes, pretrained, params["total"],
        )

    def _freeze_backbone(self) -> None:
        """Freeze backbone parameters, keep classifier trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

        logger.info("Backbone frozen, classifier trainable")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (N, 3, 224, 224).

        Returns:
            Classification logits (N, num_classes).
        """
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier.

        Args:
            x: Input tensor.

        Returns:
            Feature tensor.
        """
        # Use all layers except the classifier
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        return torch.flatten(features, 1)

    @property
    def input_size(self) -> int:
        """Get expected input size."""
        return 224
