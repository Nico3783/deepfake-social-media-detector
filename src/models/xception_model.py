"""
XceptionNet model variant with custom classifier.

Purpose: Extended XceptionNet implementation with configurable classifier heads.
Responsibilities: XceptionNet with custom classification layers for deepfake detection.
Dependencies: torch, torchvision

Research Traceability:
    Research Objective: CNN-based deepfake detection architecture
    Methodology: XceptionNet with depthwise separable convolutions (Chollet, 2017)
    Implementation: src/models/xception_model.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from src.utils.logger import setup_logger
from src.utils.helpers import count_parameters

logger = setup_logger(__name__)


class XceptionNetCustom(nn.Module):
    """XceptionNet with custom classifier head.

    Uses the XceptionNet backbone with a custom multi-layer
    classification head optimized for deepfake detection.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        hidden_dim: int = 512,
        freeze_backbone: bool = False,
    ) -> None:
        """Initialize XceptionNet with custom classifier.

        Args:
            num_classes: Number of output classes.
            pretrained: Use ImageNet pre-trained weights.
            dropout_rate: Dropout rate.
            hidden_dim: Hidden layer dimension.
            freeze_backbone: Freeze backbone parameters.
        """
        super().__init__()

        self.num_classes = num_classes

        # Load pre-trained XceptionNet
        if pretrained:
            self.backbone = models.xception(
                weights=models.Xception_Weights.IMAGENET1K_V1,
            )
        else:
            self.backbone = models.xception(weights=None)

        # Get feature dimension from the last layer
        num_features = self.backbone.fc.in_features

        # Replace final FC layer with custom head
        self.backbone.fc = nn.Sequential(
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
            "XceptionNetCustom: classes=%d, pretrained=%s, params=%s",
            num_classes, pretrained, params["total"],
        )

    def _freeze_backbone(self) -> None:
        """Freeze backbone parameters, keep classifier trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.fc.parameters():
            param.requires_grad = True

        logger.info("Backbone frozen, classifier trainable")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (N, 3, 299, 299).

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
        # Use all layers except the final FC
        feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        return feature_extractor(x)

    @property
    def input_size(self) -> int:
        """Get expected input size."""
        return 299
