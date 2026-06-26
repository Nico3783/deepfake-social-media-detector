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

from src.utils.logger import setup_logger
from src.utils.helpers import count_parameters

logger = setup_logger(__name__)


class SeparableConv2d(nn.Module):
    """Depthwise separable convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class XBlock(nn.Module):
    """Xception block with residual connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = SeparableConv2d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = SeparableConv2d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class EntryFlow(nn.Module):
    """Entry flow: initial feature extraction."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = XBlock(64, 128, stride=2)
        self.block2 = XBlock(128, 256, stride=2)
        self.block3 = XBlock(256, 728, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class MiddleFlow(nn.Module):
    """Middle flow: 8 blocks of feature refinement."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[XBlock(728, 728) for _ in range(8)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ExitFlow(nn.Module):
    """Exit flow: classification."""

    def __init__(self, num_classes: int, dropout_rate: float = 0.5) -> None:
        super().__init__()
        self.block1 = XBlock(728, 1024, stride=2)
        self.block2 = SeparableConv2d(1024, 1536)
        self.bn2 = nn.BatchNorm2d(1536)
        self.relu = nn.ReLU(inplace=True)
        self.block3 = SeparableConv2d(1536, 2048)
        self.bn3 = nn.BatchNorm2d(2048)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.relu(self.bn2(self.block2(x)))
        x = self.relu(self.bn3(self.block3(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


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
            pretrained: Use ImageNet pre-trained weights (ignored for from-scratch impl).
            dropout_rate: Dropout rate before final layer.
            freeze_base: Freeze base model parameters.
        """
        super().__init__()

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.entry = EntryFlow()
        self.middle = MiddleFlow()
        self.exit = ExitFlow(num_classes, dropout_rate)

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
        for param in self.entry.parameters():
            param.requires_grad = False
        for param in self.middle.parameters():
            param.requires_grad = False

        # Unfreeze the exit classification layer
        for param in self.exit.fc.parameters():
            param.requires_grad = True

        logger.info("Base model frozen, only classification layer trainable")

    def unfreeze_from(self, layer_idx: int) -> None:
        """Unfreeze model from specified layer index.

        Args:
            layer_idx: Layer index to unfreeze from (negative indexing).
        """
        layers = list(self.children())
        unfreeze_from = max(0, len(layers) + layer_idx)

        for i, layer in enumerate(layers):
            for param in layer.parameters():
                param.requires_grad = i >= unfreeze_from

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Unfroze from layer {unfreeze_from}, trainable params: {trainable:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor with shape (N, 3, 299, 299).

        Returns:
            Output logits with shape (N, num_classes).
        """
        x = self.entry(x)
        x = self.middle(x)
        x = self.exit(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification layer.

        Args:
            x: Input tensor with shape (N, 3, 299, 299).

        Returns:
            Feature tensor with shape (N, num_features).
        """
        x = self.entry(x)
        x = self.middle(x)
        # Global average pooling before FC
        x = nn.AdaptiveAvgPool2d(1)(x)
        return x.view(x.size(0), -1)

    def count_parameters(self) -> dict[str, int]:
        """Count model parameters.

        Returns:
            Dictionary with total and trainable parameter counts.
        """
        return count_parameters(self)
