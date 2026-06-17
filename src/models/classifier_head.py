"""
Classification head module.

Purpose: Provide reusable classification heads for CNN backbones.
Responsibilities: Global average pooling, dropout, fully connected layers.
Dependencies: torch, torch.nn

Research Traceability:
    Research Objective: Flexible CNN classification architecture
    Methodology: Modular classification heads for transfer learning
    Implementation: src/models/classifier_head.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ClassifierHead(nn.Module):
    """Reusable classification head for CNN backbones.

    Architecture:
        Global Average Pooling → Dropout → Linear → ReLU → Dropout → Linear

    Can be used to replace the default classifier in any CNN backbone.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 2,
        hidden_features: int = 512,
        dropout_rate: float = 0.5,
        use_batch_norm: bool = False,
    ) -> None:
        """Initialize classifier head.

        Args:
            in_features: Number of input features from backbone.
            num_classes: Number of output classes.
            hidden_features: Number of hidden layer features.
            dropout_rate: Dropout rate for regularization.
            use_batch_norm: Whether to use batch normalization.
        """
        super().__init__()

        self.in_features = in_features
        self.num_classes = num_classes

        layers: list[nn.Module] = [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        ]

        if use_batch_norm:
            layers.extend([
                nn.Linear(in_features, hidden_features),
                nn.BatchNorm1d(hidden_features),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
            ])
        else:
            layers.extend([
                nn.Linear(in_features, hidden_features),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
            ])

        layers.append(nn.Linear(hidden_features, num_classes))

        self.classifier = nn.Sequential(*layers)

        logger.info(
            "ClassifierHead: in=%d, hidden=%d, out=%d, bn=%s",
            in_features, hidden_features, num_classes, use_batch_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Feature tensor from backbone.

        Returns:
            Classification logits.
        """
        return self.classifier(x)


class BinaryClassifierHead(nn.Module):
    """Binary classification head with sigmoid output.

    Outputs a single probability score for fake detection.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = 256,
        dropout_rate: float = 0.5,
    ) -> None:
        """Initialize binary classifier head.

        Args:
            in_features: Number of input features.
            hidden_features: Number of hidden layer features.
            dropout_rate: Dropout rate.
        """
        super().__init__()

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Feature tensor.

        Returns:
            Binary logit.
        """
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict probability with sigmoid.

        Args:
            x: Feature tensor.

        Returns:
            Probability score.
        """
        logit = self.forward(x)
        return torch.sigmoid(logit)
