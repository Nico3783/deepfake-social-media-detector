"""
Model factory.

Purpose: Create model instances based on configuration.
Responsibilities: Instantiate models with proper settings.
Dependencies: torch

Research Traceability:
    Research Objective: Flexible model creation for experimentation
    Methodology: Factory pattern for model instantiation
    Implementation: src/models/model_factory.py
"""

from __future__ import annotations

import torch

from src.models.xception import XceptionNet
from src.models.efficientnet import EfficientNetModel
from src.config.settings import ModelConfig
from src.utils.logger import setup_logger
from src.utils.helpers import get_device

logger = setup_logger(__name__)


def create_model(
    config: ModelConfig,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Create a model instance based on configuration.

    Args:
        config: Model configuration.
        device: Target device.

    Returns:
        Initialized model.

    Raises:
        ValueError: If model name is not supported.
    """
    if device is None:
        device = get_device()

    model_name = config.name.lower()

    if model_name == "xception":
        model = XceptionNet(
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            dropout_rate=config.dropout_rate,
            freeze_base=config.freeze_backbone,
        )
    elif model_name.startswith("efficientnet"):
        variant = config.name.replace("efficientnet", "").replace("-", "").replace("_", "").upper()
        if not variant:
            variant = "B0"
        model = EfficientNetModel(
            variant=variant,
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            dropout_rate=config.dropout_rate,
            freeze_base=config.freeze_backbone,
        )
    else:
        raise ValueError(f"Unsupported model: {config.name}")

    model = model.to(device)
    logger.info(f"Created {config.name} model on {device}")

    return model


def load_model(
    checkpoint_path: str,
    model_name: str = "xception",
    num_classes: int = 2,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Load a model from a checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint.
        model_name: Name of the model architecture.
        num_classes: Number of output classes.
        device: Target device.

    Returns:
        Loaded model.
    """
    if device is None:
        device = get_device()

    # Create model
    config = ModelConfig(
        name=model_name,
        num_classes=num_classes,
        pretrained=False,
    )
    model = create_model(config, device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    logger.info(f"Loaded model from {checkpoint_path}")
    return model


def get_model_summary(model: torch.nn.Module) -> dict:
    """Get model summary information.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary with model information.
    """
    from src.utils.helpers import count_parameters

    params = count_parameters(model)

    summary = {
        "name": model.__class__.__name__,
        "total_parameters": params["total"],
        "trainable_parameters": params["trainable"],
        "device": next(model.parameters()).device,
    }

    return summary
