"""
Application settings loaded from YAML configuration files.

Purpose: Central configuration management for the deepfake detection system.
Responsibilities: Load, validate, and provide access to all configuration parameters.
Dependencies: pyyaml, python-dotenv, src.config.paths, src.config.constants

Research Traceability:
    Research Objective: Reproducible experiments
    Methodology: Hyperparameter configuration management
    Implementation: src/config/settings.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.config.constants import Constants
from src.config.paths import PathConfig


@dataclass
class TrainingConfig:
    """Training hyperparameters and configuration."""

    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    early_stopping_patience: int = 10
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = True
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    name: str = "xception"
    num_classes: int = 2
    pretrained: bool = True
    dropout_rate: float = 0.5
    freeze_backbone: bool = False
    unfreeze_after_epoch: int = 0


@dataclass
class DataConfig:
    """Dataset configuration."""

    dataset_name: str = "faceforensics"
    root_dir: str = ""
    image_size: int = 299
    channels: int = 3
    augmentation: bool = True
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15


@dataclass
class PreprocessingConfig:
    """Preprocessing pipeline configuration."""

    frame_sample_rate: int = 5
    face_detection_method: str = "retinaface"
    face_confidence_threshold: float = 0.9
    face_margin: float = 0.2
    target_size: int = 299
    normalize_mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class InferenceConfig:
    """Inference configuration."""

    model_path: str = ""
    device: str = "auto"
    threshold: float = 0.5
    aggregation_method: str = "mean"
    batch_size: int = 16


@dataclass
class ExperimentConfig:
    """Full experiment configuration combining all sub-configs."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    experiment_name: str = "default"
    seed: int = 42
    log_dir: str = "outputs/logs"
    checkpoint_dir: str = "outputs/models/checkpoints"


class Settings:
    """Application settings manager.

    Loads configuration from YAML files and environment variables.
    Provides a single source of truth for all configuration parameters.

    Usage:
        settings = Settings()
        settings.load_from_yaml("configs/training.yaml")
        print(settings.training.learning_rate)
    """

    def __init__(self) -> None:
        """Initialize settings with default values."""
        load_dotenv()
        self.paths = PathConfig()
        self.constants = Constants()
        self.config = ExperimentConfig()

    def load_from_yaml(self, config_path: str | Path) -> None:
        """Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Raises:
            FileNotFoundError: If the config file does not exist.
            yaml.YAMLError: If the YAML file is malformed.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            raw_config: dict[str, Any] = yaml.safe_load(f) or {}

        self._apply_config(raw_config)

    def _apply_config(self, raw_config: dict[str, Any]) -> None:
        """Apply raw configuration dictionary to the config dataclasses.

        Args:
            raw_config: Dictionary containing configuration values.
        """
        if "training" in raw_config:
            self._update_dataclass(self.config.training, raw_config["training"])

        if "model" in raw_config:
            self._update_dataclass(self.config.model, raw_config["model"])

        if "data" in raw_config:
            self._update_dataclass(self.config.data, raw_config["data"])

        if "preprocessing" in raw_config:
            self._update_dataclass(self.config.preprocessing, raw_config["preprocessing"])

        if "inference" in raw_config:
            self._update_dataclass(self.config.inference, raw_config["inference"])

        for key in ("experiment_name", "seed", "log_dir", "checkpoint_dir"):
            if key in raw_config:
                setattr(self.config, key, raw_config[key])

    def _update_dataclass(self, obj: Any, updates: dict[str, Any]) -> None:
        """Update a dataclass instance with values from a dictionary.

        Args:
            obj: Dataclass instance to update.
            updates: Dictionary of attribute values to apply.
        """
        for key, value in updates.items():
            if hasattr(obj, key):
                setattr(obj, key, value)

    def get_device(self) -> str:
        """Determine the compute device.

        Returns:
            Device string: 'cuda', 'mps', or 'cpu'.
        """
        import torch

        if self.config.inference.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.config.inference.device

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Settings":
        """Create Settings from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            Configured Settings instance.
        """
        settings = cls()
        settings.load_from_yaml(config_path)
        return settings

def load_settings() -> dict[str, Any]:
    """Load settings and return a flat dict for API convenience.

    Returns:
        Dictionary with merged config values and sensible defaults.
    """
    s = Settings()
    cfg = s.config
    return {
        "model_path": str(Path(cfg.checkpoint_dir) / "best_model.pth"),
        "model_type": cfg.model.name if hasattr(cfg.model, "name") else "xception",
        "frame_sample_rate": 5,
        "batch_size": cfg.training.batch_size,
        "learning_rate": cfg.training.learning_rate,
        "num_classes": cfg.model.num_classes,
        "experiment_name": cfg.experiment_name,
    }
