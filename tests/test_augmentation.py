"""Tests for data augmentation pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.data.augmentation import AugmentationConfig, ImageAugmenter, get_augmenter_from_config


class TestAugmentationConfig:
    """Tests for AugmentationConfig dataclass."""

    def test_default_config(self):
        config = AugmentationConfig()
        assert config.horizontal_flip == 0.5
        assert config.rotation_range == 15
        assert config.enabled is True

    def test_custom_config(self):
        config = AugmentationConfig(
            horizontal_flip=0.8,
            rotation_range=30,
            brightness_range=(0.5, 1.5),
            enabled=False,
        )
        assert config.horizontal_flip == 0.8
        assert config.rotation_range == 30
        assert config.brightness_range == (0.5, 1.5)
        assert config.enabled is False


class TestImageAugmenter:
    """Tests for ImageAugmenter class."""

    def test_init_default(self):
        augmenter = ImageAugmenter()
        assert augmenter.config.enabled is True

    def test_init_with_config(self):
        config = AugmentationConfig(enabled=False)
        augmenter = ImageAugmenter(config)
        assert augmenter.config.enabled is False

    def test_from_dict(self):
        config_dict = {
            "horizontal_flip": 0.7,
            "rotation_range": 20,
            "brightness_range": [0.6, 1.4],
        }
        augmenter = ImageAugmenter.from_dict(config_dict)
        assert augmenter.config.horizontal_flip == 0.7
        assert augmenter.config.rotation_range == 20
        assert augmenter.config.brightness_range == (0.6, 1.4)

    def test_augment_disabled(self):
        config = AugmentationConfig(enabled=False)
        augmenter = ImageAugmenter(config)
        img = Image.fromarray(np.random.randint(0, 255, (299, 299, 3), dtype=np.uint8))
        result = augmenter.augment(img)
        assert result.size == img.size

    def test_augment_preserves_size(self):
        augmenter = ImageAugmenter(AugmentationConfig(enabled=True))
        img = Image.fromarray(np.random.randint(0, 255, (299, 299, 3), dtype=np.uint8))
        # Run multiple times to test randomness
        for _ in range(10):
            result = augmenter.augment(img)
            assert result.size == (299, 299)

    def test_augment_returns_pil_image(self):
        augmenter = ImageAugmenter(AugmentationConfig(enabled=True))
        img = Image.fromarray(np.random.randint(0, 255, (299, 299, 3), dtype=np.uint8))
        result = augmenter.augment(img)
        assert isinstance(result, Image.Image)

    def test_get_torchvision_transforms_disabled(self):
        augmenter = ImageAugmenter(AugmentationConfig(enabled=False))
        transforms = augmenter.get_torchvision_transforms()
        assert len(transforms.transforms) == 0

    def test_get_torchvision_transforms_enabled(self):
        augmenter = ImageAugmenter(AugmentationConfig(enabled=True, horizontal_flip=0.5))
        transforms = augmenter.get_torchvision_transforms()
        assert len(transforms.transforms) > 0


class TestGetAugmenterFromConfig:
    """Tests for get_augmenter_from_config factory."""

    def test_train_mode(self):
        config = {
            "augmentation": {
                "train": {"horizontal_flip": 0.8, "rotation_range": 20},
                "val": {"enabled": False},
            }
        }
        augmenter = get_augmenter_from_config(config, mode="train")
        assert augmenter.config.horizontal_flip == 0.8
        assert augmenter.config.enabled is True

    def test_val_mode(self):
        config = {
            "augmentation": {
                "train": {"horizontal_flip": 0.8},
                "val": {"enabled": False},
            }
        }
        augmenter = get_augmenter_from_config(config, mode="val")
        assert augmenter.config.enabled is False

    def test_test_mode(self):
        config = {"augmentation": {"test": {"enabled": False}}}
        augmenter = get_augmenter_from_config(config, mode="test")
        assert augmenter.config.enabled is False

    def test_empty_config(self):
        augmenter = get_augmenter_from_config({}, mode="train")
        assert augmenter.config.horizontal_flip == 0.5
