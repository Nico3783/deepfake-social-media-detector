"""
Data augmentation pipeline for deepfake detection.

Purpose: Provide configurable data augmentation transforms for training.
Dependencies: torchvision, numpy, PIL

Research Traceability:
    Research Objective: Improve model generalization through data augmentation
    Methodology: Standard image augmentation techniques (flip, rotate, color jitter)
    Implementation: src/data/augmentation.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation parameters."""

    horizontal_flip: float = 0.5
    rotation_range: int = 15
    width_shift_range: float = 0.1
    height_shift_range: float = 0.1
    zoom_range: float = 0.1
    brightness_range: tuple[float, float] = (0.8, 1.2)
    contrast_range: tuple[float, float] = (0.8, 1.2)
    jpeg_quality_range: tuple[int, int] = (75, 100)
    enabled: bool = True


class ImageAugmenter:
    """Configurable image augmentation pipeline.

    Applies random transformations to images for training data augmentation.
    Supports both PIL-based and torchvision transforms.
    """

    def __init__(self, config: AugmentationConfig | None = None) -> None:
        """Initialize augmenter.

        Args:
            config: Augmentation configuration. Uses defaults if None.
        """
        self.config = config or AugmentationConfig()

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ImageAugmenter":
        """Create augmenter from a configuration dictionary.

        Args:
            config_dict: Dictionary with augmentation parameters.

        Returns:
            Configured ImageAugmenter instance.
        """
        config = AugmentationConfig(
            horizontal_flip=config_dict.get("horizontal_flip", 0.5),
            rotation_range=config_dict.get("rotation_range", 15),
            width_shift_range=config_dict.get("width_shift_range", 0.1),
            height_shift_range=config_dict.get("height_shift_range", 0.1),
            zoom_range=config_dict.get("zoom_range", 0.1),
            brightness_range=tuple(config_dict.get("brightness_range", [0.8, 1.2])),
            contrast_range=tuple(config_dict.get("contrast_range", [0.8, 1.2])),
            jpeg_quality_range=tuple(config_dict.get("jpeg_quality_range", [75, 100])),
            enabled=config_dict.get("enabled", True),
        )
        return cls(config)

    def augment(self, image: Image.Image) -> Image.Image:
        """Apply random augmentations to an image.

        Args:
            image: Input PIL Image.

        Returns:
            Augmented PIL Image.
        """
        if not self.config.enabled:
            return image

        image = self._random_horizontal_flip(image)
        image = self._random_rotation(image)
        image = self._random_shift(image)
        image = self._random_zoom(image)
        image = self._random_brightness(image)
        image = self._random_contrast(image)
        image = self._random_jpeg_quality(image)

        return image

    def _random_horizontal_flip(self, image: Image.Image) -> Image.Image:
        """Apply random horizontal flip."""
        if np.random.random() < self.config.horizontal_flip:
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image

    def _random_rotation(self, image: Image.Image) -> Image.Image:
        """Apply random rotation."""
        if self.config.rotation_range > 0:
            angle = np.random.uniform(-self.config.rotation_range, self.config.rotation_range)
            return image.rotate(angle, resample=Image.BILINEAR, expand=False)
        return image

    def _random_shift(self, image: Image.Image) -> Image.Image:
        """Apply random width and height shift."""
        if self.config.width_shift_range > 0 or self.config.height_shift_range > 0:
            width, height = image.size
            dx = int(width * np.random.uniform(-self.config.width_shift_range, self.config.width_shift_range))
            dy = int(height * np.random.uniform(-self.config.height_shift_range, self.config.height_shift_range))
            return image.transform(image.size, Image.AFFINE, (1, 0, dx, 0, 1, dy))
        return image

    def _random_zoom(self, image: Image.Image) -> Image.Image:
        """Apply random zoom."""
        if self.config.zoom_range > 0:
            zoom = np.random.uniform(1 - self.config.zoom_range, 1 + self.config.zoom_range)
            width, height = image.size
            new_width = int(width * zoom)
            new_height = int(height * zoom)
            image = image.resize((new_width, new_height), Image.BILINEAR)
            # Crop or pad back to original size
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            image = image.crop((left, top, left + width, top + height))
        return image

    def _random_brightness(self, image: Image.Image) -> Image.Image:
        """Apply random brightness adjustment."""
        if self.config.brightness_range != (1.0, 1.0):
            factor = np.random.uniform(self.config.brightness_range[0], self.config.brightness_range[1])
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(factor)
        return image

    def _random_contrast(self, image: Image.Image) -> Image.Image:
        """Apply random contrast adjustment."""
        if self.config.contrast_range != (1.0, 1.0):
            factor = np.random.uniform(self.config.contrast_range[0], self.config.contrast_range[1])
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(factor)
        return image

    def _random_jpeg_quality(self, image: Image.Image) -> Image.Image:
        """Apply random JPEG quality compression."""
        if self.config.jpeg_quality_range != (100, 100):
            import io

            quality = np.random.randint(self.config.jpeg_quality_range[0], self.config.jpeg_quality_range[1] + 1)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            return Image.open(buffer).convert("RGB")
        return image

    def get_torchvision_transforms(self) -> transforms.Compose:
        """Get torchvision transforms equivalent to this augmentation config.

        Returns:
            Composed torchvision transforms.
        """
        if not self.config.enabled:
            return transforms.Compose([])

        transform_list = []

        if self.config.horizontal_flip > 0:
            transform_list.append(transforms.RandomHorizontalFlip(p=self.config.horizontal_flip))

        if self.config.rotation_range > 0:
            transform_list.append(transforms.RandomRotation(degrees=self.config.rotation_range))

        if self.config.brightness_range != (1.0, 1.0) or self.config.contrast_range != (1.0, 1.0):
            transform_list.append(
                transforms.ColorJitter(
                    brightness=self.config.brightness_range,
                    contrast=self.config.contrast_range,
                )
            )

        return transforms.Compose(transform_list)


def get_augmenter_from_config(config: dict[str, Any], mode: str = "train") -> ImageAugmenter:
    """Create an augmenter from YAML config dict.

    Args:
        config: Training config dict (expected to have 'augmentation' key).
        mode: One of 'train', 'val', 'test'.

    Returns:
        Configured ImageAugmenter.
    """
    augmentation_config = config.get("augmentation", {})
    mode_config = augmentation_config.get(mode, {})

    if mode == "train":
        return ImageAugmenter.from_dict(mode_config)
    else:
        # No augmentation for val/test
        return ImageAugmenter(AugmentationConfig(enabled=False))
