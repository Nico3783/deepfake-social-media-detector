"""
Image normalization.

Purpose: Normalize images for deep learning model input.
Responsibilities: Apply mean/std normalization, value scaling.
Dependencies: torch, numpy, PIL

Research Traceability:
    Research Objective: Standardized input normalization for CNN models
    Methodology: ImageNet mean/std normalization
    Implementation: src/preprocessing/normalizer.py
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImageNormalizer:
    """Normalize images for model input.

    Supports:
    - ImageNet normalization
    - Custom mean/std normalization
    - Value scaling to [0, 1] or [-1, 1]
    """

    def __init__(
        self,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        scale_mode: str = "minmax",
    ) -> None:
        """Initialize image normalizer.

        Args:
            mean: Mean values for normalization (default: ImageNet).
            std: Standard deviation values for normalization (default: ImageNet).
            scale_mode: Scaling mode ('minmax', 'imagenet', 'tanh').
        """
        self.mean = np.array(mean or IMAGENET_MEAN, dtype=np.float32)
        self.std = np.array(std or IMAGENET_STD, dtype=np.float32)
        self.scale_mode = scale_mode

    def normalize(self, image: Image.Image | np.ndarray) -> np.ndarray:
        """Normalize an image.

        Args:
            image: Input image (PIL Image or numpy array).

        Returns:
            Normalized image as numpy array with shape (C, H, W).
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert to float and scale to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Apply scaling mode
        if self.scale_mode == "imagenet":
            image = (image - self.mean) / self.std
        elif self.scale_mode == "tanh":
            image = image * 2.0 - 1.0
        # "minmax" keeps values in [0, 1]

        return image

    def normalize_batch(
        self,
        images: list[Image.Image | np.ndarray],
    ) -> np.ndarray:
        """Normalize a batch of images.

        Args:
            images: List of input images.

        Returns:
            Normalized batch as numpy array with shape (N, C, H, W).
        """
        normalized = [self.normalize(img) for img in images]
        return np.stack(normalized, axis=0)

    def to_tensor(self, image: Image.Image | np.ndarray) -> torch.Tensor:
        """Convert image to normalized torch Tensor.

        Args:
            image: Input image.

        Returns:
            Normalized tensor with shape (C, H, W).
        """
        normalized = self.normalize(image)
        return torch.from_numpy(normalized)

    def to_tensor_batch(
        self,
        images: list[Image.Image | np.ndarray],
    ) -> torch.Tensor:
        """Convert batch of images to normalized torch Tensor.

        Args:
            images: List of input images.

        Returns:
            Normalized tensor with shape (N, C, H, W).
        """
        normalized = self.normalize_batch(images)
        return torch.from_numpy(normalized)

    def denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """Denormalize a tensor back to image values.

        Args:
            tensor: Normalized tensor with shape (C, H, W) or (N, C, H, W).

        Returns:
            Denormalized image as numpy array with values in [0, 1].
        """
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.numpy()

        # Handle batch dimension
        if tensor.ndim == 4:
            # Denormalize each image in batch
            mean = self.mean.reshape(1, 3, 1, 1)
            std = self.std.reshape(1, 3, 1, 1)
        else:
            mean = self.mean.reshape(3, 1, 1)
            std = self.std.reshape(3, 1, 1)

        # Reverse normalization
        image = tensor * std + mean

        # Clip to valid range
        image = np.clip(image, 0, 1)

        return image
