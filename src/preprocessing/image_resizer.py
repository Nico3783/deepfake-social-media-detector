"""
Image resizing.

Purpose: Resize images to target dimensions for model input.
Responsibilities: Resize images with various interpolation methods.
Dependencies: PIL, opencv-python, numpy

Research Traceability:
    Research Objective: Standardized input dimensions for CNN models
    Methodology: Lanczos resampling for quality preservation
    Implementation: src/preprocessing/image_resizer.py
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ImageResizer:
    """Resize images to target dimensions.

    Supports:
    - Fixed size resizing
    - Aspect ratio preservation with padding
    - Various interpolation methods
    """

    def __init__(
        self,
        target_size: int = 299,
        interpolation: str = "lanczos",
    ) -> None:
        """Initialize image resizer.

        Args:
            target_size: Target size (square) for resized image.
            interpolation: Interpolation method ('lanczos', 'bilinear', 'bicubic').
        """
        self.target_size = target_size
        self.interpolation = self._get_interpolation(interpolation)

    def _get_interpolation(self, method: str) -> Image.Resampling:
        """Get PIL interpolation method.

        Args:
            method: Interpolation method name.

        Returns:
            PIL Resampling enum.
        """
        methods = {
            "lanczos": Image.Resampling.LANCZOS,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "nearest": Image.Resampling.NEAREST,
        }
        return methods.get(method, Image.Resampling.LANCZOS)

    def resize(self, image: Image.Image | np.ndarray) -> Image.Image:
        """Resize image to target size.

        Args:
            image: Input image (PIL Image or numpy array).

        Returns:
            Resized image as PIL Image.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        return image.resize(
            (self.target_size, self.target_size),
            self.interpolation,
        )

    def resize_with_padding(
        self,
        image: Image.Image | np.ndarray,
        pad_color: tuple[int, int, int] = (0, 0, 0),
    ) -> Image.Image:
        """Resize image preserving aspect ratio with padding.

        Args:
            image: Input image.
            pad_color: Padding color (R, G, B).

        Returns:
            Resized and padded image.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Calculate scaling factor
        width, height = image.size
        scale = min(self.target_size / width, self.target_size / height)

        # Resize
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = image.resize((new_width, new_height), self.interpolation)

        # Create padded image
        padded = Image.new("RGB", (self.target_size, self.target_size), pad_color)
        paste_x = (self.target_size - new_width) // 2
        paste_y = (self.target_size - new_height) // 2
        padded.paste(resized, (paste_x, paste_y))

        return padded

    def resize_batch(self, images: list[Image.Image | np.ndarray]) -> list[Image.Image]:
        """Resize a batch of images.

        Args:
            images: List of input images.

        Returns:
            List of resized images.
        """
        return [self.resize(img) for img in images]
