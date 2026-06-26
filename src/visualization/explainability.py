"""
Explainability and interpretability visualization.

Purpose: Generate GradCAM heatmaps and attention visualizations for deepfake detection.
Responsibilities: Gradient-weighted class activation mapping, overlay generation.
Dependencies: torch, numpy, cv2, matplotlib

Research Traceability:
    Research Objective: Explainable deepfake detection (RQ5)
    Methodology: GradCAM for spatial localization of manipulation artifacts
    Implementation: src/visualization/explainability.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.utils.logger import setup_logger
from src.config.paths import PathConfig

logger = setup_logger(__name__)


class GradCAM:
    """Gradient-weighted Class Activation Mapping for CNN interpretability.

    Generates heatmaps showing which spatial regions most influenced
    the model's classification decision. Essential for explaining
    deepfake detection to non-technical stakeholders.

    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
    Networks via Gradient-based Localization", ICCV 2017.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | None = None,
    ) -> None:
        """Initialize GradCAM.

        Args:
            model: Trained PyTorch model.
            target_layer: Convolutional layer for activation extraction.
                          If None, uses the last conv layer in the model.
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or self._find_last_conv()
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self._register_hooks()

    def _find_last_conv(self) -> nn.Module:
        """Find the last convolutional layer in the model.

        Returns:
            Last Conv2d or Conv3d layer.

        Raises:
            RuntimeError: If no conv layer found.
        """
        conv_layers = []
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                conv_layers.append(module)

        if not conv_layers:
            raise RuntimeError("No convolutional layer found in model")
        return conv_layers[-1]

    def _register_hooks(self) -> None:
        """Register forward and backward hooks for activation/gradient capture."""
        def forward_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
            self.activations = output.detach()

        def backward_hook(module: nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor) -> None:
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """Generate GradCAM heatmap.

        Args:
            input_tensor: Input image tensor (1, C, H, W).
            target_class: Class index for heatmap. If None, uses predicted class.

        Returns:
            Heatmap as uint8 numpy array (H, W) with values in [0, 255].
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass from target class
        output[0, target_class].backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradient or activation not captured. Check target_layer.")

        # Pool gradients over spatial dimensions (channel-wise importance)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max

        # Convert to numpy uint8 heatmap
        heatmap = cam.squeeze().cpu().numpy()
        heatmap = (heatmap * 255).astype(np.uint8)
        return heatmap

    def overlay(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Overlay heatmap on original image.

        Args:
            image: Original image as numpy array (H, W, 3) in uint8 or float [0,1].
            heatmap: GradCAM heatmap (H, W) in uint8 [0, 255].
            alpha: Blending factor (0=original, 1=heatmap only).

        Returns:
            Blended image as uint8 numpy array (H, W, 3).
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Resize heatmap to match image dimensions
        h, w = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Blend
        blended = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        return blended


class ExplainabilityVisualizer:
    """Generate and save GradCAM visualizations for thesis documentation.

    Provides batch visualization, side-by-side comparisons, and
    publication-quality figure generation.
    """

    def __init__(self, output_dir: str | Path | None = None) -> None:
        """Initialize visualizer.

        Args:
            output_dir: Directory for saved figures. Uses PathConfig.OUTPUT_REPORTS if None.
        """
        if output_dir is None:
            paths = PathConfig()
            self.output_dir = paths.output_reports / "figures" / "explainability"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_single(
        self,
        gradcam: GradCAM,
        image: np.ndarray,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
        title: str = "GradCAM",
        filename: str = "gradcam",
    ) -> Path:
        """Generate and save a single GradCAM visualization.

        Args:
            gradcam: GradCAM instance.
            image: Original image (H, W, 3).
            input_tensor: Preprocessed input tensor (1, C, H, W).
            target_class: Target class index.
            title: Figure title.
            filename: Output filename.

        Returns:
            Path to saved figure.
        """
        heatmap = gradcam.generate(input_tensor, target_class)
        blended = gradcam.overlay(image, heatmap)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Activation Map")
        axes[1].axis("off")

        axes[2].imshow(blended)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout()

        path = self.output_dir / f"{filename}.pdf"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        logger.info("GradCAM visualization saved: %s", path)
        return path

    def visualize_batch(
        self,
        gradcam: GradCAM,
        samples: list[dict[str, Any]],
        filename: str = "gradcam_batch",
    ) -> Path:
        """Generate batch GradCAM comparison figure.

        Args:
            gradcam: GradCAM instance.
            samples: List of dicts with keys: image, tensor, label, pred.
            filename: Output filename.

        Returns:
            Path to saved figure.
        """
        n = len(samples)
        fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
        if n == 1:
            axes = axes.reshape(1, -1)

        for i, sample in enumerate(samples):
            image = sample["image"]
            tensor = sample["tensor"]
            label = sample.get("label", "")
            pred = sample.get("pred", "")

            heatmap = gradcam.generate(tensor)
            blended = gradcam.overlay(image, heatmap)

            axes[i, 0].imshow(image)
            axes[i, 0].set_title(f"Original\nTrue: {label}")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(heatmap, cmap="jet")
            axes[i, 1].set_title("Activation Map")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(blended)
            axes[i, 2].set_title(f"Overlay\nPred: {pred}")
            axes[i, 2].axis("off")

        fig.suptitle("GradCAM Explainability Analysis", fontsize=16, fontweight="bold")
        fig.tight_layout()

        path = self.output_dir / f"{filename}.pdf"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        logger.info("Batch GradCAM saved: %s", path)
        return path

    def save_heatmap(
        self,
        heatmap: np.ndarray,
        filename: str = "heatmap",
    ) -> Path:
        """Save raw heatmap as image file.

        Args:
            heatmap: Heatmap numpy array (H, W) in uint8.
            filename: Output filename.

        Returns:
            Path to saved file.
        """
        path = self.output_dir / f"{filename}.png"
        cv2.imwrite(str(path), heatmap)
        logger.info("Heatmap saved: %s", path)
        return path
