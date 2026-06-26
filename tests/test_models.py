"""Tests for model architecture modules."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.xception import XceptionNet
from src.models.efficientnet import EfficientNetModel
from src.models.model_factory import create_model
from src.config.settings import ModelConfig


class TestXceptionNet:
    """Tests for XceptionNet model."""

    def test_create_xception(self, device: torch.device) -> None:
        """XceptionNet initializes without error."""
        model = XceptionNet(num_classes=2, pretrained=False)
        model.to(device)
        assert isinstance(model, nn.Module)

    def test_forward_pass(self, device: torch.device, dummy_tensor: torch.Tensor) -> None:
        """XceptionNet produces correct output shape."""
        model = XceptionNet(num_classes=2, pretrained=False)
        model.to(device)
        model.eval()
        with torch.no_grad():
            output = model(dummy_tensor.to(device))
        assert output.shape == (1, 2)

    def test_output_is_logits(self, device: torch.device, dummy_tensor: torch.Tensor) -> None:
        """XceptionNet output contains raw logits (not probabilities)."""
        model = XceptionNet(num_classes=2, pretrained=False)
        model.to(device)
        model.eval()
        with torch.no_grad():
            output = model(dummy_tensor.to(device))
        assert output.dtype == torch.float32

    def test_batch_forward(self, device: torch.device) -> None:
        """XceptionNet handles batches correctly."""
        model = XceptionNet(num_classes=2, pretrained=False)
        model.to(device)
        model.eval()
        batch = torch.randn(4, 3, 299, 299).to(device)
        with torch.no_grad():
            output = model(batch)
        assert output.shape == (4, 2)

    def test_num_classes_parameter(self, device: torch.device) -> None:
        """XceptionNet respects num_classes parameter."""
        model = XceptionNet(num_classes=5, pretrained=False)
        model.to(device)
        model.eval()
        inp = torch.randn(1, 3, 299, 299).to(device)
        with torch.no_grad():
            output = model(inp)
        assert output.shape == (1, 5)


class TestEfficientNet:
    """Tests for EfficientNet model."""

    def test_create_efficientnet(self, device: torch.device) -> None:
        """EfficientNet initializes without error."""
        model = EfficientNetModel(variant="B0", num_classes=2, pretrained=False)
        model.to(device)
        assert isinstance(model, nn.Module)

    def test_forward_pass(self, device: torch.device) -> None:
        """EfficientNet produces correct output shape."""
        model = EfficientNetModel(variant="B0", num_classes=2, pretrained=False)
        model.to(device)
        model.eval()
        inp = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(inp)
        assert output.shape == (1, 2)


class TestModelFactory:
    """Tests for create_model factory function."""

    def test_create_xception(self) -> None:
        """Factory creates XceptionNet correctly."""
        config = ModelConfig(name="xception", num_classes=2, pretrained=False)
        model = create_model(config)
        assert isinstance(model, XceptionNet)

    def test_create_efficientnet(self) -> None:
        """Factory creates EfficientNet correctly."""
        config = ModelConfig(name="efficientnet_b0", num_classes=2, pretrained=False)
        model = create_model(config)
        assert isinstance(model, EfficientNetModel)

    def test_create_unknown_raises(self) -> None:
        """Factory raises ValueError for unknown model."""
        config = ModelConfig(name="unknown_model", num_classes=2)
        with pytest.raises(ValueError):
            create_model(config)

    def test_model_forward(self) -> None:
        """Factory-produced model supports forward pass."""
        config = ModelConfig(name="xception", num_classes=2, pretrained=False)
        model = create_model(config)
        model.eval()
        inp = torch.randn(1, 3, 299, 299)
        with torch.no_grad():
            output = model(inp)
        assert output.shape == (1, 2)
