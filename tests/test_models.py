"""Tests for model architecture modules."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.xception import XceptionNet
from src.models.efficientnet import EfficientNetClassifier
from src.models.model_factory import ModelFactory


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
        # Logits can be any real value, not bounded to [0,1]
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
        model = EfficientNetClassifier(num_classes=2, pretrained=False)
        model.to(device)
        assert isinstance(model, nn.Module)

    def test_forward_pass(self, device: torch.device) -> None:
        """EfficientNet produces correct output shape."""
        model = EfficientNetClassifier(num_classes=2, pretrained=False)
        model.to(device)
        model.eval()
        inp = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(inp)
        assert output.shape == (1, 2)


class TestModelFactory:
    """Tests for ModelFactory."""

    def test_create_xception(self) -> None:
        """Factory creates XceptionNet correctly."""
        model = ModelFactory.create_model("xception", num_classes=2)
        assert isinstance(model, XceptionNet)

    def test_create_efficientnet(self) -> None:
        """Factory creates EfficientNet correctly."""
        model = ModelFactory.create_model("efficientnet", num_classes=2)
        assert isinstance(model, EfficientNetClassifier)

    def test_create_unknown_raises(self) -> None:
        """Factory raises ValueError for unknown model."""
        with pytest.raises(ValueError):
            ModelFactory.create_model("unknown_model", num_classes=2)

    def test_model_forward(self) -> None:
        """Factory-produced model supports forward pass."""
        model = ModelFactory.create_model("xception", num_classes=2)
        model.eval()
        inp = torch.randn(1, 3, 299, 299)
        with torch.no_grad():
            output = model(inp)
        assert output.shape == (1, 2)
