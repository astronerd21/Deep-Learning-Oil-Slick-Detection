"""Unit tests for src.model.SARResNet."""

import pytest
import torch
import torch.nn as nn

from src.model import SARResNet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_batch(batch_size: int = 2, channels: int = 2, h: int = 64, w: int = 64):
    return torch.randn(batch_size, channels, h, w)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSARResNetInit:
    def test_default_instantiation(self):
        model = SARResNet(pretrained=False)
        assert isinstance(model, SARResNet)

    def test_unsupported_backbone_raises(self):
        with pytest.raises(ValueError, match="Unsupported backbone"):
            SARResNet(backbone="vgg16", pretrained=False)

    def test_resnet50_backbone(self):
        model = SARResNet(backbone="resnet50", pretrained=False)
        assert isinstance(model, SARResNet)

    def test_first_conv_has_two_in_channels(self):
        model = SARResNet(pretrained=False)
        assert model.net.conv1.in_channels == 2

    def test_first_conv_weight_shape(self):
        model = SARResNet(pretrained=False)
        # Expected: (out_channels=64, in_channels=2, kH=7, kW=7)
        assert model.net.conv1.weight.shape == torch.Size([64, 2, 7, 7])

    def test_fc_output_is_one(self):
        model = SARResNet(pretrained=False)
        assert model.net.fc.out_features == 1

    def test_pretrained_weight_initialisation(self):
        """Pre-trained model should not crash and first conv weights should be finite."""
        model = SARResNet(pretrained=True)
        assert torch.isfinite(model.net.conv1.weight).all()


class TestSARResNetForward:
    def test_output_shape(self):
        model = SARResNet(pretrained=False).eval()
        x = _random_batch(batch_size=4)
        with torch.no_grad():
            out = model(x)
        assert out.shape == torch.Size([4])

    def test_output_dtype(self):
        model = SARResNet(pretrained=False).eval()
        with torch.no_grad():
            out = model(_random_batch())
        assert out.dtype == torch.float32

    def test_sigmoid_range(self):
        """After sigmoid, all outputs should be in [0, 1]."""
        model = SARResNet(pretrained=False).eval()
        with torch.no_grad():
            probs = model(_random_batch(batch_size=8)).sigmoid()
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_single_sample(self):
        model = SARResNet(pretrained=False).eval()
        with torch.no_grad():
            out = model(_random_batch(batch_size=1))
        assert out.shape == torch.Size([1])

    def test_resnet50_forward(self):
        model = SARResNet(backbone="resnet50", pretrained=False).eval()
        with torch.no_grad():
            out = model(_random_batch(batch_size=2))
        assert out.shape == torch.Size([2])

    def test_gradients_flow(self):
        model = SARResNet(pretrained=False).train()
        x = _random_batch()
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        # All parameters that require grad should have a gradient
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_bce_with_logits_loss_compatible(self):
        """The raw logits must be compatible with BCEWithLogitsLoss."""
        model = SARResNet(pretrained=False).train()
        criterion = nn.BCEWithLogitsLoss()
        x = _random_batch(batch_size=4)
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0])
        logits = model(x)
        loss = criterion(logits, labels)
        assert loss.item() > 0
        loss.backward()


class TestSARResNetInChannels:
    def test_custom_in_channels(self):
        """Model should accept a custom number of input channels."""
        model = SARResNet(pretrained=False, in_channels=1)
        assert model.net.conv1.in_channels == 1
        with torch.no_grad():
            out = model(torch.randn(2, 1, 64, 64))
        assert out.shape == torch.Size([2])

    def test_three_channels(self):
        model = SARResNet(pretrained=False, in_channels=3)
        with torch.no_grad():
            out = model(torch.randn(2, 3, 64, 64))
        assert out.shape == torch.Size([2])
