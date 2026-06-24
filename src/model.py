"""ResNet-based binary classifier for 2-band SAR imagery."""

from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY


class SARResNet(nn.Module):
    """Binary classifier built on a ResNet backbone adapted for SAR data.

    Standard ResNets expect 3-channel (RGB) input.  This module replaces the
    first convolutional layer so that it accepts exactly **2 input channels**
    (VV and VH SAR polarisations) while optionally re-using ImageNet
    pre-trained weights via mean-channel initialisation.

    The final fully-connected layer is also replaced with a single output
    neuron (raw logit) suitable for use with
    :class:`torch.nn.BCEWithLogitsLoss`.

    Args:
        backbone: ResNet variant to use.  Supported values: ``"resnet18"``
            (default), ``"resnet50"``.
        pretrained: If ``True``, load ImageNet pre-trained weights for all
            layers except the adapted first convolution.
        in_channels: Number of input channels.  Defaults to 2 (VV + VH).

    Example::

        model = SARResNet(backbone="resnet18", pretrained=True)
        logits = model(torch.randn(4, 2, 256, 256))   # (4,)
        probs  = logits.sigmoid()
    """

    _BACKBONES = {
        "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
        "resnet50": (resnet50, ResNet50_Weights.DEFAULT),
    }

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        in_channels: int = 2,
    ) -> None:
        super().__init__()

        if backbone not in self._BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. "
                f"Choose from: {list(self._BACKBONES)}."
            )

        factory, weights = self._BACKBONES[backbone]
        net = factory(weights=weights if pretrained else None)

        orig_conv: nn.Conv2d = net.conv1
        new_conv = nn.Conv2d(
            in_channels,
            orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=orig_conv.bias is not None,
        )
        if pretrained:

            with torch.no_grad():
                mean_weight = orig_conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight.copy_(mean_weight.expand(-1, in_channels, -1, -1))
                if orig_conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(orig_conv.bias)
        net.conv1 = new_conv

        in_features: int = net.fc.in_features
        net.fc = nn.Linear(in_features, 1)

        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(N, C, H, W)`` where ``C`` is the
                number of SAR channels (default 2: VV, VH).

        Returns:
            Raw logits of shape ``(N,)``.  Apply ``.sigmoid()`` to obtain
            probabilities, or pass directly to
            :class:`torch.nn.BCEWithLogitsLoss`.
        """
        return self.net(x).squeeze(1)


class TerraMindClassifier(nn.Module):
    """Binary classifier built on the TerraMind-1.0-base Foundation Model."""

    def __init__(self, freeze_backbone=True):
        super().__init__()

        self.backbone = TERRATORCH_BACKBONE_REGISTRY["terramind_v1_base"]()

        if freeze_backbone:
            print(
                "INFO: TerraMind Backbone is frozen. Training Classification Head only."
            )
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden_dim = 768

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        zeros = torch.zeros((B, 1, H, W), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([x, zeros], dim=1)

        features = self.backbone(x_padded)

        if isinstance(features, (list, tuple)):
            feat = features[-1]
        elif isinstance(features, dict):
            first_key = list(features.keys())[0]
            feat = features[first_key]
        else:
            feat = features

        if len(feat.shape) == 4:

            feat = feat.mean(dim=[2, 3])
        elif len(feat.shape) == 3:

            if feat.shape[2] == 768:
                feat = feat.mean(dim=1)
            elif feat.shape[1] == 768:
                feat = feat.mean(dim=2)

        logits = self.head(feat)
        return logits.squeeze(-1)
