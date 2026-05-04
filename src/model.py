"""ResNet-based binary classifier for 2-band SAR imagery."""

from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


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

        # ------------------------------------------------------------------
        # Adapt first conv: 3 → in_channels
        # ------------------------------------------------------------------
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
            # Initialise by averaging the 3-channel ImageNet weights across
            # the channel dimension and repeating for each new channel.
            with torch.no_grad():
                mean_weight = orig_conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight.copy_(mean_weight.expand(-1, in_channels, -1, -1))
                if orig_conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(orig_conv.bias)
        net.conv1 = new_conv

        # ------------------------------------------------------------------
        # Adapt final FC: → 1 (binary logit)
        # ------------------------------------------------------------------
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
