#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Builds on rsl_rl (Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION, BSD-3-Clause).

"""Network building blocks: the SimBa residual MLP trunk and the temporal (1-D) CNN encoder."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from functools import reduce

from rsl_rl.modules import CNN
from rsl_rl.utils import get_param, resolve_nn_activation


class ResidualMLP(nn.Module):
    """SimBa-style residual MLP trunk (Lee et al., "SimBa", ICLR'25).

    ``Linear(in -> width) -> N x [x + (LayerNorm -> Linear(width -> e*width) -> act -> Linear(-> width))]
    -> LayerNorm -> Linear(-> out)``. A drop-in replacement for :class:`rsl_rl.modules.MLP` that is TorchScript- and
    ONNX-exportable. The parameter names (``embed``, ``blocks.i.{0,1,3}``, ``final_norm``, ``head``) are part of the
    checkpoint format.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int, ...] | list[int],
        width: int,
        num_blocks: int,
        expansion: int = 4,
        activation: str = "gelu",
        orthogonal_init: bool = False,
    ) -> None:
        """Initialize the residual MLP.

        Args:
            input_dim: Dimension of the input latent.
            output_dim: Dimension (or shape) of the output.
            width: Width of the residual stream.
            num_blocks: Number of pre-LayerNorm residual blocks.
            expansion: Inner expansion factor of each block.
            activation: Activation function inside each block.
            orthogonal_init: Orthogonal init with gain sqrt(2) on hidden weights and 0.01 on the head, zero biases.
        """
        super().__init__()
        self.embed = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, expansion * width),
                resolve_nn_activation(activation),
                nn.Linear(expansion * width, width),
            )
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(width)
        if isinstance(output_dim, int):
            self.head = nn.Linear(width, output_dim)
            self.unflatten: nn.Module = nn.Identity()
        else:
            self.head = nn.Linear(width, reduce(lambda x, y: x * y, output_dim))
            self.unflatten = nn.Unflatten(dim=-1, unflattened_size=tuple(output_dim))
        if orthogonal_init:
            orthogonal_init_(self, output_layer=self.head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed, apply the residual blocks, normalize, and project to the output."""
        x = self.embed(x)
        for block in self.blocks:
            x = x + block(x)
        return self.unflatten(self.head(self.final_norm(x)))


class TemporalCNN(CNN):
    """CNN over a temporal observation group of shape ``(B, T, F)``, convolving along time only.

    The input is viewed as a ``(B, F, T, 1)`` image (features as channels). ``kernel_size``, ``stride``
    and ``dilation`` take a scalar or a ``(t, 1)`` tuple for all layers, or a list with one such entry per layer. The
    layers are laid out as in :class:`rsl_rl.modules.CNN` (conv, norm, activation, max-pool per layer; then global
    pool and flatten), so the state_dict keys match. The reshape is inside :meth:`forward` and thus in exported graphs.
    """

    def __init__(
        self,
        input_dim: tuple[int, int],
        input_channels: int,
        output_channels: tuple[int, ...] | list[int],
        kernel_size: int | tuple[int, int] | list,
        stride: int | tuple[int, int] | list = 1,
        dilation: int | tuple[int, int] | list = 1,
        padding: str = "none",
        norm: str | tuple[str] | list[str] = "none",
        activation: str = "elu",
        max_pool: bool | tuple[bool] | list[bool] = False,
        global_pool: str = "none",
        flatten: bool = True,
    ) -> None:
        """Build the layers; arguments as for :class:`rsl_rl.modules.CNN`, with ``input_dim=(T, 1)``."""
        nn.Sequential.__init__(self)  # CNN.__init__ computes output sizes for square kernels only
        length, layers, channels = input_dim[0], [], input_channels
        for idx, out in enumerate(output_channels):
            k, s, d = (_time_axis(_layer_value(v, idx)) for v in (kernel_size, stride, dilation))
            p = (
                math.ceil((s * math.floor(length / s) - length - s + d * (k - 1) + 1) / 2)
                if padding in _PADDINGS
                else 0
            )
            layers.append(
                nn.Conv2d(
                    channels,
                    out,
                    (k, 1),
                    (s, 1),
                    (p, 0),
                    (d, 1),
                    padding_mode=padding if padding in _PADDINGS else "zeros",
                )
            )
            length = (length + 2 * p - d * (k - 1) - 1) // s + 1
            n = get_param(norm, idx)
            if n == "batch":
                layers.append(nn.BatchNorm2d(out))
            elif n == "layer":
                layers.append(nn.LayerNorm([out, length, 1]))
            elif n != "none":
                raise ValueError(f"Unsupported normalization type: {n}. Supported types are 'none', 'batch', 'layer'.")
            layers.append(resolve_nn_activation(activation))
            if get_param(max_pool, idx):
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
                length = math.ceil(length / 2)
            channels = out
        if global_pool in ("max", "avg"):
            layers.append(nn.AdaptiveMaxPool2d((1, 1)) if global_pool == "max" else nn.AdaptiveAvgPool2d((1, 1)))
            length = 1
        elif global_pool != "none":
            raise ValueError(
                f"Unsupported global pooling type: {global_pool}. Supported types are 'none', 'max', 'avg'."
            )
        if flatten:
            layers.append(nn.Flatten(start_dim=1))
        self._output_channels = None if flatten else channels
        self._output_dim = channels * length if flatten else (length, 1)
        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape ``(B, T, F)`` to ``(B, F, T, 1)`` and run the convolution stack."""
        # No super() call: TorchScript cannot compile it.
        x = x.permute(0, 2, 1).unsqueeze(-1)
        for layer in self:
            x = layer(x)
        return x


_PADDINGS = ("zeros", "reflect", "replicate", "circular")


def _layer_value(value: int | tuple | list, idx: int) -> int | tuple:
    """Return the entry of layer ``idx``: a tuple of ints is one ``(t, 1)`` pair shared by all layers."""
    if isinstance(value, tuple) and all(isinstance(v, int) for v in value):
        return value
    return get_param(value, idx)


def _time_axis(value: int | tuple[int, int] | list[int]) -> int:
    """Return the time-axis entry of a temporal conv parameter (a scalar or a ``(t, 1)`` pair)."""
    if isinstance(value, (tuple, list)):
        if len(value) != 2 or value[1] != 1:
            raise ValueError(f"Temporal conv parameters act on time only; expected a scalar or (t, 1), got {value}.")
        return int(value[0])
    return int(value)


def orthogonal_init_(module: nn.Module, output_layer: nn.Module | None = None, conv_only: bool = False) -> None:
    """Orthogonally initialize ``Linear`` (or only ``Conv2d``) weights with gain sqrt(2) and zero the biases.

    ``output_layer`` (if given) gets gain 0.01, which keeps the initial action mean / value near zero.
    """
    types = (nn.Conv2d,) if conv_only else (nn.Linear,)
    for m in module.modules():
        if isinstance(m, types):
            nn.init.orthogonal_(m.weight, gain=0.01 if m is output_layer else 2.0**0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
