#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Builds on rsl_rl (Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION, BSD-3-Clause).

"""rsl_rl CNN model with a SimBa residual trunk option, orthogonal init, and temporal CNN observation groups."""

from __future__ import annotations

import contextlib
import inspect
import io
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.models import CNNModel
from rsl_rl.models.cnn_model import _OnnxCNNModel
from rsl_rl.modules import CNN
from rsl_rl.modules.distribution import Distribution

from .modules import ResidualMLP, TemporalCNN, orthogonal_init_


class _TrunkMixin:
    """Replaces the upstream MLP trunk after construction according to the trunk options."""

    mlp: nn.Module
    distribution: Distribution | None

    def _build_trunk(
        self,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        activation: str,
        mlp_arch: str,
        residual_width: int | None,
        residual_blocks: int,
        residual_expansion: int,
        orthogonal_init: bool,
    ) -> None:
        if mlp_arch not in ("plain", "residual"):
            raise ValueError(f"Unknown mlp_arch '{mlp_arch}'. Expected 'plain' or 'residual'.")
        if mlp_arch == "residual":
            if self.distribution is not None and type(self.distribution).init_mlp_weights is not (
                Distribution.init_mlp_weights
            ):
                raise ValueError(
                    f"mlp_arch='residual' does not support {type(self.distribution).__name__}, whose head"
                    " initialization indexes the plain MLP layers."
                )
            self.mlp = ResidualMLP(
                self._get_latent_dim(),  # type: ignore[attr-defined]
                self.distribution.input_dim if self.distribution is not None else output_dim,
                width=residual_width if residual_width is not None else hidden_dims[0],
                num_blocks=residual_blocks,
                expansion=residual_expansion,
                activation=activation,
                orthogonal_init=orthogonal_init,
            )
        elif orthogonal_init:
            linears = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)]
            orthogonal_init_(self.mlp, output_layer=linears[-1])
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)


class HaloCNNModel(_TrunkMixin, CNNModel):
    """:class:`rsl_rl.models.CNNModel` with the trunk options and temporal observation groups.

    Besides ``(B, C, H, W)`` images, an observation group of shape ``(B, T, F)`` is encoded by a
    :class:`~.modules.TemporalCNN` (kernels given as ``(k, 1)`` tuples).

    Extra args:
        mlp_arch: ``"plain"`` (upstream MLP) or ``"residual"`` (SimBa :class:`~.modules.ResidualMLP`).
        residual_width: Residual-stream width (defaults to ``hidden_dims[0]``).
        residual_blocks: Number of residual blocks.
        residual_expansion: Inner expansion factor of each residual block.
        orthogonal_init: Orthogonal init (sqrt(2) hidden, 0.01 output), zero biases; also applied to the
            conv encoders this model builds.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        cnn_cfg: dict[str, dict] | dict[str, Any] | None = None,
        cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
        mlp_arch: str = "plain",
        residual_width: int | None = None,
        residual_blocks: int = 2,
        residual_expansion: int = 4,
        orthogonal_init: bool = False,
    ) -> None:
        """Initialize the model; see the class docstring for the extra arguments."""
        owns_cnns = cnns is None
        if owns_cnns:
            cnns = self._build_cnns(obs, obs_groups, obs_set, cnn_cfg)
        # Given ``cnns``, upstream prints a "sharing encoders" message; it is wrong for the encoders built here.
        quiet = contextlib.redirect_stdout(io.StringIO()) if owns_cnns else contextlib.nullcontext()
        with quiet:
            super().__init__(
                obs,
                obs_groups,
                obs_set,
                output_dim,
                hidden_dims,
                activation,
                obs_normalization,
                distribution_cfg,
                cnn_cfg,
                cnns,
            )
        if not owns_cnns:
            print("Sharing CNN encoders between models, the CNN configurations of the receiving model are ignored.")
        self._build_trunk(
            output_dim,
            hidden_dims,
            activation,
            mlp_arch,
            residual_width,
            residual_blocks,
            residual_expansion,
            orthogonal_init,
        )
        # Shared encoders were already initialized by the model that owns them.
        if orthogonal_init and owns_cnns:
            orthogonal_init_(self.cnns, conv_only=True)

    def _build_cnns(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        cnn_cfg: dict[str, dict] | dict[str, Any] | None,
    ) -> nn.ModuleDict:
        """Build one encoder per 2D / temporal observation group (mirrors upstream ``CNNModel.__init__``)."""
        self._get_obs_dim(obs, obs_groups, obs_set)
        if cnn_cfg is None:
            raise ValueError("CNN configurations must be provided if CNNs are not shared.")
        if not all(isinstance(v, dict) for v in cnn_cfg.values()):
            cnn_cfg = {group: cnn_cfg for group in self.obs_groups_2d}
        if len(cnn_cfg) != len(self.obs_groups_2d):
            raise ValueError("The number of CNN configurations must match the number of 2D observation groups.")
        cnns = {}
        for idx, group in enumerate(self.obs_groups_2d):
            cnn_class = TemporalCNN if self.obs_2d_is_temporal[group] else CNN
            cnns[group] = cnn_class(
                input_dim=self.obs_dims_2d[idx], input_channels=self.obs_channels_2d[idx], **cnn_cfg[group]
            )
        return nn.ModuleDict(cnns)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _HaloOnnxCNNModel(self, verbose)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Select active observation groups; rank-3 ``(B, T, F)`` groups are temporal CNN inputs."""
        obs_groups_1d, obs_dim_1d = [], 0
        self.obs_groups_2d, self.obs_dims_2d, self.obs_channels_2d = [], [], []
        self.obs_2d_is_temporal = {}
        for group in obs_groups[obs_set]:
            shape = obs[group].shape
            if len(shape) == 4:  # B, C, H, W
                self.obs_groups_2d.append(group)
                self.obs_dims_2d.append(tuple(shape[2:4]))
                self.obs_channels_2d.append(shape[1])
                self.obs_2d_is_temporal[group] = False
            elif len(shape) == 3:  # B, T, F -> encoded as (B, F, T, 1)
                self.obs_groups_2d.append(group)
                self.obs_dims_2d.append((shape[1], 1))
                self.obs_channels_2d.append(shape[2])
                self.obs_2d_is_temporal[group] = True
            elif len(shape) == 2:  # B, C
                obs_groups_1d.append(group)
                obs_dim_1d += shape[-1]
            else:
                raise ValueError(f"Invalid observation shape for {group}: {shape}")
        if not self.obs_groups_2d:
            raise ValueError("No 2D observations are provided. If this is intentional, use the MLP model instead.")
        return obs_groups_1d, obs_dim_1d


class _HaloOnnxCNNModel(_OnnxCNNModel):
    """ONNX export wrapper whose dummy inputs keep the environment layout ``(1, T, F)`` for temporal groups."""

    def __init__(self, model: HaloCNNModel, verbose: bool = False) -> None:
        # rsl-rl-lib 5.5.1 takes (model, verbose); 5.5.2 dropped ``verbose``.
        if "verbose" in inspect.signature(_OnnxCNNModel.__init__).parameters:
            super().__init__(model, verbose)
        else:
            super().__init__(model)
        self.is_temporal = [model.obs_2d_is_temporal[g] for g in model.obs_groups_2d]

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative dummy inputs for ONNX tracing."""
        dummy_2d = []
        for (h, w), c, temporal in zip(self.obs_dims_2d, self.obs_channels_2d, self.is_temporal):
            dummy_2d.append(torch.zeros(1, h, c) if temporal else torch.zeros(1, c, h, w))
        return (torch.zeros(1, self.obs_dim_1d), *dummy_2d)
