#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Model / algorithm cfg classes for official ``rsl-rl-lib`` (>=5.5.1) + ``halo_tracker.rsl_rl_ext``.

``rsl_rl`` reads ``cfg["actor"]`` / ``cfg["critic"]`` (or ``cfg["student"]`` / ``cfg["teacher"]``) and
``cfg["algorithm"]`` as plain dicts and resolves ``class_name`` (a qualified ``"module:Class"`` name for the classes of
``halo_tracker.rsl_rl_ext``). The ``isaaclab_rl`` shipped with Isaac Lab does not declare model cfgs for the rsl_rl 5.x
schema, so the minimum fields are declared here.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

RSL_RL_EXT = "halo_tracker.rsl_rl_ext"


@configclass
class GaussianDistributionCfg:
    class_name: str = "GaussianDistribution"
    init_std: float = MISSING
    std_type: Literal["scalar", "log"] = "scalar"


@configclass
class CNNModelCfg:
    """``HaloCNNModel``: conv encoders (2-D images and temporal ``(B, T, F)`` groups) + MLP / SimBa trunk."""

    class_name: str = f"{RSL_RL_EXT}:HaloCNNModel"
    hidden_dims: list[int] = MISSING
    activation: str = MISSING
    obs_normalization: bool = False
    distribution_cfg: GaussianDistributionCfg | None = None
    cnn_cfg: dict = MISSING
    """One encoder cfg per 2-D / temporal obs group (``rsl_rl.modules.CNN`` arguments)."""

    orthogonal_init: bool = False
    """Orthogonal init (gain sqrt(2) on hidden layers, 0.01 on the output layer; zero biases)."""
    mlp_arch: Literal["plain", "residual"] = "plain"
    """Trunk: ``"plain"`` (upstream MLP over ``hidden_dims``) or ``"residual"`` (SimBa pre-LN residual blocks)."""
    residual_width: int | None = None
    """Residual-stream width when ``mlp_arch="residual"`` (defaults to ``hidden_dims[0]``)."""
    residual_blocks: int = 2
    """Number of residual blocks when ``mlp_arch="residual"``."""
    residual_expansion: int = 4
    """Inner-expansion factor of each residual block."""


@configclass
class MultiCriticPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """``MultiCriticPPO``: one shared actor + N per-motion routed critics, each warm-started from its PreRL
    teacher's ``critic_state_dict`` (``teacher_registry_yaml``); the actor warm-starts from the distilled student."""

    class_name: str = f"{RSL_RL_EXT}:MultiCriticPPO"
    teacher_registry_yaml: str = ""
    """Teacher registry YAML. Set by train.py from ``--teacher_registry_yaml``."""
    normalize_advantage_per_critic: bool = True
    """Normalize advantages within each critic's routed samples, so skills with different reward scales pull the
    shared policy equally."""
    postrl_reset_std: float = 0.2
    """Exploration std given to the actor when it is loaded from the (near-deterministic) distilled student."""
