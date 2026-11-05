#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Runner cfgs for multi-teacher distillation of the scene PreRL teachers.

Two student variants distilled from the SAME PreRL teachers:

  Proprio        : CNN over ``policy_temporal`` only (blind — no exteroception).
  ProprioDepth   : CNN over ``policy_temporal`` + ``student_depth``.

The ``teacher`` model cfg MUST equal the PreRL teacher architecture so each teacher checkpoint strict-loads, so it is
taken from ``_simba_models``. The student is a SimBa residual trunk (512x2, GELU) over smaller encoders (onboard
inference budget). ``_scene_student`` is reused as the PostRL actor so the distilled student strict-loads into it.

``obs_normalization=False`` everywhere: per-teacher running statistics would drift between teachers and complicate
per-env routing; each ``ObsTerm`` already clips its values.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from .local_rsl_rl_cfg import RSL_RL_EXT, CNNModelCfg, GaussianDistributionCfg
from .rsl_rl_ppo_cfg import _simba_models

# Student temporal encoder: narrower than the teacher's (32/64/128 vs 64/128/128 channels, no LayerNorm) over the
# (B, T=30, F) ``policy_temporal`` stream.
_STUDENT_TEMPORAL_CNN_CFG = {
    "output_channels": [32, 64, 128],
    "kernel_size": [(3, 1), (3, 1), (1, 1)],
    "stride": [(2, 1), (2, 1), (1, 1)],
    "padding": "none",
    "activation": "elu",
    "flatten": True,
}
# Student depth CNN — input (B, 1, 48, 84) from ``mdp.depth_image``. Four stride-2 3×3 unpadded convs:
#   (48, 84) → (23, 41) → (11, 20) → (5, 9) → (2, 4); channels 1 → 16 → 32 → 64 → 128 → flatten 1024 features.
# No padding on purpose: at 48×84 "same" zero padding gives a layer pads=(1, 0), which onnxruntime's CPU Conv
# evaluates wrongly, so the deployed ONNX would silently differ from the trained policy.
_STUDENT_DEPTH_CNN_CFG = {
    "output_channels": [16, 32, 64, 128],
    "kernel_size": 3,
    "stride": 2,
    "padding": "none",
    "activation": "elu",
    "flatten": True,
}


def _scene_student(init_std: float, depth: bool) -> CNNModelCfg:
    """The deployable student / PostRL actor: SimBa residual trunk (512x2, GELU, ortho) over the student encoders
    (``policy_temporal`` [+ ``student_depth`` when ``depth``]). ``policy_state`` feeds the trunk directly."""
    cnn_cfg = {"policy_temporal": _STUDENT_TEMPORAL_CNN_CFG}
    if depth:
        cnn_cfg["student_depth"] = _STUDENT_DEPTH_CNN_CFG
    return CNNModelCfg(
        hidden_dims=[512],
        activation="gelu",
        obs_normalization=False,
        orthogonal_init=True,
        mlp_arch="residual",
        residual_width=512,
        residual_blocks=2,
        residual_expansion=4,
        cnn_cfg=cnn_cfg,
        distribution_cfg=GaussianDistributionCfg(init_std=init_std, std_type="scalar"),
    )


@configclass
class MultiTeacherDistillationAlgorithmCfg:
    """``MultiTeacherDistillation`` (``halo_tracker.rsl_rl_ext``): upstream distillation args + registry + DAgger."""

    class_name: str = f"{RSL_RL_EXT}:MultiTeacherDistillation"

    num_learning_epochs: int = MISSING
    learning_rate: float = MISSING
    gradient_length: int = MISSING
    max_grad_norm: None | float = None
    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    loss_type: Literal["mse", "huber"] = "mse"

    teacher_registry_yaml: str = ""
    """Teacher registry YAML (task_type → PreRL checkpoint + routing rules). Set by train.py from
    ``--teacher_registry_yaml``."""

    dagger_teacher_ratio: float = 0.0
    """DAgger-BC rollout mixture α: per-env probability (redrawn on each reset) that the env rolls out with its
    TEACHER rather than the student. The BC target is the teacher's action either way; only the state distribution
    the student trains on changes. 0 = pure on-student DAgger, 1 = pure BC on teacher rollouts."""


# Teacher = the PreRL actor (``_simba_models`` returns (actor, critic)); it carries the Gaussian head saved as
# ``actor_state_dict`` → strict load.
_TEACHER_CFG, _ = _simba_models(width=768, blocks=3)


@configclass
class MultiTeacherDistillationRunnerCfg_Proprio(RslRlOnPolicyRunnerCfg):
    """Distillation runner — proprio-only (blind) student. Pairs with ``G1SceneDistillEnvCfg_Proprio``."""

    seed = 42
    num_steps_per_env = 32
    max_iterations = 30000
    save_interval = 500
    experiment_name = "motion_tracking_scene_distill_proprio"
    empirical_normalization = False

    obs_groups = {
        "student": ["policy_state", "policy_temporal"],
        "teacher": ["critic_state", "critic_temporal", "critic_height"],
    }
    policy = None
    # init_std=1e-6 keeps BC near-deterministic (the student output is its mean); PostRL resets the std.
    student = _scene_student(init_std=1e-6, depth=False)
    teacher = _TEACHER_CFG

    algorithm = MultiTeacherDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1e-4,
        gradient_length=8,
        max_grad_norm=1.0,
        # 20% of envs drive their episode with the teacher, 80% with the student.
        dagger_teacher_ratio=0.2,
    )
    torch_compile_mode: str | None = "default"
    """``torch.compile`` mode for the student (upstream rsl_rl option; ``None`` disables it)."""


@configclass
class MultiTeacherDistillationRunnerCfg_ProprioDepth(MultiTeacherDistillationRunnerCfg_Proprio):
    """Distillation runner — proprio + depth student. Pairs with ``G1SceneDistillEnvCfg_ProprioDepth``."""

    experiment_name = "motion_tracking_scene_distill_proprio_depth"

    obs_groups = {
        "student": ["policy_state", "policy_temporal", "student_depth"],
        "teacher": ["critic_state", "critic_temporal", "critic_height"],
    }

    student = _scene_student(init_std=1e-6, depth=True)
