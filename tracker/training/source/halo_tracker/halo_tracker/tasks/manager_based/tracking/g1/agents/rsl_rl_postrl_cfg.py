#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Runner cfgs for scene PostRL — multi-critic PPO fine-tune of the distilled student.

One shared actor + N per-motion-routed critics:
  actor   ← ``_scene_student`` (== the distilled student), warm-started from the distillation checkpoint
            (``--load_checkpoint_path``); its exploration std is reset to ``postrl_reset_std`` at load.
  critics ← ``_simba_models(768, 3)[1]`` (== each PreRL teacher critic), warm-started from the teacher registry
            (``--teacher_registry_yaml``) and routed per-env by motion.

Reusing both architectures keeps both strict warm-start loads valid. The two variants differ only by the actor's depth
branch.
"""

from __future__ import annotations

import os

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from .local_rsl_rl_cfg import MultiCriticPpoAlgorithmCfg
from .rsl_rl_distill_cfg import _scene_student
from .rsl_rl_ppo_cfg import _simba_models

# num_mini_batches scales with the TOTAL env count (envs/rank × ranks) to hold ~49k-transition minibatches summed over
# ranks (NUM_ENVS × WORLD_SIZE / 2048 minibatches of 24 steps). Under torchrun, WORLD_SIZE is set before this module is
# imported; NUM_ENVS must match --num_envs.
_POSTRL_NMB = max(1, round(
    int(os.environ.get("NUM_ENVS", "4096")) * max(1, int(os.environ.get("WORLD_SIZE", "1"))) / 2048
))
_CRITIC_CFG = _simba_models(width=768, blocks=3)[1]   # per-critic template — MultiCriticPPO clones it per teacher
_CRITIC_GROUPS = ["critic_state", "critic_temporal", "critic_height"]


@configclass
class MultiCriticPostRLRunnerCfg_Proprio(RslRlOnPolicyRunnerCfg):
    """Scene PostRL — blind (proprio-only) actor. Pairs with ``G1ScenePostRLEnvCfg_Proprio``."""

    seed = 42
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 500
    experiment_name = "motion_tracking_scene_postrl_proprio"
    empirical_normalization = False

    obs_groups = {"actor": ["policy_state", "policy_temporal"], "critic": _CRITIC_GROUPS}
    policy = None
    actor = _scene_student(init_std=0.2, depth=False)
    critic = _CRITIC_CFG

    algorithm = MultiCriticPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=_POSTRL_NMB,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )
    torch_compile_mode: str | None = "default"
    """``torch.compile`` mode for the shared actor (the routed critics stay uncompiled; ``None`` disables it)."""


@configclass
class MultiCriticPostRLRunnerCfg_ProprioDepth(MultiCriticPostRLRunnerCfg_Proprio):
    """Scene PostRL — proprio + torso-depth actor. Pairs with ``G1ScenePostRLEnvCfg_ProprioDepth``."""

    experiment_name = "motion_tracking_scene_postrl_proprio_depth"
    obs_groups = {"actor": ["policy_state", "policy_temporal", "student_depth"], "critic": _CRITIC_GROUPS}
    actor = _scene_student(init_std=0.2, depth=True)
