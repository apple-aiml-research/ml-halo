#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""PreRL runner cfg: the privileged scene teachers (upstream rsl_rl PPO + ``HaloCNNModel``).

``_simba_models`` is the single source of the teacher architecture: the distillation teacher and the PostRL critics
reuse it so their checkpoints strict-load.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from .local_rsl_rl_cfg import CNNModelCfg, GaussianDistributionCfg

# LayerNorm'd encoders for the two 2-D privileged obs groups: a temporal Conv1d over the 30-frame state / reference
# history and a height-map CNN (terrain + objects).
_TEMPORAL_CNN_CFG = {
    "output_channels": [64, 128, 128],
    "kernel_size": [(3, 1), (3, 1), (1, 1)],
    "stride": [(2, 1), (2, 1), (1, 1)],
    "padding": "none",
    "activation": "elu",
    "flatten": True,
    "norm": "layer",
}


_HEIGHT_CNN_CFG = {
    "output_channels": [32, 64, 128],
    "kernel_size": 3,
    "stride": 2,
    "padding": "zeros",
    "activation": "elu",
    "flatten": True,
    "norm": "layer",
}


def _simba_models(width: int = 768, blocks: int = 3, ortho: bool = True):
    """Build the ``(actor, critic)`` SimBa ``CNNModelCfg`` pair for the PreRL teacher.

    Actor and critic share the SAME architecture — a residual MLP trunk (embed → ``blocks`` pre-LN residual FFN
    blocks of ``width``, x4 expansion, GELU → final LayerNorm → head) over the LayerNorm'd conv encoders for the
    ``critic_temporal`` and ``critic_height`` groups — differing ONLY in the distribution head (actor Gaussian,
    critic none). ``obs_normalization=False`` so the teachers don't bake divergent running statistics into their
    checkpoints. ``hidden_dims`` is a required-field filler (the trunk width comes from ``residual_width``).
    """
    common = dict(
        hidden_dims=[width], activation="gelu", obs_normalization=False, orthogonal_init=ortho,
        mlp_arch="residual", residual_width=width, residual_blocks=blocks, residual_expansion=4,
        cnn_cfg={"critic_temporal": _TEMPORAL_CNN_CFG, "critic_height": _HEIGHT_CNN_CFG},
    )
    return (
        CNNModelCfg(**common, distribution_cfg=GaussianDistributionCfg(init_std=1.0, std_type="scalar")),
        CNNModelCfg(**common, distribution_cfg=None),
    )


@configclass
class MotionTrackingScenePPORunnerCfg_PreRL(RslRlOnPolicyRunnerCfg):
    """PreRL privileged teacher for the three scene cases (robot-only / push box / lab sofa).

    Adam + wide SimBa (768x3). Actor and critic consume the SAME privileged obs groups — a 1-D ``critic_state`` trunk
    input plus conv-encoded ``critic_temporal`` and ``critic_height`` — so the trained teacher is directly reusable as
    a distillation teacher and a PostRL critic warm-start.

    ``num_mini_batches=8`` gives ~24.6k-transition minibatches at 8192 envs (8192 × 24 steps / 8).
    """

    seed = 42
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 500
    experiment_name = "motion_tracking_scene_prerl"
    empirical_normalization = False

    obs_groups = {
        "actor": ["critic_state", "critic_temporal", "critic_height"],
        "critic": ["critic_state", "critic_temporal", "critic_height"],
    }
    # ``policy=None`` overrides the ``policy`` field older ``RslRlOnPolicyRunnerCfg`` versions declare; rsl_rl 5.x
    # reads ``actor`` / ``critic`` / ``obs_groups`` instead.
    policy = None
    actor, critic = _simba_models(width=768, blocks=3)
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    torch_compile_mode: str | None = "default"
    """``torch.compile`` mode for the actor and critic (upstream rsl_rl option; ``None`` disables it)."""
