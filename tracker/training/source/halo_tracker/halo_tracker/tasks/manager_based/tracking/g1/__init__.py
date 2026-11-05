#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""HALO motion-tracker tasks for the Unitree G1.

Three-stage pipeline:

1. ``Halo-MotionTracking-<Scene>-PreRL``: one privileged PPO teacher per scene case
   (``Robot_Only``; ``Robot_Object``: push box; ``Robot_Terrain``: lab sofa). Actor and critic both
   read the privileged height-map groups; the actor is distilled away downstream.
2. ``Halo-MotionTracking-Distillation-<Variant>``: the PreRL teachers -> ONE deployable student
   via on-policy multi-teacher distillation (per-motion routing from ``--teacher_registry_yaml``).
3. ``Halo-MotionTracking-PostRL-<Variant>``: multi-critic PPO fine-tune of the distilled student.
   The actor warm-starts from the student (``--load_checkpoint_path``), one critic per teacher
   warm-starts from that teacher's critic.

Student variants: ``-Proprio`` (proprioception only) and ``-ProprioDepth`` (+ torso depth camera).
"""

from pathlib import Path

import gymnasium as gym

from . import agents

# ``tracker/`` — robot and object assets live in the sibling ``<repo>/assets`` folder.
PROJ_DIR = Path(__file__).resolve().parents[8]
ASSETS_DIR = PROJ_DIR.parent / "assets"

_PRERL_TEACHERS = {  # gym suffix -> "env_module:EnvClass"
    "-Robot_Only": "motion_tracking_scene_env_cfg:G1FlatEnvCfg_PreRL_CNN",
    "-Robot_Object": "motion_tracking_scene_env_cfg:G1ObjectEnvCfg_PreRL_CNN",
    "-Robot_Terrain": "motion_tracking_scene_env_cfg:G1TerrainEnvCfg_PreRL_CNN",
}
for _sfx, _env_ep in _PRERL_TEACHERS.items():
    for _play in ("", "-Play"):
        gym.register(
            id=f"Halo-MotionTracking{_sfx}-PreRL{_play}",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.{_env_ep}{'_Play' if _play else ''}",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MotionTrackingScenePPORunnerCfg_PreRL",
            },
        )

_STUDENT_VARIANTS = ("Proprio", "ProprioDepth")
for _variant in _STUDENT_VARIANTS:
    for _play in ("", "-Play"):
        _env_sfx = "_Play" if _play else ""
        gym.register(
            id=f"Halo-MotionTracking-Distillation-{_variant}{_play}",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": (
                    f"{__name__}.motion_tracking_scene_distill_cfg:G1SceneDistillEnvCfg_{_variant}{_env_sfx}"
                ),
                "rsl_rl_cfg_entry_point": (
                    f"{agents.__name__}.rsl_rl_distill_cfg:MultiTeacherDistillationRunnerCfg_{_variant}"
                ),
            },
        )
        gym.register(
            id=f"Halo-MotionTracking-PostRL-{_variant}{_play}",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": (
                    f"{__name__}.motion_tracking_scene_postrl_cfg:G1ScenePostRLEnvCfg_{_variant}{_env_sfx}"
                ),
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_postrl_cfg:MultiCriticPostRLRunnerCfg_{_variant}",
            },
        )
