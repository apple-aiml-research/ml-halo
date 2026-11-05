#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CLI helpers shared by ``train.py`` and ``play.py``."""

from __future__ import annotations

import argparse
import os
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

# The extension in ``halo_tracker.rsl_rl_ext`` subclasses rsl_rl internals of this release line.
RSL_RL_MIN_VERSION = "5.5.1"
RSL_RL_MAX_VERSION = "5.6"


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )


def mute_isaaclab_warnings() -> None:
    """Silence the Python warnings and the IsaacLab / Omniverse log noise (call after the app launches)."""
    import warnings
    for category in (DeprecationWarning, UserWarning, FutureWarning, ResourceWarning, RuntimeWarning):
        warnings.filterwarnings("ignore", category=category)

    # The URDF importer's "[prims.py] WARNING" goes through stdlib logging, not carb.
    import logging
    for name in ("prims", "isaaclab", "isaacsim", "omni", "pxr"):
        logging.getLogger(name).setLevel(logging.ERROR)

    try:
        import carb
        carb.settings.get_settings().set("/log/level", "Error")
    except Exception:
        pass  # carb unavailable (Isaac Sim not booted); the Python filters still apply


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlOnPolicyRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        # randomly sample a seed if seed = -1
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg


def check_rsl_rl_version() -> None:
    """Exit with an install hint unless ``rsl-rl-lib`` is in ``[RSL_RL_MIN_VERSION, RSL_RL_MAX_VERSION)``."""
    import importlib.metadata as metadata

    from packaging import version

    try:
        installed = metadata.version("rsl-rl-lib")
    except metadata.PackageNotFoundError:
        installed = None
    if installed is None or not (
        version.parse(RSL_RL_MIN_VERSION) <= version.parse(installed) < version.parse(RSL_RL_MAX_VERSION)
    ):
        print(
            f"[ERROR] The HALO tracker needs rsl-rl-lib>={RSL_RL_MIN_VERSION},<{RSL_RL_MAX_VERSION} "
            f"(installed: {installed}). Install it with:\n\n"
            f"\tpython -m pip install 'rsl-rl-lib>={RSL_RL_MIN_VERSION},<{RSL_RL_MAX_VERSION}'\n"
        )
        raise SystemExit(1)


def apply_motion_data_args(env_cfg, args_cli: argparse.Namespace) -> None:
    """Route ``--motion_data_folder`` / ``--motion_data`` to the env's motion command term."""
    motion = env_cfg.commands.motion
    if getattr(args_cli, "motion_data_folder", None) is not None:
        motion.motion_folder = args_cli.motion_data_folder
        motion.motion_file = []
        print(f"[INFO] Motion data folder(s): {args_cli.motion_data_folder}")
    elif getattr(args_cli, "motion_data", None) is not None:
        missing = [p for p in args_cli.motion_data if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(f"--motion_data files not found: {missing}")
        motion.motion_folder = None
        motion.motion_file = list(args_cli.motion_data)


def _algorithm_name(agent_cfg) -> str:
    """Bare class name of the configured algorithm (``"module:Class"`` -> ``"Class"``)."""
    return agent_cfg.algorithm.class_name.rsplit(":", 1)[-1].rsplit(".", 1)[-1]


def is_multi_teacher_distillation(agent_cfg) -> bool:
    return _algorithm_name(agent_cfg) == "MultiTeacherDistillation"


def apply_teacher_registry_arg(agent_cfg, args_cli: argparse.Namespace) -> None:
    """Forward ``--teacher_registry_yaml`` to the algorithms that route per-motion teachers / critics."""
    if _algorithm_name(agent_cfg) not in ("MultiTeacherDistillation", "MultiCriticPPO"):
        return
    if getattr(args_cli, "teacher_registry_yaml", None) is not None:
        agent_cfg.algorithm.teacher_registry_yaml = args_cli.teacher_registry_yaml
    if not getattr(agent_cfg.algorithm, "teacher_registry_yaml", ""):
        raise ValueError(
            f"{_algorithm_name(agent_cfg)} requires --teacher_registry_yaml: it maps each motion to its PreRL "
            "teacher checkpoint."
        )
