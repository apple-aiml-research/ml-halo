#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Builds on rsl_rl (Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION, BSD-3-Clause).

"""Per-motion routing of envs to teachers / critics, shared by MultiTeacherDistillation and MultiCriticPPO.

The :class:`~halo_tracker.tasks.manager_based.tracking.g1.distillation.TeacherRegistry` maps every motion of the
env's motion loader to one of N task types (one PreRL teacher each). Each step, an env is routed to the task type of
the motion it currently plays.
"""

from __future__ import annotations

import torch
from collections import Counter
from typing import Any

from rsl_rl.env import VecEnv

from .utils import reach_command_manager


def load_registry(registry_path: str | None) -> Any:
    """Load the teacher registry YAML (task_type -> PreRL checkpoint + motion-routing rules)."""
    if not registry_path:
        raise ValueError("Routing needs cfg['algorithm']['teacher_registry_yaml'] (train.py --teacher_registry_yaml).")
    from halo_tracker.tasks.manager_based.tracking.g1.distillation import TeacherRegistry

    return TeacherRegistry.from_yaml(registry_path)


def get_motion_command(env: VecEnv) -> Any:
    """Return the env's ``"motion"`` command term."""
    return reach_command_manager(env).command_manager.get_term("motion")


def build_routing_table(registry: Any, motion_term: Any, device: str) -> torch.Tensor:
    """Build and validate the per-motion index table for the motions of ``motion_term``."""
    motion_files = getattr(motion_term.motion, "motion_files", None)
    if motion_files is None:
        raise AttributeError("The motion command's loader has no 'motion_files' attribute (use a scene motion task).")
    table = registry.build_motion_teacher_index(motion_files=motion_files, device=device).to(device, torch.long)
    n = len(registry.teacher_type_list)
    if n == 0:
        raise ValueError("The routing registry has an empty teacher_type_list.")
    bad = ((table < 0) | (table >= n)).nonzero(as_tuple=True)[0].tolist()
    if bad:
        raise ValueError(f"Routing table maps {len(bad)} motion(s) outside [0, {n}); first ids: {bad[:5]}.")
    counts = Counter(registry.classify_motion_file(p) for p in motion_files)
    print("[routing] motions per task: " + ", ".join(f"{t}={counts.get(t, 0)}" for t in registry.teacher_type_list))
    env_buckets = getattr(motion_term, "_env_compatible_motions", None)
    if env_buckets is not None:
        counts = registry.summarize_env_distribution(motion_files=motion_files, env_compatible_motions=env_buckets)
        print("[routing] envs per task: " + ", ".join(f"{t}={counts.get(t, 0)}" for t in registry.teacher_type_list))
    return table


def route(motion_term: Any, table: torch.Tensor) -> torch.Tensor:
    """Return the per-env teacher / critic index of the motion each env currently plays."""
    motion_ids = motion_term.motion.get_motion_id(motion_term.time_steps)
    return table[motion_ids]


def load_checkpoint(path: str, device: str) -> dict:
    """Load a checkpoint file with an error message naming the path."""
    try:
        return torch.load(path, weights_only=False, map_location=device)
    except Exception as e:
        raise RuntimeError(f"failed to load checkpoint {path!r}: {e}") from e
