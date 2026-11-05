#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Builds on rsl_rl (Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION, BSD-3-Clause).

"""On-policy multi-teacher distillation (DAgger).

One student is cloned from N frozen teachers. Every step, each env is routed to the teacher of the motion it currently
plays (see :mod:`.routing`); routing rather than averaging avoids blending two specialists into an action neither would
take. A DAgger-BC mixture lets a fraction of the envs roll out with their teacher instead of the student; the
behavior-cloning target is the teacher's action either way.
"""

from __future__ import annotations

import copy
import inspect
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.algorithms import Distillation
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import compile_model, resolve_class, resolve_obs_groups

from .routing import build_routing_table, get_motion_command, load_checkpoint, load_registry, route
from .utils import group_indices_by_id, warn_state_dict_mismatch, warn_unused_alg_kwargs


class MultiTeacherDistillation(Distillation):
    """Distillation from N frozen, per-motion routed teachers into one student.

    Args (beyond :class:`rsl_rl.algorithms.Distillation`):
        teachers: ``{task_type: teacher}``, frozen and kept in eval mode.
        teacher_type_list: Teacher order; index ``i`` is the teacher of ``teacher_type_list[i]``.
        motion_to_teacher_idx: Per-motion teacher index.
        motion_term: The env's motion command term, queried for each env's current motion.
        teacher_registry_manifest: Description of the routing registry, stored in checkpoints.
        dagger_teacher_ratio: Probability that the teacher, not the student, drives an env for a whole episode.
    """

    teachers: nn.ModuleDict
    teacher_type_list: list[str]

    def __init__(
        self,
        student: MLPModel,
        teachers: nn.ModuleDict,
        teacher_type_list: list[str],
        motion_to_teacher_idx: torch.Tensor,
        motion_term: Any,
        teacher_registry_manifest: dict,
        storage: RolloutStorage,
        dagger_teacher_ratio: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the algorithm; see the class docstring for the arguments."""
        if not teacher_type_list:
            raise ValueError("MultiTeacherDistillation needs at least one teacher, but teacher_type_list is empty.")
        if any(t.is_recurrent for t in teachers.values()):
            # A teacher runs on a varying subset of envs each step, which a per-env hidden state cannot follow.
            raise NotImplementedError("MultiTeacherDistillation does not support recurrent teachers.")
        accepted = inspect.signature(Distillation.__init__).parameters
        warn_unused_alg_kwargs(type(self).__name__, {k: v for k, v in kwargs.items() if k not in accepted})
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        super().__init__(student, teachers[teacher_type_list[0]], storage, **kwargs)
        self.teachers = teachers.to(self.device).eval().requires_grad_(False)
        self.teacher_type_list = list(teacher_type_list)
        self.motion_to_teacher_idx = motion_to_teacher_idx.to(self.device, dtype=torch.long)
        self.teacher_registry_manifest = dict(teacher_registry_manifest)
        self._motion_term = motion_term
        self.teacher_loaded = True

        if not 0.0 <= dagger_teacher_ratio <= 1.0:
            raise ValueError(f"dagger_teacher_ratio must lie in [0, 1], got {dagger_teacher_ratio}.")
        self.dagger_teacher_ratio = float(dagger_teacher_ratio)
        # Per env: whether the teacher drives it (drawn here and on each of its resets).
        self._teacher_rollout_mask = torch.zeros(storage.num_envs, dtype=torch.bool, device=self.device)
        if self.dagger_teacher_ratio > 0.0:
            self._teacher_rollout_mask = torch.rand(storage.num_envs, device=self.device) < self.dagger_teacher_ratio

    # Rollout

    def _teacher_actions(self, obs: TensorDict) -> torch.Tensor:
        """Run each teacher only on the envs routed to it."""
        route_ids = route(self._motion_term, self.motion_to_teacher_idx)
        actions = None
        for name, idx in zip(self.teacher_type_list, group_indices_by_id(route_ids, len(self.teacher_type_list))):
            if idx.numel() == 0:
                continue
            teacher_actions = self.teachers[name](obs[idx])
            if actions is None:
                actions = teacher_actions.new_zeros(route_ids.shape[0], *teacher_actions.shape[1:])
            actions[idx] = teacher_actions
        return actions  # type: ignore[return-value]

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Record the routed teachers' actions as targets; return the student's (or, under DAgger, the teacher's)."""
        student_actions = self.student(obs, stochastic_output=True).detach()
        with torch.no_grad():
            teacher_actions = self._teacher_actions(obs).detach().to(student_actions.dtype)
        actions = student_actions
        if self.dagger_teacher_ratio > 0.0:
            actions = torch.where(self._teacher_rollout_mask.unsqueeze(1), teacher_actions, student_actions)
        self.transition.actions = actions
        self.transition.privileged_actions = teacher_actions
        self.transition.observations = obs
        return actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step and redraw the DAgger driver of the envs that reset."""
        super().process_env_step(obs, rewards, dones, extras)
        if self.dagger_teacher_ratio > 0.0:
            # Redraw every env and keep the reset ones: no host sync.
            redraw = torch.rand(dones.numel(), device=self.device) < self.dagger_teacher_ratio
            self._teacher_rollout_mask = torch.where(dones.view(-1).bool(), redraw, self._teacher_rollout_mask)

    def train_mode(self) -> None:
        """Set train mode for the student and keep the teachers in eval mode."""
        self.student.train()
        self.teachers.eval()

    def eval_mode(self) -> None:
        """Set evaluation mode for the student and the teachers."""
        self.student.eval()
        self.teachers.eval()

    def compile(self, mode: str | None = None) -> None:
        """Compile the student only; the teachers run on per-step varying sub-batches."""
        self.student = compile_model(self._raw_student, mode)  # type: ignore

    def broadcast_parameters(self) -> None:
        """Broadcast the student from rank 0 (every rank loads the teachers from the same checkpoints)."""
        for tensor in self._raw_student.state_dict().values():
            torch.distributed.broadcast(tensor, src=0)

    # Checkpointing

    def save(self) -> dict:
        """Return the student, the optimizer, all teachers and the routing metadata."""
        return {
            "student_state_dict": self._raw_student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "teacher_state_dicts": {t: self.teachers[t].state_dict() for t in self.teacher_type_list},
            "motion_to_teacher_idx": self.motion_to_teacher_idx.detach().cpu(),
            "teacher_type_list": list(self.teacher_type_list),
            "teacher_registry_manifest": dict(self.teacher_registry_manifest),
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load the student, the optimizer and the teachers saved with it.

        ``load_cfg`` keys: ``student``, ``optimizer``, ``teachers``, ``iteration`` (all enabled by default).
        """
        if load_cfg is None:
            load_cfg = {"student": True, "optimizer": True, "teachers": True, "iteration": True}
        if load_cfg.get("student"):
            self._raw_student.load_state_dict(loaded_dict["student_state_dict"], strict=strict)
        if load_cfg.get("optimizer") and "optimizer_state_dict" in loaded_dict:
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.learning_rate = self.optimizer.param_groups[0]["lr"]
        if load_cfg.get("teachers") and "teacher_state_dicts" in loaded_dict:
            saved = loaded_dict["teacher_state_dicts"]
            for task_type, state_dict in saved.items():
                if task_type in self.teachers:
                    self.teachers[task_type].load_state_dict(state_dict, strict=strict)
            warn_state_dict_mismatch(type(self).__name__, saved, self.teacher_type_list)
        return bool(load_cfg.get("iteration", False))

    # Construction

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> MultiTeacherDistillation:
        """Build the student, one registry-loaded teacher per task type, the storage and the algorithm."""
        alg_class, alg_cfg = resolve_class(cfg["algorithm"])
        student_class, student_cfg = resolve_class(cfg["student"])
        teacher_class, teacher_cfg = resolve_class(cfg["teacher"])
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["student", "teacher"])

        student = student_class(obs, cfg["obs_groups"], "student", env.num_actions, **student_cfg).to(device)
        print(f"Student Model: {student}")

        registry = load_registry(alg_cfg.pop("teacher_registry_yaml", None))
        motion_term = get_motion_command(env)
        motion_to_teacher_idx = build_routing_table(registry, motion_term, device)

        teachers = nn.ModuleDict()
        for task_type in registry.teacher_type_list:
            path = registry.get(task_type).checkpoint_path
            teacher = teacher_class(obs, cfg["obs_groups"], "teacher", env.num_actions, **copy.deepcopy(teacher_cfg))
            checkpoint = load_checkpoint(path, device)
            state_dict = checkpoint.get("actor_state_dict", checkpoint.get("student_state_dict"))
            if state_dict is None:
                raise KeyError(f"teacher checkpoint {path} has neither 'actor_state_dict' nor 'student_state_dict'.")
            teacher.load_state_dict(state_dict, strict=True)
            teacher = teacher.to(device).eval()
            # Surface an observation / action mismatch before the rollout starts.
            with torch.no_grad():
                action_dim = teacher(obs.to(device)).shape[-1]
            if action_dim != env.num_actions:
                raise RuntimeError(
                    f"teacher[{task_type}] outputs {action_dim} actions but the env has {env.num_actions}."
                )
            teachers[task_type] = teacher
            print(f"Loaded teacher[{task_type}] from {path}")

        storage = RolloutStorage("distillation", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        alg: MultiTeacherDistillation = alg_class(
            student=student,
            teachers=teachers,
            teacher_type_list=registry.teacher_type_list,
            motion_to_teacher_idx=motion_to_teacher_idx,
            motion_term=motion_term,
            teacher_registry_manifest=registry.to_manifest(),
            storage=storage,
            device=device,
            **alg_cfg,
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        alg.compile(cfg.get("torch_compile_mode"))
        return alg
