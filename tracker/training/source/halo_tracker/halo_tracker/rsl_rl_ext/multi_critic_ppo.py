#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Builds on rsl_rl (Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION, BSD-3-Clause).

"""Multi-critic PPO: one shared actor and N critics, each env routed every step to the critic of its current motion.

Used for the PostRL fine-tune of the distilled student. The actor is initialized from the distilled student (its
exploration std is reset), and each critic is warm-started from the ``critic_state_dict`` of its PreRL teacher (see
:mod:`.routing`). The per-env route is recorded with every transition so that, in the update, each critic regresses
only the returns of the transitions routed to it while the shared actor trains on all of them. This keeps skills with
large returns from dominating the shared policy gradient. With ``normalize_advantage_per_critic`` the advantages are
also normalized per critic. RND, symmetry and recurrent models are not supported.
"""

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
from collections.abc import Iterator
from itertools import chain
from tensordict import TensorDict
from typing import Any

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.modules import HeteroscedasticGaussianDistribution
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import (
    compile_model,
    reduce_gradients_in_buckets,
    resolve_class,
    resolve_obs_groups,
    resolve_optimizer,
)

from .routing import build_routing_table, get_motion_command, load_checkpoint, load_registry, route
from .utils import group_indices_by_id, standardize, warn_state_dict_mismatch

ROUTE_KEY = "_route_id"
"""Observation entry that carries the per-env critic index through the rollout storage. The models never read it."""


class MultiCriticPPO(PPO):
    """PPO with one shared actor and N per-motion routed critics.

    Args (beyond :class:`rsl_rl.algorithms.PPO`):
        critics: ``{task_type: critic}``; all critics share one architecture and observation set.
        teacher_type_list: Critic order; index ``i`` is the critic of ``teacher_type_list[i]``.
        motion_to_teacher_idx: Per-motion critic index.
        motion_term: The env's motion command term, queried for each env's current motion.
        teacher_registry_manifest: Description of the routing registry, stored in checkpoints.
        normalize_advantage_per_critic: Normalize advantages within each critic's samples instead of globally.
        postrl_reset_std: Exploration std given to an actor loaded from a distilled student.

    Loading a distillation checkpoint (``student_state_dict`` without ``actor_state_dict``) initializes the actor from
    the student, resets its exploration std, and keeps the critics' teacher warm start.
    """

    critics: nn.ModuleDict
    teacher_type_list: list[str]

    def __init__(
        self,
        actor: MLPModel,
        critics: nn.ModuleDict,
        teacher_type_list: list[str],
        motion_to_teacher_idx: torch.Tensor,
        motion_term: Any,
        teacher_registry_manifest: dict,
        storage: RolloutStorage,
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        normalize_advantage_per_critic: bool = True,
        postrl_reset_std: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize the algorithm; see the class docstring for the arguments."""
        if not teacher_type_list:
            raise ValueError("MultiCriticPPO needs at least one critic, but teacher_type_list is empty.")
        if actor.is_recurrent or any(c.is_recurrent for c in critics.values()):
            raise NotImplementedError("MultiCriticPPO does not support recurrent actors or critics.")
        if kwargs.get("rnd_cfg") is not None or kwargs.get("symmetry_cfg") is not None:
            raise ValueError("MultiCriticPPO does not support the RND and symmetry extensions.")
        if normalize_advantage_per_critic and kwargs.get("normalize_advantage_per_mini_batch"):
            raise ValueError("normalize_advantage_per_critic and normalize_advantage_per_mini_batch are exclusive.")
        self.critics = critics.to(kwargs.get("device", "cpu"))
        self.teacher_type_list = list(teacher_type_list)
        super().__init__(
            actor, critics[self.teacher_type_list[0]], storage, optimizer=optimizer, learning_rate=learning_rate, **kwargs
        )
        # The upstream optimizer covers the actor and ONE critic; rebuild it over the actor and all critics.
        self.optimizer = resolve_optimizer(optimizer)(self._optimizer_params(), lr=learning_rate)  # type: ignore
        self.motion_to_teacher_idx = motion_to_teacher_idx.to(self.device, dtype=torch.long)
        self.teacher_registry_manifest = dict(teacher_registry_manifest)
        self.normalize_advantage_per_critic = normalize_advantage_per_critic
        self.postrl_reset_std = postrl_reset_std
        self._motion_term = motion_term
        self._normalizing_critics = [t for t in self.teacher_type_list if self.critics[t].obs_normalization]
        if self._normalizing_critics and self.is_multi_gpu:
            # The upstream normalizer all-reduces per update call and is NaN on an empty batch, while a critic's
            # routed batch can be empty on some ranks.
            raise NotImplementedError("MultiCriticPPO does not support critic obs_normalization under multi-GPU.")

    # Routing

    def _current_route(self) -> torch.Tensor:
        return route(self._motion_term, self.motion_to_teacher_idx)

    def _routed_values(self, obs: TensorDict, route_ids: torch.Tensor) -> torch.Tensor:
        """Evaluate each env with its routed critic."""
        values = torch.zeros(route_ids.shape[0], 1, device=self.device)
        for name, idx in zip(self.teacher_type_list, group_indices_by_id(route_ids, len(self.teacher_type_list))):
            if idx.numel() > 0:
                values[idx] = self.critics[name](obs[idx]).to(values.dtype)
        return values

    def _optimizer_params(self) -> Iterator[nn.Parameter]:
        return chain(self._raw_actor.parameters(), *(self.critics[t].parameters() for t in self.teacher_type_list))

    # Rollout

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions with the actor and record each env's routed value and critic index."""
        route_ids = self._current_route()
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        self.transition.values = self._routed_values(obs, route_ids).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()  # type: ignore
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        # Shallow copy: the stored observations carry the route, the env's (and the models' inputs) do not.
        self.transition.observations = obs.copy()
        self.transition.observations[ROUTE_KEY] = route_ids
        return self.transition.actions  # type: ignore

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute GAE returns and advantages, bootstrapping each env with its routed critic.

        The advantages are normalized globally or per critic, with statistics pooled over all ranks under multi-GPU.
        """
        st = self.storage
        last_values = self._routed_values(obs, self._current_route()).detach()
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        st.advantages = st.returns - st.values
        if not self.normalize_advantage_per_mini_batch:
            n = len(self.teacher_type_list) if self.normalize_advantage_per_critic else 1
            ids = st.observations[ROUTE_KEY] if self.normalize_advantage_per_critic else None
            st.advantages = standardize(st.advantages, self.is_multi_gpu, ids, n)

    # Update

    def update(self) -> dict[str, float]:
        """Run the PPO epochs; each critic regresses only its routed samples, the actor trains on all of them."""
        totals = {key: torch.zeros((), device=self.device) for key in ("value", "surrogate", "entropy")}
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        num_critics = len(self.teacher_type_list)

        for batch in generator:
            critic_ids = batch.observations.pop(ROUTE_KEY)  # type: ignore[union-attr]
            groups = group_indices_by_id(critic_ids, num_critics)
            self._batch_critic_counts = torch.bincount(critic_ids, minlength=num_critics)
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = standardize(batch.advantages, self.is_multi_gpu)  # type: ignore

            with torch.amp.autocast(  # type: ignore
                device_type=torch.device(self.device).type, enabled=self.use_mixed_precision, dtype=torch.bfloat16
            ):
                self.actor(batch.observations, stochastic_output=True)
                actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
                entropy = self.actor.output_entropy
                if self.desired_kl is not None and self.schedule == "adaptive":
                    self._adapt_learning_rate(batch.old_distribution_params, self.actor.output_distribution_params)

                ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
                advantages = torch.squeeze(batch.advantages)  # type: ignore
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.max(-advantages * ratio, -advantages * clipped_ratio).mean()

                # Each critic regresses only its routed samples; the sum is divided by the batch size, as a mean.
                value_loss = torch.zeros((), device=self.device)
                for name, idx in zip(self.teacher_type_list, groups):
                    if idx.numel() > 0:
                        values = self.critics[name](batch.observations[idx])
                        value_loss = value_loss + self._value_loss_per_sample(
                            values, batch.values[idx], batch.returns[idx]  # type: ignore[index]
                        ).sum()
                value_loss = value_loss / critic_ids.numel()
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            for critic in self.critics.values():
                nn.utils.clip_grad_norm_(critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            totals["value"] += value_loss.detach()
            totals["surrogate"] += surrogate_loss.detach()
            totals["entropy"] += entropy.detach().mean()

        obs = self.storage.observations.flatten(0, 1)
        self._update_normalizers(obs)
        num_updates = self.num_learning_epochs * self.num_mini_batches
        loss_dict = {key: (total / num_updates).item() for key, total in totals.items()}
        loss_dict.update(self._per_critic_rewards(obs))
        self.storage.clear()
        return loss_dict

    def _value_loss_per_sample(
        self, values: torch.Tensor, old_values: torch.Tensor, returns: torch.Tensor
    ) -> torch.Tensor:
        """Return the PPO value loss of each sample, clipped around the rollout values if configured."""
        if not self.use_clipped_value_loss:
            return (returns - values).pow(2)
        values_clipped = old_values + (values - old_values).clamp(-self.clip_param, self.clip_param)
        return torch.max((values - returns).pow(2), (values_clipped - returns).pow(2))

    def _adapt_learning_rate(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> None:
        """Upstream adaptive-KL learning-rate schedule (bounds 1e-5 / 1e-2)."""
        with torch.inference_mode():
            kl_mean = torch.mean(self.actor.get_kl_divergence(old_params, new_params))
            if self.is_multi_gpu:
                torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                kl_mean /= self.gpu_world_size
            if self.gpu_global_rank == 0:
                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
            if self.is_multi_gpu:
                lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                torch.distributed.broadcast(lr_tensor, src=0)
                self.learning_rate = lr_tensor.item()
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

    def _update_normalizers(self, obs: TensorDict) -> None:
        """Update the actor normalizer on all samples and each critic normalizer on its routed samples."""
        self.actor.update_normalization(obs)  # type: ignore
        if self._normalizing_critics:
            groups = group_indices_by_id(obs[ROUTE_KEY], len(self.teacher_type_list))
            for name, idx in zip(self.teacher_type_list, groups):
                if name in self._normalizing_critics and idx.numel() > 0:
                    self.critics[name].update_normalization(obs[idx])

    def _per_critic_rewards(self, obs: TensorDict) -> dict[str, float]:
        """Return the mean stored reward (incl. the time-out bootstrap) per critic as ``reward_<task>``."""
        route_ids, n = obs[ROUTE_KEY], len(self.teacher_type_list)
        rewards = self.storage.rewards.reshape(-1)
        stats = torch.cat([
            rewards.new_zeros(n).index_add_(0, route_ids, rewards),
            rewards.new_zeros(n).index_add_(0, route_ids, torch.ones_like(rewards)),
        ])
        if self.is_multi_gpu:
            torch.distributed.all_reduce(stats)
        total, count = stats[:n].tolist(), stats[n:].tolist()
        return {f"reward_{t}": total[i] / count[i] for i, t in enumerate(self.teacher_type_list) if count[i] > 0}

    def train_mode(self) -> None:
        """Set train mode for the actor and all critics."""
        self.actor.train()
        self.critics.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for the actor and all critics."""
        self.actor.eval()
        self.critics.eval()

    def compile(self, mode: str | None = None) -> None:
        """Compile the actor only; the critics run on per-step varying sub-batches."""
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore

    # Checkpointing

    def save(self) -> dict:
        """Return the actor, all critics, the optimizer and the routing metadata."""
        return {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_state_dicts": {t: self.critics[t].state_dict() for t in self.teacher_type_list},
            "optimizer_state_dict": self.optimizer.state_dict(),
            "motion_to_teacher_idx": self.motion_to_teacher_idx.detach().cpu(),
            "teacher_type_list": list(self.teacher_type_list),
            "teacher_registry_manifest": dict(self.teacher_registry_manifest),
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load the actor (or a distilled student as actor), the critics and the optimizer.

        ``load_cfg`` keys: ``actor``, ``critics``, ``optimizer``, ``iteration``, ``actor_from_student``. Without
        ``load_cfg``, a distilled-student checkpoint initializes the actor (the critics keep their teacher warm start)
        and any other checkpoint is a full resume.
        """
        if load_cfg is None:
            if "student_state_dict" in loaded_dict and "actor_state_dict" not in loaded_dict:
                load_cfg = {"actor_from_student": True, "iteration": False}
            else:
                load_cfg = {"actor": True, "critics": True, "optimizer": True, "iteration": True}
        if load_cfg.get("actor_from_student"):
            if "student_state_dict" not in loaded_dict:
                raise KeyError("load_cfg requests actor_from_student, but the checkpoint has no 'student_state_dict'.")
            self._raw_actor.load_state_dict(loaded_dict["student_state_dict"], strict=strict)
            self._reset_actor_exploration_std(self.postrl_reset_std)
            if "critic_state_dicts" in loaded_dict:
                self._load_critics(loaded_dict["critic_state_dicts"], strict)
                print("[MultiCriticPPO.load] actor from distilled student, critics loaded from the checkpoint.")
            else:
                print("[MultiCriticPPO.load] actor from distilled student, critics keep their teacher warm start.")
            return bool(load_cfg.get("iteration", False))

        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critics") and "critic_state_dicts" in loaded_dict:
            self._load_critics(loaded_dict["critic_state_dicts"], strict)
        saved_critics = loaded_dict.get("teacher_type_list")
        if load_cfg.get("optimizer") and saved_critics is not None and list(saved_critics) != self.teacher_type_list:
            # Optimizer state is matched by parameter position, so it would land on the wrong critics.
            print(
                f"[MultiCriticPPO.load] WARNING: checkpoint critics {list(saved_critics)} differ from the registry "
                f"{self.teacher_type_list}; starting a fresh optimizer."
            )
        elif load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            learning_rate = self.optimizer.param_groups[0]["lr"]
            if self.schedule == "adaptive" and self.desired_kl is not None:
                learning_rate = min(max(learning_rate, 1e-5), 1e-2)
            self.learning_rate = learning_rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = learning_rate
        return bool(load_cfg.get("iteration", False))

    def _load_critics(self, critic_state_dicts: dict, strict: bool) -> None:
        for task_type, state_dict in critic_state_dicts.items():
            if task_type in self.critics:
                self.critics[task_type].load_state_dict(state_dict, strict=strict)
        warn_state_dict_mismatch(type(self).__name__, critic_state_dicts, self.teacher_type_list)

    def _reset_actor_exploration_std(self, initial_std: float) -> None:
        """Set the actor's state-independent exploration std; a distilled student's is near zero."""
        distribution = getattr(self._raw_actor, "distribution", None)
        if isinstance(distribution, HeteroscedasticGaussianDistribution):
            return  # the std is an output of the network
        with torch.no_grad():
            if hasattr(distribution, "std_param"):
                distribution.std_param.fill_(initial_std)  # type: ignore[union-attr]
            elif hasattr(distribution, "log_std_param"):
                distribution.log_std_param.fill_(math.log(initial_std))  # type: ignore[union-attr]
            else:
                raise ValueError("MultiCriticPPO: the actor has no state-independent std to reset.")

    # Multi-GPU

    def broadcast_parameters(self) -> None:
        """Broadcast the actor and all critics from rank 0."""
        for model in (self._raw_actor, *(self.critics[t] for t in self.teacher_type_list)):
            for tensor in model.state_dict().values():
                torch.distributed.broadcast(tensor, src=0)

    def reduce_parameters(self) -> None:
        """Average the gradients of the actor and all critics across ranks.

        A critic can receive samples on some ranks but not on others, so missing gradients are zero-filled to keep
        the packed buffers identical across ranks. A critic without samples on every rank keeps no gradient, so the
        optimizer skips it as it does on a single GPU.
        """
        params = [p for p in self._optimizer_params() if p.requires_grad]
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        reduce_gradients_in_buckets(params, self.gpu_world_size, self.grad_reduce_bucket_mb)
        counts = self._batch_critic_counts.clone()
        torch.distributed.all_reduce(counts)
        for name, count in zip(self.teacher_type_list, counts.tolist()):
            if count == 0:
                for p in self.critics[name].parameters():
                    p.grad = None

    # Construction

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> MultiCriticPPO:
        """Build the actor, one registry warm-started critic per task type, the storage and the algorithm."""
        alg_class, alg_cfg = resolve_class(cfg["algorithm"])
        actor_class, actor_cfg = resolve_class(cfg["actor"])
        critic_class, critic_cfg = resolve_class(cfg["critic"])
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        if alg_cfg.pop("share_cnn_encoders", None):
            raise ValueError("MultiCriticPPO does not support share_cnn_encoders.")

        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)
        print(f"Actor Model: {actor}")

        registry = load_registry(alg_cfg.pop("teacher_registry_yaml", None))
        motion_term = get_motion_command(env)
        motion_to_teacher_idx = build_routing_table(registry, motion_term, device)

        critics = nn.ModuleDict()
        for task_type in registry.teacher_type_list:
            critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **copy.deepcopy(critic_cfg)).to(device)
            path = registry.get(task_type).checkpoint_path
            checkpoint = load_checkpoint(path, device)
            if "critic_state_dict" not in checkpoint:
                raise KeyError(
                    f"teacher checkpoint {path} has no 'critic_state_dict'; point the registry at PreRL (PPO) checkpoints."
                )
            critic.load_state_dict(checkpoint["critic_state_dict"], strict=True)
            print(f"critic[{task_type}]: warm-started from {path}")
            critics[task_type] = critic

        # The route entry rides along in the stored observations.
        storage_obs = obs.copy()
        storage_obs[ROUTE_KEY] = torch.zeros(env.num_envs, dtype=torch.long, device=obs.device)
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], storage_obs, [env.num_actions], device)

        alg: MultiCriticPPO = alg_class(
            actor=actor,
            critics=critics,
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
