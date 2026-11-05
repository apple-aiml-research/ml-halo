#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Builds on rsl_rl (Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION, BSD-3-Clause).

"""Helpers shared by the algorithms and runners of this package."""

from __future__ import annotations

import torch
import warnings
from collections.abc import Iterable
from typing import Any


def warn_unused_alg_kwargs(alg_name: str, kwargs: dict) -> None:
    """Warn about algorithm-config keys that are not accepted, so a misspelled key is not silently ignored."""
    if kwargs:
        warnings.warn(f"[{alg_name}] ignoring unknown algorithm-cfg key(s): {sorted(kwargs)}.", stacklevel=3)


def warn_state_dict_mismatch(owner: str, saved: Iterable[str], current: Iterable[str]) -> None:
    """Warn when a checkpoint and the registry hold different sets of per-task models."""
    saved, current = set(saved), set(current)
    if saved != current:
        print(
            f"[{owner}.load] WARNING: skipped checkpoint models not in the registry {sorted(saved - current)};"
            f" registry models without a checkpoint entry keep their weights {sorted(current - saved)}."
        )


def group_indices_by_id(ids: torch.Tensor, num_groups: int) -> list[torch.Tensor]:
    """Return, for each id in ``[0, num_groups)``, the ascending positions of ``ids`` equal to it.

    Costs a constant number of host syncs, unlike a per-group ``(ids == g).nonzero()`` loop.
    """
    sizes = torch.bincount(ids, minlength=num_groups).tolist()
    if len(sizes) > num_groups:
        raise ValueError(f"group_indices_by_id: ids contain values >= num_groups ({num_groups}).")
    return list(torch.split(torch.argsort(ids, stable=True), sizes))


def standardize(
    x: torch.Tensor, is_multi_gpu: bool, group_ids: torch.Tensor | None = None, num_groups: int = 1
) -> torch.Tensor:
    """Return ``x`` shifted to zero mean and scaled to unit std, per group if ``group_ids`` is given.

    Under multi-GPU the statistics are pooled over all ranks. The mean and the squared deviations are reduced in two
    passes; ``E[x^2] - E[x]^2`` would lose precision in fp32. Groups with a single sample are only centered.
    """
    if group_ids is None and not is_multi_gpu:
        return (x - x.mean()) / (x.std() + 1e-8)  # upstream rsl_rl formula
    flat = x.reshape(-1)
    ids = torch.zeros_like(flat, dtype=torch.long) if group_ids is None else group_ids.reshape(-1)
    count = flat.new_zeros(num_groups).index_add_(0, ids, torch.ones_like(flat))
    total = flat.new_zeros(num_groups).index_add_(0, ids, flat)
    if is_multi_gpu:
        stats = torch.cat([count, total])
        torch.distributed.all_reduce(stats)
        count, total = stats[:num_groups], stats[num_groups:]
    deviation = flat - (total / count.clamp_min(1.0))[ids]
    sq_dev = flat.new_zeros(num_groups).index_add_(0, ids, deviation * deviation)
    if is_multi_gpu:
        torch.distributed.all_reduce(sq_dev)
    std = torch.where(count > 1, (sq_dev / (count - 1.0).clamp_min(1.0)).sqrt(), torch.zeros_like(sq_dev))
    return (deviation / (std[ids] + 1e-8)).view_as(x)


def reach_command_manager(env: Any, max_depth: int = 8) -> Any:
    """Walk a vec-env wrapper chain (``.unwrapped`` / ``.env``) to the object owning ``command_manager``."""
    inner = env
    for _ in range(max_depth):
        if hasattr(inner, "command_manager"):
            return inner
        nxt = getattr(inner, "unwrapped", None)
        if nxt is None or nxt is inner:
            nxt = getattr(inner, "env", None)
        if nxt is None or nxt is inner:
            break
        inner = nxt
    raise AttributeError(f"No 'command_manager' found on the wrapper chain of {type(env).__name__}.")


_NAN_WARN_EVERY = 1000


def sanitize_nan(obs: Any, rewards: torch.Tensor, dones: torch.Tensor) -> None:
    """Drop-in for upstream ``rsl_rl.utils.check_nan`` that absorbs NaN env outputs instead of raising.

    One env whose physics diverges should not end a run with thousands of envs. NaN rewards and observation entries
    are zeroed in place and the affected envs are marked done, so the rollout cuts their GAE bootstrap. The simulator
    reset comes from the env's own terminations (the tracking envs end an episode on a non-finite robot state). The
    clean case costs a single host sync. The warning repeats every ``_NAN_WARN_EVERY`` NaN-affected steps so a
    persistent problem stays visible.
    """
    bad = torch.isnan(rewards)
    for tensor in obs.values():
        bad = bad | torch.isnan(tensor).reshape(tensor.shape[0], -1).any(dim=1)
    if not bool(bad.any()):
        return
    sanitize_nan.nan_steps = getattr(sanitize_nan, "nan_steps", 0) + 1
    if (sanitize_nan.nan_steps - 1) % _NAN_WARN_EVERY == 0:
        print(f"[sanitize_nan] NaN env output in {int(bad.sum())} env(s); zeroed and marked done"
              f" (NaN step #{sanitize_nan.nan_steps}).")
    rewards.masked_fill_(torch.isnan(rewards), 0.0)
    for tensor in obs.values():
        tensor.masked_fill_(torch.isnan(tensor), 0.0)
    dones[bad] = True


def install_nan_sanitizer() -> None:
    """Make the upstream runners call :func:`sanitize_nan` where they would call ``check_nan``."""
    import rsl_rl.runners.on_policy_runner as on_policy_runner

    on_policy_runner.check_nan = sanitize_nan
