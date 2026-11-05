#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Builds on BeyondMimic whole_body_tracking (Copyright (c) 2024, The Isaac Lab Project Developers, MIT License).

"""Termination functions specific to the motion-tracking environment.

All terminations consume per-env state from the ``MotionCommand`` term
(``env.command_manager.get_term(command_name)``) so they stay agnostic
to the env class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, quat_error_magnitude

from .commands import MotionCommand, get_body_indexes

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def motion_ends(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """Terminate when the motion clip reaches its end."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.motion.is_at_motion_boundary(command.time_steps)


def bad_anchor_pos(env: "ManagerBasedRLEnv", command_name: str, threshold: float) -> torch.Tensor:
    """Terminate when the robot anchor drifts more than ``threshold`` (m) from the reference."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_ori(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    """Terminate when the robot anchor's gravity z-component diverges beyond ``threshold``.

    Gravity-vector based instead of quaternion-error based — ignores yaw,
    which the motion command intentionally re-aligns each step.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_proj_grav_b = quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)
    robot_proj_grav_b = quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)
    return (motion_proj_grav_b[:, 2] - robot_proj_grav_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: "ManagerBasedRLEnv",
    command_name: str,
    threshold: float,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Terminate when any tracked body deviates more than ``threshold`` (m) from its anchor-relative target."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = get_body_indexes(command, body_names)
    error = torch.norm(
        command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes],
        dim=-1,
    )
    return torch.any(error > threshold, dim=-1)


# -- Safety, Z-only and object terminations --------------------------------

def joint_vel_out_of_limit(
    env: ManagerBasedRLEnv,
    threshold: float = 30.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when any joint velocity magnitude exceeds ``threshold`` (rad/s), or the robot state is not finite.

    Safety circuit-breaker: G1 joint speeds normally stay below 20 rad/s, so a larger speed almost always means a
    physics blow-up. NaN fails every threshold comparison, so a diverged env is caught by the explicit finiteness
    check (the only termination that resets it).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids is not None:
        joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    else:
        joint_vel = asset.data.joint_vel
    finite = torch.isfinite(joint_vel).all(dim=-1) & torch.isfinite(asset.data.root_pos_w).all(dim=-1)
    return (joint_vel.abs() > threshold).any(dim=-1) | ~finite


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)


def bad_object_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Terminate if object position deviates from reference — masked for envs without object."""
    command = env.command_manager.get_term(command_name)
    if command.robot_object_pos_w is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    exceeded = torch.norm(command.object_pos_w - command.robot_object_pos_w, dim=-1) > threshold
    return exceeded & command.has_object_mask


def bad_object_ori(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Terminate if object orientation deviates from reference — masked for envs without object."""
    command = env.command_manager.get_term(command_name)
    if command.robot_object_quat_w is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    exceeded = quat_error_magnitude(command.object_quat_w, command.robot_object_quat_w) > threshold
    return exceeded & command.has_object_mask
