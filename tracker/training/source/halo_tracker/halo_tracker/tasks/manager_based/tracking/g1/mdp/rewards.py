#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Builds on BeyondMimic whole_body_tracking (Copyright (c) 2024, The Isaac Lab Project Developers, MIT License).

"""Reward functions for the G1 motion-tracking tasks.

The motion-tracking terms consume per-env state from the ``MotionCommand`` term
(``env.command_manager.get_term(command_name)``) so the rewards stay agnostic to the env class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_error_magnitude

from .commands import get_body_indexes

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import MotionCommand


# -- Motion-tracking rewards (command-driven) -------------------------------

def motion_global_anchor_position_error_exp(
    env: "ManagerBasedRLEnv", command_name: str, std: float
) -> torch.Tensor:
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
    env: "ManagerBasedRLEnv", command_name: str, std: float
) -> torch.Tensor:
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: "ManagerBasedRLEnv", command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    body_indexes = get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(
            command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: "ManagerBasedRLEnv", command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    body_indexes = get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(
            command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes]
        )
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: "ManagerBasedRLEnv", command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    body_indexes = get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(
            command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: "ManagerBasedRLEnv", command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    body_indexes = get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(
            command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def survival_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Constant +1 per step the robot stays alive."""
    return torch.ones(env.num_envs, device=env.device)


# -- Object-interaction and safety terms -------------------------------------

def _hand_object_proximity_gate(
    env: ManagerBasedRLEnv,
    command: "MotionCommand",
    asset_cfg: SceneEntityCfg,
    hand_body_names: list[str],
    gate_std: float,
    use_sim_object: bool = False,
) -> torch.Tensor:
    """Soft gate ∈ (0, 1] = exp(-(min_hand_to_object_dist / gate_std)^2).

    Gates object-tracking rewards so the policy is only paid when a hand is
    near the object (avoids premature-reach/collapse). Ref: ResMimic, MaskedMimic.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_idx, _ = asset.find_bodies(hand_body_names)
    hand_pos_w = asset.data.body_pos_w[:, body_idx, :]                           # (B, H, 3)
    # Gate against the REFERENCE object by default; ``use_sim_object`` switches to
    # the SIMULATED object (the physical object a hand must actually reach — used
    # by the contact-seeking reward).
    obj_ref = command.robot_object_pos_w if use_sim_object else command.object_pos_w
    obj_pos_w = obj_ref.unsqueeze(1)                                             # (B, 1, 3)
    dist = torch.linalg.norm(hand_pos_w - obj_pos_w, dim=-1).min(dim=-1).values  # (B,)
    return torch.exp(-(dist / gate_std) ** 2)


def object_global_position_error_exp_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hand_body_names: tuple[str, ...] = ("left_wrist_yaw_link", "right_wrist_yaw_link"),
    gate_std: float = 0.3,
    use_sim_object: bool = False,
) -> torch.Tensor:
    """Object position tracking ``exp(-|p_ref - p_sim|^2 / std^2)``, gated by hand proximity.

    Masked to envs with an active object. ``use_sim_object=True`` gates on the REAL
    (physics) object instead of the reference, so the reward only fires when the hand
    is near the actual object — use it when the motion reference may be imperfect.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    if command.robot_object_pos_w is None:
        return torch.zeros(env.num_envs, device=env.device)
    error = torch.sum(torch.square(command.object_pos_w - command.robot_object_pos_w), dim=-1)
    reward = torch.exp(-error / std**2)
    reward = reward * _hand_object_proximity_gate(
        env, command, asset_cfg, list(hand_body_names), gate_std, use_sim_object=use_sim_object)
    reward = reward * command.has_object_mask.float()
    return reward


def object_global_orientation_error_exp_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hand_body_names: tuple[str, ...] = ("left_wrist_yaw_link", "right_wrist_yaw_link"),
    gate_std: float = 0.3,
    use_sim_object: bool = False,
) -> torch.Tensor:
    """Object orientation tracking ``exp(-angle_err^2 / std^2)``, gated by hand proximity (see the position term)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    if command.robot_object_quat_w is None:
        return torch.zeros(env.num_envs, device=env.device)
    error = quat_error_magnitude(command.object_quat_w, command.robot_object_quat_w) ** 2
    reward = torch.exp(-error / std**2)
    reward = reward * _hand_object_proximity_gate(
        env, command, asset_cfg, list(hand_body_names), gate_std, use_sim_object=use_sim_object)
    reward = reward * command.has_object_mask.float()
    return reward


def object_hand_contact_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hand_body_names: tuple[str, ...] = ("left_wrist_yaw_link", "right_wrist_yaw_link"),
) -> torch.Tensor:
    """Contact-seeking bonus: reward bringing a hand to the (sim) object.

    ``reward = exp(-(min_hand_to_object_dist / std)^2)`` against the SIMULATED
    object pose (``robot_object_pos_w``), masked to active-object envs. Supplies
    the gradient that pulls a hand onto the object from a cold start, complementing
    the hand-GATED pose tracking. Apply a POSITIVE weight. Ref: FALCON / WoCoCo / ResMimic.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    if command.robot_object_pos_w is None:
        return torch.zeros(env.num_envs, device=env.device)
    # Reuse the proximity kernel against the SIM object (gate_std = std).
    reward = _hand_object_proximity_gate(
        env, command, asset_cfg, list(hand_body_names), std, use_sim_object=True
    )
    reward = reward * command.has_object_mask.float()
    return reward


def object_drop_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    height_margin: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hand_body_names: tuple[str, ...] = ("left_wrist_yaw_link", "right_wrist_yaw_link"),
    gate_std: float = 0.3,
    use_sim_object: bool = False,
) -> torch.Tensor:
    """Penalize the object dropping below its reference height during interaction.

    ``penalty = relu((ref_object_z - height_margin) - sim_object_z)`` — positive
    only once the object falls more than ``height_margin`` below the reference.
    Gated by hand proximity and masked to active-object envs. A graded precursor
    to the hard ``bad_object_pos`` termination. Returns a POSITIVE magnitude — apply
    a NEGATIVE weight in the cfg.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    if command.robot_object_pos_w is None:
        return torch.zeros(env.num_envs, device=env.device)
    ref_z = command.object_pos_w[:, 2]
    sim_z = command.robot_object_pos_w[:, 2]
    drop = torch.clamp((ref_z - height_margin) - sim_z, min=0.0)
    gate = _hand_object_proximity_gate(env, command, asset_cfg, list(hand_body_names), gate_std, use_sim_object=use_sim_object)
    penalty = drop * gate
    penalty = penalty * command.has_object_mask.float()
    return penalty


def power_safety_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    deadband: float = 150.0,
    normalizer: float = 500.0,
    clip_max: float = 10.0,
) -> torch.Tensor:
    """Penalize large negative mechanical power (regenerative braking).

    Guards the real motors against overcurrent and thermal stress during high-impact contacts (knees especially).

    P_j = τ_j · ω_j.  Only negative power beyond a deadband is penalized:
        L = Σ_j (max(-P_j - deadband, 0) / normalizer)²

    Clipped to ``clip_max`` so a single pathological step (velocity saturation, contact explosion) cannot blow up
    the reward; ``clip_max=float('inf')`` disables the clip.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    velocities = asset.data.joint_vel[:, asset_cfg.joint_ids]
    power = torques * velocities
    violation = torch.clamp(-power - deadband, min=0.0) / normalizer
    return torch.sum(violation ** 2, dim=-1).clamp(max=clip_max)


def joint_pos_limits_clipped(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    clip_max: float = 10.0,
) -> torch.Tensor:
    """``isaaclab.envs.mdp.joint_pos_limits`` (sum of the per-joint out-of-soft-limit amounts) clamped per env to
    ``clip_max``, so a physics blow-up cannot drown out the other terms."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    soft_lo = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    soft_hi = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    # Out-of-limit amount below soft_lo OR above soft_hi, per joint.
    below = -(joint_pos - soft_lo).clamp(max=0.0)
    above = (joint_pos - soft_hi).clamp(min=0.0)
    return torch.sum(below + above, dim=1).clamp(max=clip_max)


def feet_stance_slip_motion_track(
    env, command_name: str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_thresh: float = 20.0, ref_stance_z: float = 0.055,
    sole_offset: tuple[float, float, float] = (0.04, 0.0, -0.035), max_speed: float = 1.0,
) -> torch.Tensor:
    """Horizontal sole speed while the foot is loaded AND the reference foot is in stance (non-negative cost).

    Velocity at a sole point (``v + w x r``, link frame) so heel-to-toe roll is not slip; gated on the current
    contact force and on the reference ankle height, so commanded slides / swings are never punished.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    command = env.command_manager.get_term(command_name)
    feet = ("left_ankle_roll_link", "right_ankle_roll_link")
    body_ids = [asset.body_names.index(n) for n in feet]
    sensor_ids = [sensor.find_bodies(n)[0][0] for n in feet]
    ref_ids = [list(command.cfg.body_names).index(n) for n in feet]

    q = asset.data.body_quat_w[:, body_ids]
    v = asset.data.body_link_lin_vel_w[:, body_ids]
    w = asset.data.body_link_ang_vel_w[:, body_ids]
    r = torch.tensor(sole_offset, device=v.device, dtype=v.dtype).expand(v.shape[0], 2, 3)
    v_sole = v + torch.cross(w, quat_apply(q.reshape(-1, 4), r.reshape(-1, 3)).reshape(v.shape), dim=-1)
    speed = torch.linalg.norm(v_sole[..., :2], dim=-1).clamp(max=max_speed)
    loaded = torch.linalg.norm(sensor.data.net_forces_w[:, sensor_ids], dim=-1) > force_thresh
    ref_stance = (command.body_pos_w[:, ref_ids, 2] - env.scene.env_origins[:, 2:3]) < ref_stance_z
    return torch.sum(speed * (loaded & ref_stance).float(), dim=1)


def undesired_contacts_ref_gated(
    env, command_name: str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 1.0, ref_clear_z: float = 0.15, robot_low_z: float = 0.20,
) -> torch.Tensor:
    """Count floor-level contacts on bodies whose REFERENCE is clear of the floor (non-negative; use a negative weight).

    The plain ``undesired_contacts`` also charges contacts the reference wants (sitting, kneeling). A body counts only
    if its contact force exceeds ``threshold``, its reference is more than ``ref_clear_z`` above the env origin, and
    the robot body is below ``robot_low_z`` (on the floor, not on a sofa seat or a box). Every sensor body must be
    tracked by the command (it needs the reference height).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    command = env.command_manager.get_term(command_name)
    cache = getattr(sensor, "_ref_gated_idx", None)
    if cache is None:
        names = list(command.cfg.body_names)
        s_idx = list(sensor_cfg.body_ids)
        s_names = [sensor.body_names[i] for i in s_idx]       # body_names is slow: read once
        untracked = [n for n in s_names if n not in names]
        if untracked:
            raise ValueError(f"undesired_contacts_ref_gated: {untracked} are not tracked by command '{command_name}'")
        cache = (s_idx, [names.index(n) for n in s_names], [asset.body_names.index(n) for n in s_names])
        sensor._ref_gated_idx = cache
    s_idx, r_idx, a_idx = cache
    origin_z = env.scene.env_origins[:, 2:3]
    in_contact = torch.linalg.norm(sensor.data.net_forces_w[:, s_idx], dim=-1) > threshold
    ref_clear = (command.body_pos_w[:, r_idx, 2] - origin_z) > ref_clear_z
    robot_low = (asset.data.body_pos_w[:, a_idx, 2] - origin_z) < robot_low_z
    return torch.sum((in_contact & ref_clear & robot_low).float(), dim=1)
