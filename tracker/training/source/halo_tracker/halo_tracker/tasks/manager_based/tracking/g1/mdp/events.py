#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Builds on BeyondMimic whole_body_tracking (Copyright (c) 2024, The Isaac Lab Project Developers, MIT License).

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions for the motion-tracking environment.

Reference-motion sampling and the initial robot pose write live inside
``MotionCommand._resample_command`` (see ``mdp/commands.py``); the
events here cover the remaining domain-randomization terms.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Literal

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the joint default positions to model calibration error vs URDF."""
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the joint-position action's offset since it is not refreshed automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Add a random CoM offset (one draw per env, shared by the selected bodies; sampled once at startup)."""
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu"
    ).unsqueeze(1)

    coms = asset.root_physx_view.get_coms().clone()
    coms[:, body_ids, :3] += rand_samples
    asset.root_physx_view.set_coms(coms, env_ids)


def randomize_rigid_body_com_per_body(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Like ``randomize_rigid_body_com``, but an independent offset per selected body (per-link CAD error).

    Startup only (CPU tensors, like ``randomize_rigid_body_com``).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    env_ids = torch.arange(env.scene.num_envs, device="cpu") if env_ids is None else env_ids.cpu()
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.long, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.long, device="cpu")
    ranges = torch.tensor([com_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z")], device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), len(body_ids), 3), device="cpu")
    coms = asset.root_physx_view.get_coms().clone()
    coms[env_ids[:, None], body_ids[None, :], :3] += rand_samples
    asset.root_physx_view.set_coms(coms, env_ids)


# ±2.5° in radians (RPL §III-B / Perceptive-Humanoid-Parkour camera-modeling recipe).
_DEG25_RAD: float = math.radians(2.5)


def randomize_depth_camera_extrinsics(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    sensor_cfg: SceneEntityCfg,
    pos_range_m: tuple[float, float] = (-0.025, 0.025),
    rot_range_rad: tuple[float, float] = (-_DEG25_RAD, _DEG25_RAD),
    parent_body_name: str = "torso_link",
    robot_name: str = "robot",
) -> None:
    """Per-reset extrinsic DR for the student's depth camera.

    Composes uniform translation (``pos_range_m``) and rotation (``rot_range_rad``)
    noise with the calibrated mount offset (``sensor.cfg.offset``) and writes the
    resulting world pose — RPL §III-B: ±2.5 cm / ±2.5° around the calibrated value.
    ``sensor.cfg.offset`` is assumed to be ``convention="world"`` (body-frame +X fwd, +Z up).
    """
    sensor = env.scene[sensor_cfg.name]
    robot = env.scene[robot_name]
    device = env.device

    # Cache the parent body index on first call.
    attr = f"_camera_dr_{parent_body_name}_idx"
    body_idx = getattr(env, attr, None)
    if body_idx is None:
        body_idx = robot.body_names.index(parent_body_name)
        setattr(env, attr, body_idx)

    parent_state = robot.data.body_state_w[env_ids, body_idx, 0:7]  # (n, 7) pos+quat
    parent_pos = parent_state[:, 0:3]
    parent_quat = parent_state[:, 3:7]

    base_pos = torch.as_tensor(sensor.cfg.offset.pos, device=device, dtype=torch.float32)
    base_quat = torch.as_tensor(sensor.cfg.offset.rot, device=device, dtype=torch.float32)

    n = env_ids.numel()
    pos_noise = torch.empty(n, 3, device=device).uniform_(*pos_range_m)
    rpy_noise = torch.empty(n, 3, device=device).uniform_(*rot_range_rad)
    noise_quat = math_utils.quat_from_euler_xyz(rpy_noise[:, 0], rpy_noise[:, 1], rpy_noise[:, 2])

    # Perturbed offset in the parent (body) frame, then compose into world.
    perturbed_offset_pos = base_pos.unsqueeze(0) + pos_noise
    perturbed_offset_quat = math_utils.quat_mul(base_quat.unsqueeze(0).expand(n, 4), noise_quat)
    world_pos = parent_pos + math_utils.quat_apply(parent_quat, perturbed_offset_pos)
    world_quat = math_utils.quat_mul(parent_quat, perturbed_offset_quat)

    # convention="world": set_world_poses converts world→USD camera convention internally.
    sensor.set_world_poses(
        positions=world_pos,
        orientations=world_quat,
        env_ids=env_ids,
        convention="world",
    )


def randomize_actuator_effort_scale(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    scale_range: tuple[float, float],
    not_reached_min: float | None = None,
    not_reached_joints: tuple[str, ...] = (),
):
    """Scale each explicit actuator's torque limit per env by U[scale_range] around its nominal (battery sag).

    Joints matching ``not_reached_joints`` (limit is a rated value, not a measured clip) use ``not_reached_min``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    lo, hi = float(scale_range[0]), float(scale_range[1])
    lo_nr = lo if not_reached_min is None else float(not_reached_min)
    for act in asset.actuators.values():
        lim = getattr(act, "effort_limit", None)
        if not isinstance(lim, torch.Tensor) or lim.ndim != 2:
            continue
        if not hasattr(act, "_effort_limit_nominal"):
            act._effort_limit_nominal = lim.clone()
        lows = torch.tensor([lo_nr if any(re.fullmatch(p, n) for p in not_reached_joints) else lo
                             for n in act.joint_names], device=lim.device)
        u = torch.rand(len(env_ids), lim.shape[1], device=lim.device)
        lim[env_ids] = act._effort_limit_nominal[env_ids] * (lows + u * (hi - lows))


def randomize_rigid_body_inertia_only(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_distribution_params: tuple[float, float],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Scale the bodies' inertia tensor by one random factor per env, keeping the mass (payload shape).

    Startup only: reads and writes the PhysX CPU inertia buffer.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    env_ids = torch.arange(env.scene.num_envs, device="cpu") if env_ids is None else env_ids.cpu()
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")
    inertias = asset.root_physx_view.get_inertias().clone()
    ones = torch.ones(inertias.shape[0], inertias.shape[1], device="cpu")
    scale = _randomize_prop_by_op(ones, inertia_distribution_params, env_ids, body_ids, operation="scale",
                                  distribution=distribution)
    inertias *= scale.unsqueeze(-1)
    asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_joint_friction(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    friction_ceiling_by_group: dict[str, float],
    group_joint_regex: dict[str, str],
):
    """Joint stiction U[0, ceiling] per joint per env; ``group_joint_regex`` maps each ceiling's family to its joints.

    Written through the PhysX joint friction buffer, so it also acts on explicit (DelayedPD) actuators. Each joint
    draws independently: the measured left / right stiction differs.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    cache = getattr(env, "_joint_friction_cache", None)
    if cache is None:
        ids, ceils = [], []
        for fam, ceil in friction_ceiling_by_group.items():
            rx = re.compile(group_joint_regex[fam])
            for i, n in enumerate(asset.joint_names):
                if rx.fullmatch(n) and i not in ids:
                    ids.append(i)
                    ceils.append(float(ceil))
        if not ids:
            raise ValueError(f"randomize_joint_friction matched no joints for {group_joint_regex}")
        cache = (torch.tensor(ids, dtype=torch.long, device=asset.device),
                 torch.tensor(ceils, dtype=torch.float32, device=asset.device))
        env._joint_friction_cache = cache
    ids, ceil_t = cache
    fric = torch.rand((len(env_ids), len(ids)), device=asset.device) * ceil_t
    asset.write_joint_friction_coefficient_to_sim(fric, joint_ids=ids, env_ids=env_ids)


def floor_compliance_levels(stiffness_range: tuple[float, float], damping_ratio_range: tuple[float, float],
                            stiffness_levels: int, damping_ratio_levels: int,
                            contact_mass_kg: float) -> list[tuple[float, float]]:
    """(stiffness N/m, damping N·s/m) of the compliant floor materials: log-spaced stiffness levels, each paired with
    linearly spaced damping ratios, damping d = 2·zeta·sqrt(k·m_c) on the mass m_c one contact point carries."""
    def spaced(lo, hi, n, log):
        if n <= 1:
            return [lo]
        return [lo * (hi / lo) ** (j / (n - 1)) if log else lo + (hi - lo) * j / (n - 1) for j in range(n)]

    return [(k, 2.0 * z * math.sqrt(k * contact_mass_kg))
            for k in spaced(*stiffness_range, stiffness_levels, True)
            for z in spaced(*damping_ratio_range, damping_ratio_levels, False)]


def floor_compliance_assignment(is_flat: list[bool], fraction: float, n_levels: int) -> list[int]:
    """Per env: -1 = rigid floor, j = compliant level j. Only flat-plate envs are eligible. A golden-ratio sequence
    over the flat envs picks ``fraction`` of them without clustering in env-id order (so every rank of a multi-GPU
    run gets its share), and the picked envs cycle through the levels."""
    out, picked, n_flat = [], 0, 0
    for flat in is_flat:
        if not flat:
            out.append(-1)
            continue
        n_flat += 1
        if (n_flat * 0.6180339887498949) % 1.0 < fraction:
            out.append(picked % max(1, n_levels))
            picked += 1
        else:
            out.append(-1)
    return out


def randomize_floor_compliance(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    terrain_name: str = "terrain_object",
    fraction: float = 1.0 / 3.0,
    stiffness_range: tuple[float, float] = (1300.0, 6500.0),
    damping_ratio_range: tuple[float, float] = (0.3, 1.0),
    stiffness_levels: int = 4,
    damping_ratio_levels: int = 3,
    contact_mass_kg: float = 1.3,
    raise_m: float = 0.01,
):
    """PRESTARTUP: a compliant floor (carpet) under ``fraction`` of the flat-plate envs, a rigid one under the rest.

    Friction and restitution cannot produce carpet: its defining property is a damped sink of a few millimetres to
    centimetres under load. Each picked env's plate gets a PhysX compliant-contact material (an implicit spring per
    contact point, stiffness from ``stiffness_range``, damping from ``damping_ratio_range``) and is raised by
    ``raise_m`` like a rug, so the foot can sink by up to the plate's height above the ground plane before the rigid
    ground stops it. Every other flat plate gets the rigid floor material of the ground (friction 1, restitution 0,
    multiply combine), so the per-body friction DR acts the same on every floor.

    The plates are the env's ``TerrainMesh`` (``FloorPlateCfg``: thick, so a sinking foot never crosses its mid-plane
    and flips the contact normal), which the height scanner and the depth camera ray-cast, so both see the carpet's
    top. Needs ``scene.replicate_physics=False`` (per-env materials).

    Per-env results: ``env.floor_compliance_level`` (-1 = rigid or no plate), ``env.floor_stiffness`` /
    ``env.floor_damping`` (0 = rigid) and ``env.floor_top`` (plate top in the env frame; 0 = no plate)."""
    import isaaclab.sim as sim_utils
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    tcfg = getattr(env.cfg.scene, terrain_name, None)
    assets = getattr(getattr(tcfg, "spawn", None), "assets_cfg", None)
    if not assets:
        raise ValueError(f"randomize_floor_compliance: the scene has no per-env terrain '{terrain_name}'")
    if getattr(env.cfg.scene, "replicate_physics", True):
        raise ValueError("randomize_floor_compliance needs scene.replicate_physics=False (per-env materials)")
    is_flat = [isinstance(assets[i % len(assets)], sim_utils.MeshCuboidCfg) for i in range(env.num_envs)]
    levels = floor_compliance_levels(stiffness_range, damping_ratio_range, stiffness_levels, damping_ratio_levels,
                                     contact_mass_kg)
    level = floor_compliance_assignment(is_flat, fraction, len(levels))

    def material(path, k=0.0, d=0.0):
        cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0, friction_combine_mode="multiply",
            restitution_combine_mode="multiply", compliant_contact_stiffness=k, compliant_contact_damping=d)
        cfg.func(path, cfg)
        return path

    rigid_path = material("/World/PhysicsMaterials/floor_rigid")
    compliant_paths = [material(f"/World/PhysicsMaterials/floor_compliant_{j}", k, d) for j, (k, d) in enumerate(levels)]

    stage = env.scene.stage
    leaf = tcfg.prim_path.rsplit("/", 1)[-1]   # the terrain prim under each env, e.g. "TerrainMesh"
    stiffness, damping, top = (torch.zeros(env.num_envs) for _ in range(3))
    for i in range(env.num_envs):
        if not is_flat[i]:
            continue
        plate = stage.GetPrimAtPath(f"{env.scene.env_prim_paths[i]}/{leaf}")
        colliders = [p for p in Usd.PrimRange(plate) if p.HasAPI(UsdPhysics.CollisionAPI)]
        if not colliders:
            raise ValueError(f"randomize_floor_compliance: no collider under {plate.GetPath()}")
        translate = next(op for op in UsdGeom.Xformable(plate).GetOrderedXformOps()
                         if op.GetOpType() == UsdGeom.XformOp.TypeTranslate)
        if level[i] >= 0:
            t = translate.Get()
            translate.Set(Gf.Vec3d(t[0], t[1], t[2] + raise_m))
            stiffness[i], damping[i] = levels[level[i]]
        for c in colliders:
            sim_utils.bind_physics_material(str(c.GetPath()), compliant_paths[level[i]] if level[i] >= 0 else rigid_path,
                                            stage=stage)
        # Plate top in the env frame, from the collider's points (verifies the placement; the plate is invisible, which
        # a bounding-box cache would skip).
        now = Usd.TimeCode.Default()
        origin_z = UsdGeom.Xformable(stage.GetPrimAtPath(env.scene.env_prim_paths[i])).ComputeLocalToWorldTransform(
            now).ExtractTranslation()[2]
        top[i] = max(UsdGeom.Xformable(c).ComputeLocalToWorldTransform(now).Transform(Gf.Vec3d(*p))[2]
                     for c in colliders for p in UsdGeom.Mesh(c).GetPointsAttr().Get()) - origin_z
    dev = env.device
    env.floor_compliance_level = torch.tensor(level, dtype=torch.long, device=dev)
    env.floor_stiffness, env.floor_damping, env.floor_top = stiffness.to(dev), damping.to(dev), top.to(dev)
    n_flat, n_soft = sum(is_flat), sum(1 for j in level if j >= 0)
    soft_top = top[env.floor_compliance_level.cpu() >= 0]
    rigid_top = top[torch.tensor(is_flat) & (env.floor_compliance_level.cpu() < 0)]
    print(f"[floor] {n_soft} of {n_flat} flat-plate envs ({env.num_envs} total) on a compliant floor, "
          f"{len(levels)} levels (k {stiffness_range[0]:.0f}-{stiffness_range[1]:.0f} N/m, zeta "
          f"{damping_ratio_range[0]}-{damping_ratio_range[1]}); plate top rigid "
          f"{rigid_top.min().item() if len(rigid_top) else float('nan'):+.3f} m, compliant "
          f"{soft_top.min().item() if len(soft_top) else float('nan'):+.3f} m", flush=True)
