#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-GPU-safe ``MultiMeshRayCaster`` (the object-aware height scanner).

IsaacLab's ``MultiMeshRayCaster`` poses each tracked dynamic mesh (the push box) through an ``XformPrim`` view, which
goes through usdrt on ``cuda:0`` and fails on the other ranks of a multi-GPU run (the depth camera solves the same
problem in ``RPLDepthCamera.set_dynamic_object_assets``). After the stock ``_initialize_warp_meshes``, this subclass
swaps every tracked target that has a physics prim to a per-rank PhysX view (``RigidBodyView`` / ``ArticulationView``),
as upstream ``RayCaster._obtain_trackable_prim_view`` does. Only the pose source changes; targets without a physics
prim (static terrain) keep their ``XformPrim`` view.
"""

from __future__ import annotations

from isaaclab.sensors.ray_caster.multi_mesh_ray_caster import MultiMeshRayCaster


def _resolve_physics_view_and_offset(sensor, target_prim_path: str):
    """Walk up from ``target_prim_path`` to the first physics prim and build a per-rank physics view.

    Mirrors upstream IsaacLab ``RayCaster._obtain_trackable_prim_view``: returns
    ``(physx_view, (pos_offset, ori_offset))`` where the offset is the tracked-prim pose relative to the
    physics prim (so the existing ``mesh_offsets`` correction in ``_update_buffers_impl`` stays valid),
    or ``None`` if no physics prim is found / the mesh↔physics prim counts differ (e.g. a multi-link
    articulation) — in which case the caller keeps the original ``XformPrim`` view unchanged.
    """
    import torch

    import isaaclab.sim as sim_utils
    from pxr import UsdPhysics

    matches = sim_utils.find_matching_prims(target_prim_path)
    if not matches:
        return None
    current_prim = matches[0]
    current_path_expr = target_prim_path

    view = None
    while True:
        # As upstream, an applied but disabled API counts as present.
        if current_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            view = sensor._physics_sim_view.create_articulation_view(current_path_expr.replace(".*", "*"))
            break
        if current_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            view = sensor._physics_sim_view.create_rigid_body_view(current_path_expr.replace(".*", "*"))
            break
        parent = current_prim.GetParent()
        if parent is None or not parent.IsValid():
            return None  # reached the stage root with no physics API → keep XformPrim (e.g. terrain)
        current_path_expr = current_path_expr.rsplit("/", 1)[0]
        current_prim = parent

    if view is None or getattr(view, "count", 0) == 0:
        return None

    # relative pose between the tracked (target) prims and the physics-view prims
    mesh_prims = sim_utils.find_matching_prims(target_prim_path)
    view_prims = sim_utils.find_matching_prims(current_path_expr)
    if len(mesh_prims) != len(view_prims):
        return None  # count mismatch (e.g. many links → one articulation root) → don't swap

    positions, quaternions = [], []
    for mesh_prim, view_prim in zip(mesh_prims, view_prims):
        pos, quat = sim_utils.resolve_prim_pose(mesh_prim, view_prim)
        positions.append(torch.tensor(pos, dtype=torch.float32, device=sensor.device))
        quaternions.append(torch.tensor(quat, dtype=torch.float32, device=sensor.device))
    return view, (torch.stack(positions), torch.stack(quaternions))


def _swap_tracked_views_to_physics(sensor) -> None:
    """Swap each tracked target's ``XformPrim`` view to a per-rank physics view (idempotent).

    Updates the instance ``_mesh_views`` and the shared ``MultiMeshRayCaster.mesh_views`` / ``mesh_offsets`` class
    variables, which sibling sensors reuse."""
    import omni.physics.tensors.impl.api as physx

    swapped = []
    for i, target_cfg in enumerate(sensor._raycast_targets_cfg):
        if not target_cfg.track_mesh_transforms:
            continue
        view = sensor._mesh_views[i]
        if isinstance(view, (physx.ArticulationView, physx.RigidBodyView)):
            continue  # already a physics view
        resolved = _resolve_physics_view_and_offset(sensor, target_cfg.prim_expr)
        if resolved is None:
            continue  # no physics prim (static terrain) or count mismatch → leave as-is
        new_view, offset = resolved
        sensor._mesh_views[i] = new_view
        MultiMeshRayCaster.mesh_views[target_cfg.prim_expr] = new_view
        MultiMeshRayCaster.mesh_offsets[target_cfg.prim_expr] = offset
        swapped.append(f"{target_cfg.prim_expr} -> {type(new_view).__name__}")
    if swapped:
        print(f"[mgpu_ray_caster] swapped {len(swapped)} dynamic target(s) to per-rank physics views: {swapped}")


class MultiMeshRayCasterMGPU(MultiMeshRayCaster):
    """``MultiMeshRayCaster`` (height scanner) that tracks dynamic objects via per-rank physics views."""

    def _initialize_warp_meshes(self):
        super()._initialize_warp_meshes()
        _swap_tracked_views_to_physics(self)

