#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RPL-style multi-depth ray-casting camera (RPL Algorithm 1) for the student's torso depth image.

A subclass of IsaacLab's ``MultiMeshRayCasterCamera`` that adds robot self-occlusion: each body-link mesh is loaded
once in its link frame, posed every step from the robot's ``body_link_pose_w``, ray-cast in a second launch that
starts ``body_near_clip`` ahead of the lens, and min'd per pixel with the stock terrain + object ray-cast.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCamera, MultiMeshRayCasterCameraCfg
from isaaclab.sensors.ray_caster.multi_mesh_ray_caster import MultiMeshRayCaster
from isaaclab.sensors.ray_caster.ray_cast_utils import obtain_world_pose_from_view
from isaaclab.utils import configclass
from isaaclab.utils.warp import raycast_dynamic_meshes


class RPLDepthCamera(MultiMeshRayCasterCamera):
    """Multi-mesh ray-casting depth camera with robot self-occlusion.

    The body-link meshes (``cfg.body_prim_expr``) are posed from the robot asset, which a startup event hands over
    with :meth:`set_robot_asset`. Only the ``distance_to_image_plane`` output is supported.
    """

    cfg: RPLDepthCameraCfg

    def __init__(self, cfg: RPLDepthCameraCfg):
        super().__init__(cfg)
        if set(cfg.data_types) != {"distance_to_image_plane"} or getattr(cfg, "update_mesh_ids", False):
            raise ValueError(
                f"RPLDepthCamera supports only data_types=['distance_to_image_plane']; got "
                f"data_types={cfg.data_types}, update_mesh_ids={getattr(cfg, 'update_mesh_ids', False)}. "
                "Use IsaacLab's MultiMeshRayCasterCamera for other outputs."
            )
        # robot-body (self-occlusion) state, populated in _initialize_warp_meshes / set_robot_asset
        self._rpl_n_body = 0
        self._rpl_body_mesh_ids_wp = None  # (num_envs, n_body) shared body-mesh ids (2nd launch)
        self._rpl_body_pos_w: torch.Tensor | None = None  # (num_envs, n_body, 3) per-link world pos
        self._rpl_body_ori_w: torch.Tensor | None = None  # (num_envs, n_body, 4) wxyz
        self._rpl_body_link_names: list[str] = []
        self._rpl_body_meshes: list = []  # keep wp.Mesh refs alive (else GC frees the ids)
        self._rpl_robot = None  # robot Articulation, set by set_robot_asset
        self._rpl_body_col_idx: torch.Tensor | None = None
        # Tracked dynamic objects (the push box) keyed by RaycastTargetCfg.prim_expr, posed from their physics
        # view; empty = the parent's XformPrim (usdrt) pose source, single-GPU only.
        self._rpl_obj_assets: dict = {}

    """
    Mesh setup.
    """

    def _initialize_warp_meshes(self):
        # terrain + dynamic objects (the standard targets in cfg.mesh_prim_paths)
        MultiMeshRayCaster._initialize_warp_meshes(self)
        # robot body links (self-occlusion) — loaded directly, NOT via a RaycastTargetCfg
        if self.cfg.body_prim_expr:
            self._rpl_load_body_meshes()

    def _rpl_load_body_meshes(self):
        """Load each robot body-link mesh once, in its link frame (posed each step in ``_rpl_update_body_poses``)."""
        import numpy as np
        import trimesh

        import isaaclab.sim as sim_utils
        from isaaclab.utils.math import matrix_from_quat
        from isaaclab.utils.mesh import (
            PRIMITIVE_MESH_TYPES,
            create_trimesh_from_geom_mesh,
            create_trimesh_from_geom_shape,
        )
        from isaaclab.utils.warp import convert_to_warp_mesh

        # enumerate env_0's link prims (meshes are identical across envs → load once, share)
        expr0 = self.cfg.body_prim_expr.format(ENV_REGEX_NS="/World/envs/env_.*").replace("env_.*", "env_0")
        link_prims = sim_utils.find_matching_prims(expr0)
        body_ids: list[int] = []
        body_names: list[str] = []
        for link_prim in link_prims:
            mesh_prims = sim_utils.get_all_matching_child_prims(
                link_prim.GetPath(), lambda p: p.GetTypeName() in PRIMITIVE_MESH_TYPES + ["Mesh"]
            )
            if not mesh_prims:
                continue  # frame-only link with no geometry
            tris = []
            for mp in mesh_prims:
                tm = create_trimesh_from_geom_mesh(mp) if mp.GetTypeName() == "Mesh" else create_trimesh_from_geom_shape(mp)
                tm.apply_scale(sim_utils.resolve_prim_scale(mp))
                # bake the mesh into the LINK frame so the per-step pose is exactly the link pose
                rel_pos, rel_quat = sim_utils.resolve_prim_pose(mp, link_prim)
                transform = np.eye(4)
                transform[:3, :3] = matrix_from_quat(torch.tensor(rel_quat, dtype=torch.float32)).numpy()
                transform[:3, 3] = np.asarray(rel_pos, dtype=np.float64)
                tm.apply_transform(transform)
                tris.append(tm)
            tm = tris[0] if len(tris) == 1 else trimesh.util.concatenate(tris)
            wp_mesh = convert_to_warp_mesh(tm.vertices, tm.faces, device=self.device)
            self._rpl_body_meshes.append(wp_mesh)  # hold a ref so wp.Mesh.id stays valid
            body_ids.append(wp_mesh.id)
            body_names.append(link_prim.GetName())

        n_body = len(body_ids)
        if n_body == 0:
            print(
                f"[RPLDepthCamera] WARNING: body_prim_expr {self.cfg.body_prim_expr!r} matched no link "
                "meshes under env_0 — robot self-occlusion DISABLED. Verify the expr against the robot USD."
            )
            self._rpl_n_body = 0
            return

        # A separate mesh set (the parent's terrain + object arrays stay untouched), rendered in a second launch.
        # The meshes are shared across envs (same id row per env); the poses are per env.
        self._rpl_body_mesh_ids_wp = wp.array2d(
            [list(body_ids)] * self._num_envs, dtype=wp.uint64, device=self.device
        )
        # poses start FAR (1e6) so body produces no hit until real link poses are written.
        self._rpl_body_pos_w = torch.full((self._num_envs, n_body, 3), 1.0e6, device=self.device)
        self._rpl_body_ori_w = torch.zeros((self._num_envs, n_body, 4), device=self.device)
        self._rpl_body_ori_w[..., 0] = 1.0  # identity quat (wxyz)
        self._rpl_body_link_names = body_names
        self._rpl_n_body = n_body
        preview = ", ".join(body_names[:6]) + ("…" if n_body > 6 else "")
        print(f"[RPLDepthCamera] loaded {n_body} robot body-link meshes for self-occlusion ({preview}).")

    def set_robot_asset(self, robot) -> None:
        """Hand over the robot Articulation whose per-link poses drive self-occlusion (called by a startup event).

        Reading the asset's own physics view is per-rank, so this is multi-GPU-safe.
        """
        self._rpl_robot = robot

    def set_dynamic_object_assets(self, mapping: dict) -> None:
        """Hand over the scene RigidObjects of the tracked dynamic targets, keyed by ``RaycastTargetCfg.prim_expr``.

        Their poses are then read from the per-rank physics view (``root_pose_w``) instead of the parent's
        ``XformPrim`` view, which goes through usdrt on ``cuda:0`` and fails on the other ranks of a multi-GPU run.
        """
        self._rpl_obj_assets = dict(mapping)

    def reset(self, env_ids: Sequence[int] | None = None):
        if self._rpl_n_body > 0 and self._rpl_robot is None:
            raise RuntimeError(
                "RPLDepthCamera: no robot asset bound for self-occlusion. Call set_robot_asset() from a startup "
                "event (see _bind_rpl_depth_robot in motion_tracking_scene_distill_cfg.py)."
            )
        super().reset(env_ids)

    def _rpl_update_body_poses(self):
        """Write the per-link world poses of the body meshes from the robot asset.

        Until the startup event binds the robot (the observation manager probes the sensor before it), the body
        meshes stay far away and the image shows no body.
        """
        if self._rpl_robot is None:
            return
        if self._rpl_body_col_idx is None and not self._rpl_map_body_to_asset():
            return
        pose = self._rpl_robot.data.body_link_pose_w  # (num_envs, num_bodies, 7), quat wxyz
        sel = pose[:, self._rpl_body_col_idx]
        self._rpl_body_pos_w[:] = sel[..., :3]
        self._rpl_body_ori_w[:] = sel[..., 3:7]

    def _rpl_map_body_to_asset(self) -> bool:
        """Map the loaded body-mesh link names to robot body indices (once). Disables self-occlusion
        (returns False) if a name isn't in ``robot.data.body_names``."""
        names = list(self._rpl_robot.data.body_names)
        missing = [n for n in self._rpl_body_link_names if n not in names]
        if missing:
            print(f"[RPLDepthCamera] body link(s) {missing} not in robot.data.body_names; self-occlusion DISABLED.")
            self._rpl_n_body = 0
            return False
        self._rpl_body_col_idx = torch.tensor(
            [names.index(n) for n in self._rpl_body_link_names], device=self.device, dtype=torch.long
        )
        print(f"[RPLDepthCamera] self-occlusion ACTIVE (robot asset link poses): {self._rpl_n_body} body meshes.")
        return True

    """
    Per-step buffer fill.
    """

    def _update_buffers_impl(self, env_ids: Sequence[int] | torch.Tensor | None):
        """Fill the sensor buffers: the parent's ray-cast plus the robot body. Always runs over all envs (depth is
        stateless per frame)."""
        env_ids = self._ALL_INDICES
        self._update_ray_infos(env_ids)
        self._frame[env_ids] += 1

        # Tracked dynamic objects (the push box): write their columns of _mesh_positions_w / _mesh_orientations_w.
        mesh_idx = 0
        for view, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
            if not target_cfg.track_mesh_transforms:
                mesh_idx += self._num_meshes_per_env[target_cfg.prim_expr]
                continue
            # A bound RigidObject is posed from its per-rank physics view (multi-GPU-safe); otherwise the parent's
            # view is used (single-GPU only, see set_dynamic_object_assets).
            asset = self._rpl_obj_assets.get(target_cfg.prim_expr)
            if asset is not None:
                root = asset.data.root_pose_w  # (num_envs, 7), quat wxyz
                pos_w, ori_w = root[:, :3], root[:, 3:7]
                if target_cfg.prim_expr in MultiMeshRayCaster.mesh_offsets:
                    pos_offset, ori_offset = MultiMeshRayCaster.mesh_offsets[target_cfg.prim_expr]
                    pos_w = pos_w - pos_offset
                    ori_w = math_utils.quat_mul(ori_offset.expand(ori_w.shape[0], -1), ori_w)
                count = view.count
                if count != 1:
                    count = count // self._num_envs
                # broadcast the per-env object pose across this target's mesh column(s)
                self._mesh_positions_w[:, mesh_idx : mesh_idx + count] = pos_w.unsqueeze(1)
                self._mesh_orientations_w[:, mesh_idx : mesh_idx + count] = ori_w.unsqueeze(1)
                mesh_idx += count
                continue
            pos_w, ori_w = obtain_world_pose_from_view(view, None)
            pos_w = pos_w.squeeze(0) if len(pos_w.shape) == 3 else pos_w
            ori_w = ori_w.squeeze(0) if len(ori_w.shape) == 3 else ori_w
            if target_cfg.prim_expr in MultiMeshRayCaster.mesh_offsets:
                pos_offset, ori_offset = MultiMeshRayCaster.mesh_offsets[target_cfg.prim_expr]
                pos_w = pos_w - pos_offset
                ori_w = math_utils.quat_mul(ori_offset.expand(ori_w.shape[0], -1), ori_w)
            count = view.count
            if count != 1:  # per-env meshes
                count = count // self._num_envs
                pos_w = pos_w.view(self._num_envs, count, 3)
                ori_w = ori_w.view(self._num_envs, count, 4)
            self._mesh_positions_w[:, mesh_idx : mesh_idx + count] = pos_w
            self._mesh_orientations_w[:, mesh_idx : mesh_idx + count] = ori_w
            mesh_idx += count

        if self._rpl_n_body > 0:
            self._rpl_update_body_poses()

        # Two launches, per-pixel min: (1) terrain + objects with the parent's arrays, (2) the robot body.
        # raycast_dynamic_meshes returns +inf for misses.
        self.ray_hits_w[env_ids], ray_depth, _, _, _ = raycast_dynamic_meshes(
            self._ray_starts_w[env_ids],
            self._ray_directions_w[env_ids],
            mesh_ids_wp=self._mesh_ids_wp,
            max_dist=self.cfg.max_distance,
            mesh_positions_w=self._mesh_positions_w[env_ids],
            mesh_orientations_w=self._mesh_orientations_w[env_ids],
            return_distance=True,
        )
        if self._rpl_n_body > 0:
            # Body rays start body_near_clip ahead of the lens, so they skip the camera mount right at the lens.
            near = float(getattr(self.cfg, "body_near_clip", 0.0) or 0.0)
            body_starts = self._ray_starts_w[env_ids]
            if near > 0.0:
                body_starts = body_starts + near * self._ray_directions_w[env_ids]
            _, body_depth, _, _, _ = raycast_dynamic_meshes(
                body_starts,
                self._ray_directions_w[env_ids],
                mesh_ids_wp=self._rpl_body_mesh_ids_wp,
                max_dist=self.cfg.max_distance,
                mesh_positions_w=self._rpl_body_pos_w[env_ids],
                mesh_orientations_w=self._rpl_body_ori_w[env_ids],
                return_distance=True,
            )
            if near > 0.0:
                body_depth = body_depth + near  # distance from the lens
            ray_depth = torch.minimum(ray_depth, body_depth)

        # Output, as in MultiMeshRayCasterCamera: the ray depth projected on the camera z-axis.
        distance_to_image_plane = (
            math_utils.quat_apply(
                math_utils.quat_inv(self._data.quat_w_world[env_ids]).repeat(1, self.num_rays),
                (ray_depth[:, :, None] * self._ray_directions_w[env_ids]),
            )
        )[:, :, 0]
        if self.cfg.depth_clipping_behavior == "max":
            distance_to_image_plane = torch.clip(distance_to_image_plane, max=self.cfg.max_distance)
            distance_to_image_plane[torch.isnan(distance_to_image_plane)] = self.cfg.max_distance
        elif self.cfg.depth_clipping_behavior == "zero":
            distance_to_image_plane[distance_to_image_plane > self.cfg.max_distance] = 0.0
            distance_to_image_plane[torch.isnan(distance_to_image_plane)] = 0.0
        self._data.output["distance_to_image_plane"][env_ids] = distance_to_image_plane.view(
            -1, *self.image_shape, 1
        )


@configclass
class RPLDepthCameraCfg(MultiMeshRayCasterCameraCfg):
    """Configuration for :class:`RPLDepthCamera`.

    ``mesh_prim_paths`` lists the terrain and dynamic-object targets as in :class:`MultiMeshRayCasterCameraCfg`; the
    robot body is set by ``body_prim_expr`` instead (an articulation's per-link poses cannot be tracked through a
    ``RaycastTargetCfg``).
    """

    class_type: type = RPLDepthCamera

    body_near_clip: float = 0.3
    """Distance (m) ahead of the lens at which the body rays start, so they skip the camera mount while the lower
    body and arms beyond it are kept. Applies to the body launch only; 0.0 disables it."""

    body_prim_expr: str | None = "{ENV_REGEX_NS}/Robot/.*_link"
    """Prim-expr of the robot body-link prims whose meshes occlude the camera; ``None`` disables self-occlusion.
    A no-match logs a warning and renders terrain + objects only."""
