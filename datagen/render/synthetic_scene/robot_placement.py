#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Robot placement helpers for synthetic scenes."""

from __future__ import annotations

import math

import numpy as np


def _heading0(q):
    """Walk heading of a trajectory: start->end direction, or the frame-0 facing if it moves < 0.5 m."""
    from scipy.spatial.transform import Rotation as R

    disp = q[-1, :2] - q[0, :2]
    if np.linalg.norm(disp) > 0.5:
        return math.atan2(disp[1], disp[0])
    q0 = q[0, 3:7]
    fwd0 = R.from_quat([q0[1], q0[2], q0[3], q0[0]]).apply([1, 0, 0])
    return math.atan2(fwd0[1], fwd0[0])


def place_trajectory(qpos_traj, spawn_pos_zup, spawn_yaw, floor_z):
    """Rigidly place a raw Frame-3 kimodo trajectory into an infinigen room (xy + yaw, z lifted by floor_z)."""
    from scipy.spatial.transform import Rotation as R

    q = np.asarray(qpos_traj, dtype=np.float64).copy()
    dyaw = spawn_yaw - _heading0(q)
    origin = q[0, :2].copy()
    cz, sz = math.cos(dyaw), math.sin(dyaw)
    Rz = np.array([[cz, -sz], [sz, cz]])
    q[:, :2] = (q[:, :2] - origin) @ Rz.T + np.array(spawn_pos_zup[:2])
    rot = R.from_euler("z", dyaw) * R.from_quat(np.column_stack([q[:, 4], q[:, 5], q[:, 6], q[:, 3]]))  # wxyz->xyzw
    xyzw = rot.as_quat()
    q[:, 3] = xyzw[:, 3]
    q[:, 4:7] = xyzw[:, :3]
    q[:, 2] += floor_z
    return q


def rigid_place_xy(pts_xy, qpos_traj, spawn_pos_zup, spawn_yaw):
    """Apply the rigid xy+yaw transform of ``place_trajectory`` to extra points (e.g. the target)."""
    q = np.asarray(qpos_traj, dtype=np.float64)
    dyaw = spawn_yaw - _heading0(q)
    origin = q[0, :2].copy()
    cz, sz = math.cos(dyaw), math.sin(dyaw)
    Rz = np.array([[cz, -sz], [sz, cz]])
    pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 2)
    return (pts - origin) @ Rz.T + np.array(spawn_pos_zup[:2])


def build_scene_clearance(inf_analysis, cell=None):
    """Rebuild the (occ, clearance) grids from a scene_analysis dict at render time."""
    from types import SimpleNamespace
    from render.synthetic_scene.analyze_scene import build_occupancy

    room = inf_analysis["room"]
    floor_z = float(inf_analysis["floor_z"])
    if cell is None:
        cell = float(inf_analysis.get("occupancy", {}).get("cell", 0.10))
    assets = [SimpleNamespace(**a) for a in inf_analysis.get("assets", [])]
    return build_occupancy(assets, room["W"], room["D"], floor_z, cell=cell)


def solve_path_placement(
    clearance,
    cell,
    qpos,
    target_xy,
    *,
    robot_radius=0.35,
    target_radius=0.30,
    seed=0,
    n_yaw=24,
    path_stride=3,
    max_cell_tries=800,
):
    """Seeded search for a (spawn_xy, spawn_yaw) whose placed walk and target both have clearance; None if none."""
    q = np.asarray(qpos, dtype=np.float64)
    ny, nx = clearance.shape

    heading0 = _heading0(q)
    origin = q[0, :2].copy()
    rel_path = q[::path_stride, :2] - origin  # [P,2] pre-rotation
    tgt_rel = np.asarray(target_xy, dtype=np.float64).reshape(2) - origin

    rng = np.random.default_rng(seed)
    free = clearance >= robot_radius
    cand = np.argwhere(free)  # [(j,i), ...]
    if len(cand) == 0:
        return None
    order = rng.permutation(len(cand))[:max_cell_tries]
    yaws = rng.permutation(n_yaw) * (2.0 * math.pi / n_yaw)

    for ci in order:
        j, i = cand[ci]
        sx, sy = (i + 0.5) * cell, (j + 0.5) * cell
        for yaw in yaws:
            dyaw = yaw - heading0
            cz, sz = math.cos(dyaw), math.sin(dyaw)
            Rz = np.array([[cz, -sz], [sz, cz]])
            wp = rel_path @ Rz.T + np.array([sx, sy])
            wt = tgt_rel @ Rz.T + np.array([sx, sy])
            pi = np.floor(wp[:, 0] / cell).astype(int)
            pj = np.floor(wp[:, 1] / cell).astype(int)
            ti = int(math.floor(wt[0] / cell))
            tj = int(math.floor(wt[1] / cell))
            if pi.min() < 0 or pi.max() >= nx or pj.min() < 0 or pj.max() >= ny:
                continue
            if ti < 0 or ti >= nx or tj < 0 or tj >= ny:
                continue
            if clearance[pj, pi].min() < robot_radius:
                continue
            if clearance[tj, ti] < target_radius:
                continue
            return [float(sx), float(sy)], float(yaw)
    return None


def target_footprint_radius(spec):
    """Clearance radius for the spec's target prop: half its scaled XY bbox diagonal + 0.10 m (default 0.30)."""
    _tp = next((p for p in spec.placements if p.role == "target"), None)
    _tr = 0.30
    if _tp is not None and getattr(_tp, "bbox_min", None) and getattr(_tp, "bbox_max", None):
        _sc = getattr(_tp, "scale", 1.0) or 1.0
        _sc = float(_sc[0]) if isinstance(_sc, (list, tuple)) else float(_sc)
        _dx = (_tp.bbox_max[0] - _tp.bbox_min[0]) * _sc
        _dy = (_tp.bbox_max[1] - _tp.bbox_min[1]) * _sc
        _tr = 0.5 * math.hypot(_dx, _dy) + 0.10
    return _tr


def set_robot_spawn(spec, spawn_pos, spawn_yaw, floor_z):
    """Record the placed spawn (xy, yaw, floor z) on ``spec.robot``."""
    spec.robot.xy = [float(spawn_pos[0]), float(spawn_pos[1])]
    spec.robot.yaw_rad = float(spawn_yaw)
    spec.robot.z = float(floor_z)
    return spec


def se2_fit_solve(spec, qpos, inf_clearance, inf_cell, seed):
    """Seeded path-free SE(2) spawn for a canonical sit episode: ``([sx, sy], yaw)`` or None."""
    return solve_path_placement(
        inf_clearance, inf_cell, qpos, list(spec.target_xyz[:2]), target_radius=target_footprint_radius(spec), seed=seed
    )


def se2_fit_apply(spec, qpos, spawn_pos, spawn_yaw, inf_floor_z):
    """Apply a spawn (pos, yaw) to a canonical sit episode: place the qpos, target, seat target and chair."""
    _seat_z = float(spec.target_xyz[2])  # canonical floor_z=0 -> seat height
    _dd = qpos[-1, :2] - qpos[0, :2]
    if float(np.linalg.norm(_dd)) > 0.5:
        _h0 = math.atan2(_dd[1], _dd[0])
    else:
        _q0 = qpos[0]
        _h0 = math.atan2(2 * (_q0[3] * _q0[6] + _q0[4] * _q0[5]), 1 - 2 * (_q0[5] ** 2 + _q0[6] ** 2))
    _dyaw = spawn_yaw - _h0
    _t2 = rigid_place_xy([list(spec.target_xyz[:2])], qpos, spawn_pos, spawn_yaw)[0]
    if spec.seat_target_xyz:
        _st = rigid_place_xy([list(spec.seat_target_xyz[:2])], qpos, spawn_pos, spawn_yaw)[0]
        spec.seat_target_xyz = [
            float(_st[0]),
            float(_st[1]),
            inf_floor_z + float(spec.seat_target_xyz[2] if len(spec.seat_target_xyz) > 2 else 0.0),
        ]
    qpos = place_trajectory(qpos, spawn_pos, spawn_yaw, inf_floor_z)
    spec.target_xyz = [float(_t2[0]), float(_t2[1]), inf_floor_z + _seat_z]
    for _p in spec.placements:
        if _p.role == "target":
            _p.pos_xyz = [float(_t2[0]), float(_t2[1]), inf_floor_z]
            _q = _p.quat_wxyz or [1.0, 0.0, 0.0, 0.0]
            _cur = 2.0 * math.atan2(float(_q[3]), float(_q[0]))  # pure-z quat yaw
            _ny = _cur + _dyaw
            _p.quat_wxyz = [math.cos(_ny / 2.0), 0.0, 0.0, math.sin(_ny / 2.0)]
    if spec.seat_yaw_rad is not None:
        spec.seat_yaw_rad = float(spec.seat_yaw_rad + _dyaw)
    set_robot_spawn(spec, spawn_pos, spawn_yaw, inf_floor_z)
    return spec, qpos, [float(_t2[0]), float(_t2[1])]


def se2_fit_place(spec, qpos, inf_analysis, inf_clearance, inf_cell, inf_floor_z, seed):
    """Motion-first sit placement (``SIT_PLACEMENT_MODE=se2_fit``), used by both render backends."""
    _sol = se2_fit_solve(spec, qpos, inf_clearance, inf_cell, seed)
    fallback = _sol is None
    if _sol is not None:
        (_sx, _sy), _spawn_yaw = _sol
        _spawn_pos = [_sx, _sy, inf_floor_z]
    else:
        _sp = inf_analysis["spawn"]
        _spawn_pos, _spawn_yaw = _sp["pos_zup"], _sp["yaw"]
    spec, qpos, _ = se2_fit_apply(spec, qpos, _spawn_pos, _spawn_yaw, inf_floor_z)
    return spec, qpos, fallback
