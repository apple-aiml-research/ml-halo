#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Scene-conditioned sit keyframes: relocate the committed sit body-poses onto a"""

from __future__ import annotations

import os

import numpy as np

from motion_gen.episode_spec import EpisodeSpec

_KF_DIR = os.path.dirname(os.path.abspath(__file__))
SEAT_KF_TEMPLATE = os.path.join(_KF_DIR, "kf_world_6772_sit.npy")
STANDING_POSE = os.path.join(_KF_DIR, "g1_standing_pose.npy")

K_STAND, K_WALK_END, K_TURN_END, K_STEP, K_SIT = 0, 1, 2, 3, 4


def _yaw_of(q) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def set_root_pose(qpos36, xy, yaw) -> np.ndarray:
    """Copy of a [36] pose with root xy set and yaw SET (preserving pitch/roll"""
    p = np.asarray(qpos36, dtype=np.float64).copy()
    delta = yaw - _yaw_of(p[3:7])
    p[3:7] = _quat_mul(_yaw_quat(delta), p[3:7])
    p[0], p[1] = float(xy[0]), float(xy[1])
    return p


def build_sit_kfs_from_spec(
    spec: EpisodeSpec, seat_template: np.ndarray | None = None, stand_pose: np.ndarray | None = None
) -> np.ndarray:
    """Return 5 world-frame ``[36]`` keyframes ``[stand, walk_end, turn_end,"""
    if seat_template is None:
        seat_template = np.load(SEAT_KF_TEMPLATE).astype(np.float64)
    if stand_pose is None:
        stand_pose = np.load(STANDING_POSE).astype(np.float64)

    seat_xy = np.asarray(spec.seat_target_xyz[:2], dtype=float)
    seat_yaw = float(spec.seat_yaw_rad)
    spawn_xy = np.asarray(spec.robot.xy, dtype=float)
    spawn_yaw = float(spec.robot.yaw_rad)
    if spec.path_xy:
        stop_xy = np.asarray(spec.path_xy[-1], dtype=float)
        approach_yaw = float(spec.path_yaw[-1]) if spec.path_yaw else spawn_yaw
    else:  # no path planned → walk straight toward the seat
        stop_xy = seat_xy
        approach_yaw = spawn_yaw
    step_xy = 0.5 * (stop_xy + seat_xy)

    kf_stand = set_root_pose(stand_pose, spawn_xy, spawn_yaw)
    kf_walk_end = set_root_pose(seat_template[K_WALK_END], stop_xy, approach_yaw)
    kf_turn_end = set_root_pose(seat_template[K_TURN_END], stop_xy, seat_yaw)
    kf_step = set_root_pose(seat_template[K_STEP], step_xy, seat_yaw)
    kf_sit = set_root_pose(seat_template[K_SIT], seat_xy, seat_yaw)
    return np.stack([kf_stand, kf_walk_end, kf_turn_end, kf_step, kf_sit])
