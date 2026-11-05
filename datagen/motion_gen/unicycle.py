#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Accel-limited unicycle root trajectory along an A* corridor (the walk of the sit task)."""

from __future__ import annotations

import math
import os

import numpy as np

from common.placement import is_circle_free

FPS = 30
DT = 1.0 / FPS
V_MAX = 0.7  # m/s   comfortable indoor walk (hardware max ~2)
OMEGA_MAX = 1.2  # rad/s (~69 deg/s) stable stepping turn
A_MAX = 1.0  # m/s^2  linear accel limit
ALPHA_MAX = 3.0  # rad/s^2 angular accel limit
K_OMEGA = 2.5  # heading-error gain
# start decelerating this far from the stop point; 0.2 keeps the robot striding into the turn
SLOW_RADIUS = float(os.environ.get("SLOW_RADIUS", "0.2"))
STOP_TOL = 0.03  # arrival tolerance (avoids asymptotic creep at the ring)


def wrap(a):
    """Angle wrapped to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def _resample(pts, step):
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < 1e-6:
        return pts[:1]
    n = max(2, int(math.ceil(total / step)) + 1)
    s = np.linspace(0.0, total, n)
    return np.stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])], axis=1)


def simulate_corridor(
    spawn_xy,
    yaw0,
    corridor_xy,
    *,
    occ,
    robot_radius=0.30,
    lookahead=0.6,
    goal_face_dist=0.9,
    v_max=V_MAX,
    max_steps=1600,
):
    """Accel-limited unicycle following ``corridor_xy`` to its end; ``[T, 3]`` (x, y, yaw) at 30 fps."""
    corr = _resample(corridor_xy, step=max(0.1, v_max * DT))
    if len(corr) < 2:
        corr = np.asarray([spawn_xy, corridor_xy[-1]], dtype=float)
    goal = np.asarray(corr[-1], dtype=float)
    pos = np.asarray(spawn_xy, dtype=float).copy()
    th = float(yaw0)
    v = omega = 0.0
    dv, dom = A_MAX * DT, ALPHA_MAX * DT
    out = [(pos[0], pos[1], th)]
    seg_i = 0
    for _ in range(max_steps):
        to_goal = float(np.hypot(*(goal - pos)))
        if to_goal <= STOP_TOL + 0.02:
            break  # the walk flows into the turn segment
        while seg_i < len(corr) - 1 and np.hypot(*(corr[seg_i] - pos)) < lookahead:
            seg_i += 1
        target = goal if to_goal < goal_face_dist else corr[min(seg_i, len(corr) - 1)]
        e = wrap(math.atan2(target[1] - pos[1], target[0] - pos[0]) - th)
        omega_cmd = max(-OMEGA_MAX, min(OMEGA_MAX, K_OMEGA * e))
        decel = max(0.0, min(1.0, to_goal / SLOW_RADIUS))
        v_cmd = v_max * max(0.0, math.cos(e)) * decel
        omega += max(-dom, min(dom, omega_cmd - omega))
        v += max(-dv, min(dv, v_cmd - v))
        th = wrap(th + omega * DT)
        nxt = pos + v * np.array([math.cos(th), math.sin(th)]) * DT
        if not is_circle_free(occ, nxt, robot_radius):
            break  # blocked: stop here
        pos = nxt
        out.append((pos[0], pos[1], th))
    return np.asarray(out)
