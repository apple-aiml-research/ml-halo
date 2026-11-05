#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Third-person-view (TPV) camera placement for a trajectory inside a room."""

from __future__ import annotations

import math

import numpy as np

from common.placement import OccGrid, is_circle_free, is_segment_clear_with_bubble


WALL_CLEARANCE_M = 0.30
CEILING_CLEARANCE_M = 0.30


def _in_room(occ: OccGrid, xy, clearance: float = WALL_CLEARANCE_M) -> bool:
    ny, nx = occ.shape
    x0, y0 = occ.x_min + clearance, occ.y_min + clearance
    x1 = occ.x_min + nx * occ.resolution - clearance
    y1 = occ.y_min + ny * occ.resolution - clearance
    return x0 <= float(xy[0]) <= x1 and y0 <= float(xy[1]) <= y1


def _best_in_room_eye(
    occ: OccGrid,
    p: np.ndarray,
    centroid: np.ndarray,
    dist: float,
    base_ang: float,
    eye_radius: float,
    step: float = 0.2,
):
    """Free in-room eye with the best view of the trajectory, or None."""
    ny, nx = occ.shape
    xs = np.arange(occ.x_min + WALL_CLEARANCE_M, occ.x_min + nx * occ.resolution - WALL_CLEARANCE_M + 1e-9, step)
    ys = np.arange(occ.y_min + WALL_CLEARANCE_M, occ.y_min + ny * occ.resolution - WALL_CLEARANCE_M + 1e-9, step)
    samples = p[np.linspace(0, len(p) - 1, min(len(p), 12)).astype(int)]
    best = None
    for y in ys:
        for x in xs:
            eye = np.array([x, y])
            if not is_circle_free(occ, eye, eye_radius):
                continue
            los_c = is_segment_clear_with_bubble(occ, eye, centroid, bubble_radius=0.15)
            vis = sum(is_segment_clear_with_bubble(occ, eye, q, bubble_radius=0.15) for q in samples)
            d = float(np.linalg.norm(eye - centroid))
            ang = abs(
                (math.atan2(eye[1] - centroid[1], eye[0] - centroid[0]) - base_ang + math.pi) % (2 * math.pi) - math.pi
            )
            key = (0 if los_c else 1, -vis, round(abs(d - dist), 6), round(ang, 6))
            if best is None or key < best[0]:
                best = (key, eye)
    return None if best is None else best[1]


def pick_tpv_pose(
    occ: OccGrid,
    traj_xy,
    floor_z: float,
    *,
    fov_deg: float = 55.0,
    height: float = 1.7,
    target_z: float = 0.4,
    margin: float = 1.35,
    eye_radius: float = 0.25,
    n_angles: int = 48,
    ceiling_z: float | None = None,
):
    """Return (eye_xyz, target_xyz) for a TPV that frames the whole ``traj_xy``."""
    p = np.asarray(traj_xy, dtype=float)[:, :2]
    centroid = p.mean(axis=0)
    r_bound = float(np.max(np.linalg.norm(p - centroid, axis=1))) + 0.4
    dist = margin * r_bound / max(math.tan(math.radians(fov_deg) / 2), 1e-3)
    dist = float(np.clip(dist, 1.8, 6.0))

    spawn = p[0]
    base = spawn - centroid
    base_ang = math.atan2(base[1], base[0]) if np.linalg.norm(base) > 1e-3 else 0.0

    def _ok(eye_xy):
        return (
            _in_room(occ, eye_xy)
            and is_circle_free(occ, eye_xy, eye_radius)
            and is_segment_clear_with_bubble(occ, eye_xy, centroid, bubble_radius=0.15)
        )

    best = None
    for da in sorted(range(0, 181, max(1, 360 // n_angles)), key=lambda d: d):  # 0,±… widening fan
        for sign in (1, -1):
            ang = base_ang + math.radians(da) * sign
            for d in (dist, dist * 0.8, dist * 1.2):
                eye = centroid + d * np.array([math.cos(ang), math.sin(ang)])
                if _ok(eye):
                    score = da + 0.5 * float(np.linalg.norm(eye - spawn))
                    if best is None or score < best[0]:
                        best = (score, eye)
            if best is not None and da == 0:
                break  # exact preferred dir is free
        if best is not None and da <= 30:
            break  # good enough near-preferred
    if best is not None:
        ex, ey = best[1]
    else:
        alt = _best_in_room_eye(occ, p, centroid, dist, base_ang, eye_radius)
        if alt is None:  # no free cell in the room at all: room centre
            ny, nx = occ.shape
            alt = np.array([occ.x_min + nx * occ.resolution / 2, occ.y_min + ny * occ.resolution / 2])
        ex, ey = alt

    ez = floor_z + height
    if ceiling_z is not None:
        ez = min(ez, ceiling_z - CEILING_CLEARANCE_M)
    eye = [float(ex), float(ey), float(ez)]
    target = [float(centroid[0]), float(centroid[1]), float(floor_z + target_z)]
    return eye, target


def select_infinigen_tpv(analysis: dict, traj_xy, floor_z: float, *, standoff_m: float = 3.0, target_z: float = 0.5):
    """TPV (eye, target): the analysed view nearest ``standoff_m`` with line of sight, else ``pick_tpv_pose``."""
    p = np.asarray(traj_xy, dtype=float)[:, :2]
    midxy = (float(p[:, 0].mean()), float(p[:, 1].mean()))
    target = [midxy[0], midxy[1], float(floor_z + target_z)]

    occb = analysis["occupancy"]
    occ = OccGrid(grid=np.asarray(occb["grid"]) != 0, x_min=0.0, y_min=0.0, resolution=float(occb["cell"]))
    views = analysis.get("tpv_views") or []
    clear = [
        v
        for v in views
        if is_segment_clear_with_bubble(occ, (float(v["pos"][0]), float(v["pos"][1])), midxy, bubble_radius=0.15)
    ]
    if clear:
        best = min(clear, key=lambda v: abs(math.hypot(v["pos"][0] - midxy[0], v["pos"][1] - midxy[1]) - standoff_m))
        eye = [float(best["pos"][0]), float(best["pos"][1]), float(best["pos"][2])]
        return eye, target
    room_h = float((analysis.get("room") or {}).get("H") or 0.0)
    ceiling = floor_z + room_h if 1.5 < room_h < 10.0 else None  # ignore a missing / broken height
    return pick_tpv_pose(occ, p, floor_z, target_z=target_z, ceiling_z=ceiling)
