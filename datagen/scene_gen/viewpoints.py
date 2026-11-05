#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""viewpoints.py -- sample N camera viewpoints inside a baked room."""

import math
import random

MIN_WALL = 0.8  # m, minimum clearance from any surface around the camera
MIN_HIT = 2.5  # m, closest the look ray may hit geometry
MAX_HIT = 15.0  # m, furthest -- beyond this the camera is staring into a void
CAM_H = 1.45  # m above the floor
PITCH_RANGE = (-20.0, -4.0)  # degrees; negative = downward. Never look up (the ceiling is not baked).
INSET = 0.8  # m, keep sampled camera positions this far inside the room bounds

FRUSTUM_GRID = 5  # 5x5 = 25 probe rays over the frame
FRUSTUM_FOV_DEG = 54.4  # horizontal FOV of the 35 mm / 36 mm sensor the renderers use
MIN_DEPTH_SPREAD = 0.60  # m, required std of frustum hit depths
REACH_W = 1.0
MIN_DEPTH_RANGE = 1.50  # m, required max-min of frustum hit depths
MIN_FRUSTUM_HIT_FRAC = 0.75  # at least this fraction of probe rays must hit something

ORBIT_TILT_DEG = -8.0
ORBIT_MIN_RADIUS = 1.6  # preferred floor radius in a room that has the space
ORBIT_ABS_MIN_RADIUS = 0.35  # hard floor: a tiny bathroom gets a tiny ring, not no ring
ORBIT_WALL_MARGIN = 0.3  # keep the ring (and its centre) this far off every wall
ORBIT_SHRINK = 0.85  # shrink factor when the ring clips furniture
ORBIT_SHRINK_TRIES = 8


def _norm(v):
    n = math.sqrt(sum(c * c for c in v))
    return (v[0] / n, v[1] / n, v[2] / n) if n > 1e-9 else (1.0, 0.0, 0.0)


def _pitch_deg(eye, look):
    d = (look[0] - eye[0], look[1] - eye[1], look[2] - eye[2])
    horiz = math.hypot(d[0], d[1])
    if horiz < 1e-9:
        return -90.0 if d[2] < 0 else 90.0
    return math.degrees(math.atan2(d[2], horiz))


def _clearance_ok(eye, raycast_fn, min_wall):
    """Fire short rays along +-x, +-y to check the camera is not inside geometry."""
    for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)):
        hit = raycast_fn(eye, d, min_wall)
        if hit is not None and hit < min_wall:
            return False
    return True


def _inside_room(eye, raycast_fn, ceil_probe=6.0):
    """A floor below and a ceiling above means we are inside the shell, not outside it."""
    down = raycast_fn(eye, (0, 0, -1), eye[2] + 1.0)
    up = raycast_fn(eye, (0, 0, 1), ceil_probe)
    return down is not None and up is not None


def frustum_dirs(eye, look, grid=FRUSTUM_GRID, fov_deg=FRUSTUM_FOV_DEG):
    """The grid x grid ray directions over the camera frustum (1000x625 aspect)."""
    f = _norm((look[0] - eye[0], look[1] - eye[1], look[2] - eye[2]))
    up0 = (0.0, 0.0, 1.0)
    r = _norm((f[1] * up0[2] - f[2] * up0[1], f[2] * up0[0] - f[0] * up0[2], f[0] * up0[1] - f[1] * up0[0]))
    u = (r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0])
    half = math.tan(math.radians(fov_deg) / 2.0)
    aspect = 0.625
    out = []
    for iy in range(grid):
        for ix in range(grid):
            sx = (2.0 * (ix + 0.5) / grid - 1.0) * half
            sy = (2.0 * (iy + 0.5) / grid - 1.0) * half * aspect
            out.append(
                _norm((f[0] + sx * r[0] + sy * u[0], f[1] + sx * r[1] + sy * u[1], f[2] + sx * r[2] + sy * u[2]))
            )
    return out


def _room_scaled(room):
    """Scale the look-distance and content thresholds to the room."""
    diag = math.hypot(room["W"], room["D"])
    min_hit = min(MIN_HIT, max(0.9, 0.55 * diag))
    spread = min(MIN_DEPTH_SPREAD, max(0.18, 0.14 * diag))
    rng = min(MIN_DEPTH_RANGE, max(0.45, 0.35 * diag))
    inset = min(INSET, max(0.30, 0.16 * diag))
    min_wall = min(MIN_WALL, max(0.35, 0.16 * diag))
    return min_hit, spread, rng, inset, min_wall


def _view(eye, look, kind, relax, **extra):
    return {
        "pos": [round(c, 4) for c in eye],
        "look_at": [round(c, 4) for c in look],
        "kind": kind,
        "relax": relax,
        **extra,
    }


def _make_look(eye, yaw, pitch_deg, dist=3.0):
    pr = math.radians(pitch_deg)
    return (
        eye[0] + dist * math.cos(pr) * math.cos(yaw),
        eye[1] + dist * math.cos(pr) * math.sin(yaw),
        eye[2] + dist * math.sin(pr),
    )


def content_score(eye, look, raycast_fn, diag):
    """Higher = this view shows more of the room. Returns (score, stats)."""
    hits = [raycast_fn(eye, d, MAX_HIT) for d in frustum_dirs(eye, look)]
    got = [h for h in hits if h is not None]
    frac = len(got) / max(1, len(hits))
    if len(got) < 2:
        return 0.0, dict(hit_frac=round(frac, 3), depth_std=0.0, depth_range=0.0, depth_med=0.0)
    m = sum(got) / len(got)
    std = math.sqrt(sum((h - m) ** 2 for h in got) / len(got))
    rng = max(got) - min(got)
    med = sorted(got)[len(got) // 2]
    st = dict(hit_frac=round(frac, 3), depth_std=round(std, 3), depth_range=round(rng, 3), depth_med=round(med, 3))
    st["reach"] = round(med / diag, 3)
    return ((std + 0.5 * rng) / diag + REACH_W * (med / diag)) * frac, st


def sample_interior(room, raycast_fn, n, seed, max_attempts_per_view=120):
    """Sample `n` interior viewpoints, RANKED by how much of the room they show."""
    rng = random.Random(seed)
    W, D, floor_z = room["W"], room["D"], room.get("floor_z", 0.0)
    diag = math.hypot(W, D)
    base_hit, base_spread, base_range, inset, base_wall = _room_scaled(room)
    x0, x1 = inset, max(inset + 1e-3, W - inset)
    y0, y1 = inset, max(inset + 1e-3, D - inset)

    pool = []
    for _ in range(max_attempts_per_view * n * 3):
        eye = (rng.uniform(x0, x1), rng.uniform(y0, y1), floor_z + CAM_H)
        yaw = rng.uniform(0.0, 2.0 * math.pi)
        pitch = rng.uniform(*PITCH_RANGE)
        look = _make_look(eye, yaw, pitch)
        if not _inside_room(eye, raycast_fn) or not _clearance_ok(eye, raycast_fn, base_wall):
            continue
        hit = raycast_fn(eye, _norm((look[0] - eye[0], look[1] - eye[1], look[2] - eye[2])), MAX_HIT)
        if hit is None or hit < base_hit * 0.5:
            continue
        sc, st = content_score(eye, look, raycast_fn, diag)
        pool.append((sc, eye, look, hit, st))
        if len(pool) >= n * 8:
            break
    pool.sort(key=lambda t: -t[0])

    # best-first with a minimum spacing, then top up from the rest of the pool
    min_sep = max(0.35, 0.10 * diag)
    taken, chosen = set(), []
    for i, cand in enumerate(pool):
        if len(chosen) >= n:
            break
        if all(math.dist(cand[1][:2], c[1][:2]) >= min_sep for c in chosen):
            chosen.append(cand)
            taken.add(i)
    for i, cand in enumerate(pool):
        if len(chosen) >= n:
            break
        if i not in taken:
            chosen.append(cand)

    views, n_strict = [], 0
    for sc, eye, look, hit, st in chosen[:n]:
        strict = (
            st["depth_std"] >= base_spread
            and st["depth_range"] >= base_range
            and st["hit_frac"] >= MIN_FRUSTUM_HIT_FRAC
            and hit >= base_hit
        )
        n_strict += strict
        views.append(
            _view(eye, look, "interior", 0 if strict else 1, content=round(sc, 3), look_dist=round(hit, 3), **st)
        )

    need = n - len(views)  # pool too small: a jittered ring around the room centre
    for i in range(need):
        yaw = 2.0 * math.pi * i / need
        jr = 0.12 * min(W, D) * ((i % 3) + 1) / 3.0
        eye = (
            min(max(W / 2.0 + jr * math.cos(yaw), inset), W - inset),
            min(max(D / 2.0 + jr * math.sin(yaw), inset), D - inset),
            floor_z + CAM_H,
        )
        look = _make_look(eye, yaw, sum(PITCH_RANGE) / 2.0)
        views.append(_view(eye, look, "interior", -1, content=0.0))
    return views[:n], n_strict


def sample_orbit(room, anchor, raycast_fn, n, fallback_positions):
    """`n` viewpoints ringing the anchor look-at, at even azimuth steps."""
    W, D, floor_z = room["W"], room["D"], room.get("floor_z", 0.0)
    z = floor_z + CAM_H
    look, eye0 = tuple(anchor["look"]), tuple(anchor["pos"])
    cx = min(max(look[0], ORBIT_WALL_MARGIN), W - ORBIT_WALL_MARGIN)
    cy = min(max(look[1], ORBIT_WALL_MARGIN), D - ORBIT_WALL_MARGIN)
    centre = (cx, cy, look[2])
    max_r = min(cx, cy, W - cx, D - cy) - ORBIT_WALL_MARGIN
    desired = math.hypot(eye0[0] - centre[0], eye0[1] - centre[1])
    radius = max(ORBIT_ABS_MIN_RADIUS, min(max(desired, ORBIT_MIN_RADIUS), max_r))
    clear = lambda e: _clearance_ok(e, raycast_fn, MIN_WALL * 0.6)
    at = lambda r, i: (
        centre[0] + r * math.cos(2.0 * math.pi * i / n),
        centre[1] + r * math.sin(2.0 * math.pi * i / n),
        z,
    )

    for _ in range(ORBIT_SHRINK_TRIES):
        if all(clear(at(radius, i)) for i in range(n)):
            break
        if radius * ORBIT_SHRINK < ORBIT_ABS_MIN_RADIUS:
            radius = ORBIT_ABS_MIN_RADIUS
            break
        radius *= ORBIT_SHRINK

    views = []
    n_rescued = n_unsafe = n_borrowed = 0
    for i in range(n):
        r_i, eye = radius, at(radius, i)
        ok = clear(eye)
        if not ok:  # per-azimuth radius scan, outside in
            lo_r, hi_r = ORBIT_ABS_MIN_RADIUS, max(max_r, ORBIT_ABS_MIN_RADIUS)
            for s in range(25):
                rr = hi_r - (hi_r - lo_r) * s / 24
                if clear(at(rr, i)):
                    r_i, eye, ok = rr, at(rr, i), True
                    n_rescued += 1
                    break
        if not ok:  # borrow a validated interior position
            eye = tuple(fallback_positions[i % len(fallback_positions)])
            if clear(eye):
                n_borrowed += 1
            else:
                n_unsafe += 1
        p = _pitch_deg(eye, centre)
        lo, hi = PITCH_RANGE
        if p < lo or p > hi or (abs(eye[0] - centre[0]) < 1e-6 and abs(eye[1] - centre[1]) < 1e-6):
            yaw = 2.0 * math.pi * i / n + math.pi  # face outward from the centre
            dist = max(0.8, math.hypot(centre[0] - eye[0], centre[1] - eye[1]))
            tgt = _make_look(eye, yaw, min(hi, max(lo, ORBIT_TILT_DEG)), dist)
        else:
            tgt = centre
        views.append(_view(eye, tgt, "orbit", 0 if r_i == radius else 1))
    if n_rescued or n_unsafe or n_borrowed:
        print(f"[vp] orbit: {n_rescued} rescued, {n_borrowed} borrowed, {n_unsafe} still unsafe", flush=True)
    return views, round(radius, 4)


def sample_views(room, raycast_fn, anchor, n_interior=48, n_orbit=16, seed=0):
    """Build the full views document. Deterministic for a given (room, anchor, seed)."""
    interior, n_strict = sample_interior(room, raycast_fn, n_interior, seed)
    orbit, radius = sample_orbit(room, anchor, raycast_fn, n_orbit, [v["pos"] for v in interior])
    return {
        "room": {"W": room["W"], "D": room["D"], "floor_z": room.get("floor_z", 0.0)},
        "n_interior": n_interior,
        "n_interior_actual": n_strict,
        "n_orbit": n_orbit,
        "orbit_radius": radius,
        "pitch_range": list(PITCH_RANGE),
        "seed": seed,
        "views": interior + orbit,
    }


def blender_raycast(scene, depsgraph, want_object=False):
    """A raycast_fn backed by Blender's scene.ray_cast: hit distance (or hit object name), None on a miss."""
    from mathutils import Vector

    def fn(origin, direction, max_dist):
        hit, loc, _n, _idx, obj, _m = scene.ray_cast(depsgraph, Vector(origin), Vector(direction), distance=max_dist)
        if not hit:
            return None
        if want_object:
            return obj.name if obj is not None else None
        return (Vector(loc) - Vector(origin)).length

    return fn
