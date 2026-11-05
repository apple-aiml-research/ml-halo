#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""MuJoCo forward-kinematics + geometry extraction for the quality checker."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field

import numpy as np


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_ROBOT_ONLY_SCENE = os.path.join(_repo_root(), "..", "assets", "robots", "g1", "mjcf", "scene_g1_29dof.xml")


_SELF_GROUP_KW = {
    "l_arm": ("left_elbow", "left_wrist", "left_rubber_hand"),
    "r_arm": ("right_elbow", "right_wrist", "right_rubber_hand"),
    "l_leg": ("left_hip", "left_knee", "left_ankle"),
    "r_leg": ("right_hip", "right_knee", "right_ankle"),
    "torso": ("pelvis", "waist_", "torso"),
    "head": ("head",),
}
_SELF_CROSSLIMB = frozenset(
    {
        frozenset(p)
        for p in (
            ("l_arm", "r_arm"),
            ("l_arm", "torso"),
            ("r_arm", "torso"),
            ("l_arm", "head"),
            ("r_arm", "head"),
            ("l_arm", "l_leg"),
            ("l_arm", "r_leg"),
            ("r_arm", "l_leg"),
            ("r_arm", "r_leg"),
            ("l_leg", "r_leg"),
        )
    }
)

_FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
_WRIST_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")


def _self_group(name: str) -> str | None:
    for grp, kws in _SELF_GROUP_KW.items():
        if any(k in name for k in kws):
            return grp
    return None


@dataclass
class Tracks:
    """Numpy geometry bundle for one episode (all lengths T unless noted)."""

    fps: float
    T: int
    scene_source: str  # "lab:<id>" | "robot_only"
    body_pos: dict[str, np.ndarray]  # name -> [T, 3] world position
    foot_pos: np.ndarray  # [T, 2, 3] left/right ankle-roll xyz
    foot_sole_z: np.ndarray  # [T, 2] LOWEST sole-corner world z
    floor_z_foot: np.ndarray  # [T, 2] ray-cast floor under each foot
    self_min_dist: np.ndarray  # [T] min signed arm↔leg/torso distance
    self_worst: dict  # {frame, dist, pair:(a,b)}
    foot_sole_z_max: np.ndarray | None = None  # [T, 2] HIGHEST sole-corner z (flatness)
    scene_min_dist: np.ndarray | None = None  # [T] min signed robot↔scene distance
    scene_worst: dict | None = None
    plant_offset: np.ndarray | None = None  # [T] root-z lift to plant lowest foot (diag)
    notes: list[str] = field(default_factory=list)


def _stage_scene(episode, cfg) -> tuple[str, str, list[str]]:
    """Return ``(xml_path, scene_source, notes)`` for the collision scene."""
    import mujoco  # noqa: F401 (ensures mujoco present before we commit)

    notes = [f"no MuJoCo scene for scene_id={episode.scene_id!r}; robot-only floor plane"]
    return _ROBOT_ONLY_SCENE, "robot_only", notes


def _robot_body_ids(model) -> set[int]:
    """Body ids in the robot subtree (walk parents up to 'pelvis')."""
    import mujoco

    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    ids: set[int] = set()
    for b in range(model.nbody):
        p = b
        while p > 0:
            if p == pelvis:
                ids.add(b)
                break
            p = int(model.body_parentid[p])
    return ids


def _geom_body_name(model, g: int) -> str:
    import mujoco

    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""


def _collision_geoms(model) -> list[int]:
    return [g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]]


def _sole_geoms(model, foot_bid: int) -> list[int]:
    """Small low geoms on a foot body (the toe/heel corner spheres)."""
    out = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == foot_bid and model.geom_aabb[g][5] < 0.03:
            out.append(g)
    return out


def _floor_sampler(model):
    """Return ``f(xy)->z`` for the visible floor, with robot + any freejoint"""
    import mujoco

    fd = mujoco.MjData(model)
    mujoco.mj_resetData(model, fd)
    fd.qpos[:3] = [50.0, 50.0, 0.0]
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            adr = int(model.jnt_qposadr[j])
            fd.qpos[adr : adr + 3] = [60.0 + j, 60.0, -5.0]
    mujoco.mj_forward(model, fd)

    plane_z = None
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0 and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            z = float(fd.geom_xpos[g, 2])
            plane_z = z if plane_z is None else max(plane_z, z)

    def f(xy) -> float:
        gid = np.zeros(1, np.int32)
        dd = mujoco.mj_ray(model, fd, np.array([xy[0], xy[1], 3.0]), np.array([0.0, 0.0, -1.0]), None, 1, -1, gid)
        raw = float(3.0 - dd) if dd >= 0 else None
        if plane_z is not None:
            if raw is None or raw > plane_z + 0.06:
                return plane_z
            return raw
        return raw if raw is not None else 0.0

    return f


def build_tracks(episode, cfg) -> Tracks:
    """Run FK over ``episode.qpos`` and return the ``Tracks`` bundle."""
    import mujoco

    xml_path, scene_source, notes = _stage_scene(episode, cfg)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    if scene_source.startswith("lab:") and xml_path.startswith(tempfile.gettempdir()):
        try:
            os.remove(xml_path)  # model is loaded; temp XML no longer needed
        except OSError:
            pass

    qpos = episode.qpos
    T = episode.T
    nq_robot = min(36, model.nq)

    def bid(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    robot_bodies = _robot_body_ids(model)
    col = _collision_geoms(model)
    robot_col = [g for g in col if model.geom_bodyid[g] in robot_bodies]
    grp = {g: _self_group(_geom_body_name(model, g)) for g in robot_col}
    self_pairs = [
        (a, b)
        for i, a in enumerate(robot_col)
        for b in robot_col[i + 1 :]
        if grp[a] and grp[b] and grp[a] != grp[b] and frozenset((grp[a], grp[b])) in _SELF_CROSSLIMB
    ]

    scene_geoms = [
        g
        for g in col
        if model.geom_bodyid[g] not in robot_bodies
        and model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE
    ]

    foot_bids = [bid(n) for n in _FOOT_BODIES]
    sole_geoms = [_sole_geoms(model, fb) if fb >= 0 else [] for fb in foot_bids]
    track_bodies = {n: bid(n) for n in ("pelvis", "head_link", "torso_link", *_FOOT_BODIES, *_WRIST_BODIES)}
    track_bodies = {n: i for n, i in track_bodies.items() if i >= 0}

    body_pos = {n: np.zeros((T, 3)) for n in track_bodies}
    foot_pos = np.zeros((T, 2, 3))
    foot_sole_z = np.full((T, 2), np.nan)
    foot_sole_z_max = np.full((T, 2), np.nan)
    self_min = np.full(T, 9.0)
    self_worst = {"frame": -1, "dist": 9.0, "pair": None}
    do_scene = len(scene_geoms) > 0
    scene_min = np.full(T, 9.0) if do_scene else None
    scene_worst = {"frame": -1, "dist": 9.0, "pair": None} if do_scene else None

    self_cut = float(cfg.coll_self_cutoff_m)
    scene_cut = 0.10

    for t in range(T):
        data.qpos[:nq_robot] = qpos[t, :nq_robot]
        mujoco.mj_forward(model, data)

        for n, i in track_bodies.items():
            body_pos[n][t] = data.xpos[i]
        for k, fb in enumerate(foot_bids):
            if fb >= 0:
                foot_pos[t, k] = data.xpos[fb]
            lo = hi = np.nan
            for g in sole_geoms[k]:
                bot = data.geom_xpos[g, 2] + model.geom_aabb[g][2] - model.geom_aabb[g][5]
                lo = bot if (np.isnan(lo) or bot < lo) else lo
                hi = bot if (np.isnan(hi) or bot > hi) else hi
            foot_sole_z[t, k] = lo
            foot_sole_z_max[t, k] = hi


        wd = 9.0
        wpair = None
        for a, b in self_pairs:
            di = mujoco.mj_geomDistance(model, data, a, b, self_cut, None)
            if di < wd:
                wd, wpair = di, (a, b)
        self_min[t] = wd
        if wd < self_worst["dist"]:
            self_worst.update(
                frame=t,
                dist=float(wd),
                pair=(_geom_body_name(model, wpair[0]), _geom_body_name(model, wpair[1])) if wpair else None,
            )

        if do_scene:
            wd = 9.0
            wpair = None
            for a in robot_col:
                for o in scene_geoms:
                    di = mujoco.mj_geomDistance(model, data, a, o, scene_cut, None)
                    if di < wd:
                        wd, wpair = di, (a, o)
            scene_min[t] = wd
            if wd < scene_worst["dist"]:
                scene_worst.update(
                    frame=t,
                    dist=float(wd),
                    pair=(
                        _geom_body_name(model, wpair[0]),
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, wpair[1]) or f"g{wpair[1]}",
                    )
                    if wpair
                    else None,
                )

    floor_at = _floor_sampler(model)
    floor_z_foot = np.zeros((T, 2))
    for t in range(T):
        for k in range(2):
            floor_z_foot[t, k] = floor_at(foot_pos[t, k, :2])

    gap = foot_sole_z - floor_z_foot  # [T,2]; <0 = sunk
    with np.errstate(invalid="ignore"):
        min_gap = np.nanmin(gap, axis=1)
    plant_offset = np.where(min_gap < 0, -min_gap, 0.0)

    if not do_scene:
        notes.append(
            "scene_collision unavailable (MuJoCo scene has no furniture "
            "geometry — floor only); foot_sink covers the floor"
        )

    return Tracks(
        fps=episode.fps,
        T=T,
        scene_source=scene_source,
        body_pos=body_pos,
        foot_pos=foot_pos,
        foot_sole_z=foot_sole_z,
        foot_sole_z_max=foot_sole_z_max,
        floor_z_foot=floor_z_foot,
        self_min_dist=self_min,
        self_worst=self_worst,
        scene_min_dist=scene_min,
        scene_worst=scene_worst,
        plant_offset=plant_offset,
        notes=notes,
    )
