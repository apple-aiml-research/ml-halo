#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Sit-on-chair episode specs, generated motion-first: chair at the origin of an empty canonical scene."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

from common import placement as _placement
from motion_gen import unicycle as _uni
from motion_gen.episode_spec import AssetPlacement, EpisodeSpec, RenderVariation, RobotSpawn

ARM_CHAIR_OPEN_YAW0 = -math.pi / 2.0  # mesh yaw at which the arm chair's opening faces +x
_CHAIR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "objects", "chairs"))
ARM_CHAIR_USDZ = os.path.join(_CHAIR_DIR, "arm_chair.usdz")

ROBOT_RADIUS = 0.30

_DR_SALT = 0x5A17D0
_DR_PROMPT = "sit down on armchair"
_DR_CAM_ARTIFACT_RANGES = {
    "vignette": (0.05, 0.18),
    "chroma": (0.00, 0.08),
    "noise": (0.000, 0.003),
    "salt_pepper": (0.000, 0.001),
    "defocus": (0.00, 0.06),
    "posterize": (0.00, 0.08),
    "scanline": (0.00, 0.06),
    "brightness": (0.65, 1.50),
    "contrast": (0.95, 1.06),
    "saturation": (0.92, 1.08),
    "gamma": (0.95, 1.12),
    "color_tone": (-1.0, 1.0),
}


@dataclass
class SceneOcc:
    """Occupancy grid + floor height of the scene a spec is sampled in."""

    occ: _placement.OccGrid
    floor_z: float
    scene_id: str
    scene_dir: str


def canonical_scene(span: float = 12.0, res: float = 0.10) -> SceneOcc:
    """The scene-free motion-first frame: an all-free ``span`` x ``span`` grid centred on the chair, floor at z=0."""
    n = int(span / res)
    occ = _placement.OccGrid(grid=np.zeros((n, n), dtype=bool), x_min=-span / 2, y_min=-span / 2, resolution=res)
    return SceneOcc(occ=occ, floor_z=0.0, scene_id="canonical", scene_dir="")


@dataclass
class SitBatchConfig:
    """Knobs for sampling a sit episode."""

    chair_half_xy: tuple[float, float] = (0.35, 0.35)
    chair_asset_path: str = ARM_CHAIR_USDZ
    chair_name: str = "arm_chair"
    seat_z: float = 0.45
    back_offset: float = 0.18  # seated pelvis sits this far in front of the chair centre
    approach_range: tuple[float, float] = (1.5, 3.0)
    walk_speed: float = 0.55
    spawn_yaw_perturb_deg: float = 30.0
    stand_gap: float = 0.55
    max_walk_end_gap: float = 0.40
    n_spawn_candidates: int = 400
    n_waypoints: int = 20
    vdr_level: str = "light"
    asset_color: str = "default"
    chair_asset_pool: tuple[str, ...] = ()


def _sample_dr(seed: int, cfg: SitBatchConfig) -> dict:
    """Per-episode chair variant + light camera-artifact DR from a seeded stream."""
    if (cfg.vdr_level, cfg.asset_color) != ("light", "default"):
        raise ValueError(
            f"only vdr_level=light, asset_color=default are supported, got {cfg.vdr_level, cfg.asset_color}"
        )
    rng = np.random.default_rng((int(seed) & 0xFFFFFFFF) ^ _DR_SALT)
    variant = str(rng.choice(list(cfg.chair_asset_pool))) if cfg.chair_asset_pool else None
    cam = {k: round(float(rng.uniform(*lohi)), 4) for k, lohi in _DR_CAM_ARTIFACT_RANGES.items()}
    cam["motion_blur_gain"] = 0.3
    cam["motion_blur_rand_dir_prob"] = 0.15
    return {"chair_variant": variant, "cam_artifacts": cam}


def _yaw_quat_wxyz(yaw: float):
    h = yaw / 2.0
    return (math.cos(h), 0.0, 0.0, math.sin(h))


def make_sit_episode_spec(
    seed: int,
    scene: SceneOcc,
    cfg: SitBatchConfig | None = None,
    *,
    cam_kind: str = "fisheye",
    robot_variant: str = "g1_29dof_thor",
) -> EpisodeSpec | None:
    """Sample a reproducible sit ``EpisodeSpec`` (chair at the origin, robot walks up, turns, sits)."""
    cfg = cfg or SitBatchConfig()
    if scene.scene_id != "canonical":
        raise ValueError("sit specs are generated in the canonical scene only (placed into rooms at render time)")
    rng = np.random.default_rng(seed)
    occ = scene.occ
    dr = _sample_dr(seed, cfg)
    half = cfg.chair_half_xy
    chair_xy, seated_yaw = (0.0, 0.0), 0.0
    chair_yaw = float(seated_yaw - ARM_CHAIR_OPEN_YAW0)

    occ_work = _placement.OccGrid(grid=occ.grid.copy(), x_min=occ.x_min, y_min=occ.y_min, resolution=occ.resolution)
    _placement.add_box(occ_work, chair_xy, half, chair_yaw)

    seat_xy = (
        chair_xy[0] + cfg.back_offset * math.cos(seated_yaw),
        chair_xy[1] + cfg.back_offset * math.sin(seated_yaw),
    )
    seat_target_xyz = [seat_xy[0], seat_xy[1], scene.floor_z + cfg.seat_z]
    stop_xy = (seat_xy[0] + cfg.stand_gap * math.cos(seated_yaw), seat_xy[1] + cfg.stand_gap * math.sin(seated_yaw))
    if not _placement.is_circle_free(occ_work, stop_xy, ROBOT_RADIUS):
        return None

    spawn_xy = spawn_yaw = corridor = None
    dil = _placement.dilate(occ_work, ROBOT_RADIUS)
    for _ in range(cfg.n_spawn_candidates):
        d = float(rng.uniform(*cfg.approach_range))
        ang = seated_yaw + float(rng.uniform(-math.radians(70), math.radians(70)))
        cand = (seat_xy[0] + d * math.cos(ang), seat_xy[1] + d * math.sin(ang))
        if not _placement.is_circle_free(occ_work, cand, ROBOT_RADIUS):
            continue
        path = _placement.plan_path_astar(dil, cand, stop_xy)
        if path is None or len(path) < 2:
            continue
        corridor = _placement.simplify_path(path, n_waypoints=cfg.n_waypoints)
        spawn_xy = (float(cand[0]), float(cand[1]))
        bearing_to_seat = math.atan2(seat_xy[1] - cand[1], seat_xy[0] - cand[0])
        perturb = math.radians(cfg.spawn_yaw_perturb_deg)
        spawn_yaw = float(_uni.wrap(bearing_to_seat + rng.uniform(-perturb, perturb)))
        break
    if spawn_xy is None:
        return None

    plan = _uni.simulate_corridor(
        spawn_xy, spawn_yaw, corridor, occ=occ, robot_radius=ROBOT_RADIUS, v_max=cfg.walk_speed
    )
    if len(plan) < 2 or float(np.linalg.norm(plan[-1, :2] - np.asarray(stop_xy))) > cfg.max_walk_end_gap:
        return None

    chair = AssetPlacement(
        name=cfg.chair_name,
        asset_path=dr["chair_variant"] or cfg.chair_asset_path,
        role="target",
        pos_xyz=[chair_xy[0], chair_xy[1], scene.floor_z],
        quat_wxyz=list(_yaw_quat_wxyz(chair_yaw)),
        bbox_min=[-half[0], -half[1], 0.0],
        bbox_max=[half[0], half[1], 2 * cfg.seat_z],
        scale=1.0,
    )
    return EpisodeSpec(
        seed=seed,
        scene_id="infinigen",
        task_type="sit",
        target_object_id=cfg.chair_name,
        target_xyz=[chair_xy[0], chair_xy[1], scene.floor_z + cfg.seat_z],
        robot=RobotSpawn(xy=spawn_xy, yaw_rad=spawn_yaw, cam_kind=cam_kind, variant=robot_variant),
        placements=[chair],
        path_xy=plan[:, :2].tolist(),
        path_yaw=plan[:, 2].tolist(),
        scene_source="infinigen",
        scene_dir=scene.scene_dir,
        floor_z=scene.floor_z,
        seat_target_xyz=seat_target_xyz,
        seat_yaw_rad=float(seated_yaw),
        render=RenderVariation(prompt=_DR_PROMPT, cam_artifacts=dr["cam_artifacts"]),
    )


class SitTask:
    """Sit-on-chair task: ``build_scenario(scene, seed, cfg)`` → ``EpisodeSpec`` or ``None``."""

    name = "sit"

    def __init__(self, *, cam_kind: str = "fisheye", robot_variant: str = "g1_29dof_thor"):
        self.cam_kind = cam_kind
        self.robot_variant = robot_variant

    def build_scenario(self, scene, seed: int, cfg: SitBatchConfig | None = None) -> EpisodeSpec | None:
        return make_sit_episode_spec(seed, scene, cfg, cam_kind=self.cam_kind, robot_variant=self.robot_variant)
