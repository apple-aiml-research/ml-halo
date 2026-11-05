#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Per-episode ``EpisodeSpec``: the scene + task description shared by motion gen, render and the gate."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Literal

import numpy as np


CamKind = Literal["pinhole", "fisheye"]
RobotVariant = Literal["g1_29dof", "g1_29dof_thor"]


@dataclass
class AssetPlacement:
    name: str
    asset_path: str  # absolute USDZ path
    role: str  # "target" | "clutter" | "support" | "obstacle"
    pos_xyz: list[float]  # [x, y, z] in scene frame, Z-up
    quat_wxyz: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    bbox_min: list[float] = field(default_factory=lambda: [-0.1, -0.1, 0.0])
    bbox_max: list[float] = field(default_factory=lambda: [0.1, 0.1, 0.2])
    scale: float = 1.0
    material_rgba: list[float] | None = None
    material_roughness: float | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class RobotSpawn:
    xy: tuple[float, float]
    yaw_rad: float
    z: float = 0.0  # spawn height; 0 = derive from scene floor_z at gen time
    init_pose: list[float] | None = None  # optional initial joint/qpos pose; None = generator default
    cam_kind: CamKind = "fisheye"  # default ego camera model (OAK fisheye)
    variant: RobotVariant = "g1_29dof"


@dataclass
class RenderVariation:
    """Per-episode render / visual domain-randomization config."""

    prompt: str = ""  # grounded kimodo text prompt for the task segment
    duration_s: float = 0.0  # task-segment duration (0 = generator default)
    cam_artifacts: dict = field(default_factory=dict)  # FPV+OAK sensor-FX knobs
    lighting_variation: str = "default"  # render lighting preset


@dataclass
class EpisodeSpec:
    seed: int
    scene_id: str
    task_type: str
    target_object_id: str  # which placement is the target
    target_xyz: list[float]
    robot: RobotSpawn
    placements: list[AssetPlacement] = field(default_factory=list)
    dome_intensity: float = 800.0
    overhead_intensity: float = 15000.0
    dome_color: tuple[float, float, float] = (0.95, 0.95, 0.97)
    path_xy: list[list[float]] = field(default_factory=list)
    interaction_kind: str = "none"
    grasp_frame: int = -1
    release_frame: int = -1

    scene_source: str = "infinigen"
    scene_dir: str = ""  # local scene dir (…/scenes/<scene_id>); rendered: staging dir
    scene_s3_path: str = ""  # storage key under HALO_S3_ROOT (rendered specs)
    floor_z: float = 0.0

    path_yaw: list[float] = field(default_factory=list)
    seat_target_xyz: list[float] = field(default_factory=list)
    seat_yaw_rad: float = 0.0
    sit_start_frame: int = -1
    sit_end_frame: int = -1

    render: "RenderVariation" = field(default_factory=RenderVariation)

    @property
    def prompt(self) -> str:
        return self.render.prompt

    @property
    def duration_s(self) -> float:
        return self.render.duration_s

    @property
    def cam_artifacts(self) -> dict:
        return self.render.cam_artifacts

    @property
    def lighting_variation(self) -> str:
        return self.render.lighting_variation

    def to_dict(self) -> dict:
        return _to_jsonable(asdict(self))

    def dump(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=False)

    @classmethod
    def load(cls, path: str) -> "EpisodeSpec":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodeSpec":
        d = dict(d)
        d["robot"] = RobotSpawn(**_only_fields(RobotSpawn, {**d["robot"], "xy": tuple(d["robot"]["xy"])}))
        d["placements"] = [AssetPlacement(**_only_fields(AssetPlacement, p)) for p in d.get("placements", [])]
        d["dome_color"] = tuple(d.get("dome_color", (0.95, 0.95, 0.97)))
        d["render"] = RenderVariation(**_only_fields(RenderVariation, d.get("render") or {}))
        return cls(**_only_fields(cls, d))


def _only_fields(cls, d: dict) -> dict:
    """``d`` restricted to ``cls``'s dataclass fields (older specs carry extra keys)."""
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in valid}


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj
