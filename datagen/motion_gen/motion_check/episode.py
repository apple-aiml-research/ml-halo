#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Load a generated motion episode into a uniform ``MotionEpisode``."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _first_existing(*paths: str) -> str | None:
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


@dataclass
class MotionEpisode:
    """One generated sequence: robot qpos and metadata."""

    episode_id: str
    qpos: np.ndarray  # [T, 36] pos(3)+quat_wxyz(4)+29 joints
    task_type: str = "unknown"
    meta: dict = field(default_factory=dict)
    fps: float = 30.0
    _spec: Any = field(default=None, repr=False)
    _spec_loaded: bool = field(default=False, repr=False)

    @property
    def T(self) -> int:
        return int(self.qpos.shape[0])

    @property
    def root_xyz(self) -> np.ndarray:
        return self.qpos[:, 0:3]

    @property
    def root_quat_wxyz(self) -> np.ndarray:
        return self.qpos[:, 3:7]

    @property
    def joints(self) -> np.ndarray:
        return self.qpos[:, 7:36]

    @property
    def spec(self):
        """Deserialized ``EpisodeSpec`` (lazy, best-effort). ``None`` if the meta"""
        if not self._spec_loaded:
            self._spec_loaded = True
            es = self.meta.get("episode_spec")
            if isinstance(es, dict):
                try:
                    from motion_gen.episode_spec import EpisodeSpec

                    self._spec = EpisodeSpec.from_dict(es)
                except Exception as e:  # noqa: BLE001 — never fail a check on this
                    print(f"[motion_gen.motion_check.episode] EpisodeSpec parse skipped: {e}", flush=True)
                    self._spec = None
        return self._spec

    def _es(self, key: str, default: Any = None) -> Any:
        es = self.meta.get("episode_spec")
        if isinstance(es, dict) and key in es:
            return es[key]
        return default

    @property
    def phases(self) -> dict[str, list[int]]:
        """``{phase_name: [start, end]}`` (e.g. pick/carry/place). Empty if none."""
        return dict(self.meta.get("phases", {}) or {})

    @property
    def seams(self) -> list[int]:
        """Segment-boundary frame indices where Kimodo segments were stitched."""
        return list(self.meta.get("seams", []) or [])

    @property
    def goal_xy(self) -> list[float] | None:
        g = self.meta.get("goal")
        return [float(g[0]), float(g[1])] if g is not None else None

    @property
    def target_xyz(self) -> list[float] | None:
        v = self._es("target_xyz")
        return [float(x) for x in v] if v is not None else None

    @property
    def scene_id(self) -> str | None:
        return self._es("scene_id") or self.meta.get("scene")

    @classmethod
    def load(cls, path: str, tag: str | None = None, fps: float = 30.0) -> "MotionEpisode":
        """Load from a directory (+ optional ``tag`` like ``"ep0"``) or a qpos"""
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            d = path
            if tag:
                qpos_p = _first_existing(os.path.join(d, f"{tag}_qpos.npy"), os.path.join(d, f"{tag}.npy"))
                meta_p = _first_existing(os.path.join(d, f"{tag}_meta.json"), os.path.join(d, "meta.json"))
                eid = f"{os.path.basename(d.rstrip('/'))}/{tag}"
            else:
                qpos_p = _first_existing(
                    os.path.join(d, "g1_qpos.npy"), os.path.join(d, "qpos.npy"), os.path.join(d, "traj.npy")
                )
                meta_p = _first_existing(os.path.join(d, "meta.json"))
                eid = os.path.basename(d.rstrip("/"))
        else:  # a direct qpos file
            d = os.path.dirname(path)
            qpos_p = path
            stem = os.path.basename(path)
            pre = stem[:-9] if stem.endswith("_qpos.npy") else os.path.splitext(stem)[0]
            meta_p = _first_existing(os.path.join(d, f"{pre}_meta.json"), os.path.join(d, "meta.json"))
            eid = pre or stem

        if qpos_p is None:
            raise FileNotFoundError(f"no qpos .npy found under {path!r} (tag={tag!r})")

        qpos = np.load(qpos_p).astype(np.float64)
        if qpos.ndim != 2 or qpos.shape[1] < 7:
            raise ValueError(f"expected [T, >=7] qpos, got {qpos.shape} from {qpos_p}")
        meta = json.load(open(meta_p)) if meta_p else {}
        task_type = (meta.get("episode_spec") or {}).get("task_type") or meta.get("task_type") or "unknown"
        return cls(episode_id=eid, qpos=qpos, task_type=task_type, meta=meta, fps=fps)
