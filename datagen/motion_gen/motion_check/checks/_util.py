#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Small shared helpers for building CheckResults (numpy-only)."""

from __future__ import annotations

import numpy as np

from motion_gen.motion_check.schema import SOFT, CheckResult, Status, mask_to_intervals


def graded_score(value: float, good: float, bad: float) -> float:
    """1.0 when ``value <= good``, 0.0 when ``value >= bad``, linear between."""
    if good == bad:
        return 1.0 if value <= good else 0.0
    frac = (value - good) / (bad - good)
    return float(np.clip(1.0 - frac, 0.0, 1.0))


_G = 9.81


def root_curvature_g(root_xyz: np.ndarray, fps: float) -> np.ndarray:
    """Per-interior-frame path curvature as an implied acceleration in g."""
    p = np.asarray(root_xyz, dtype=float)
    if p.shape[0] < 3:
        return np.zeros(0)
    dev = np.linalg.norm(p[1:-1] - 0.5 * (p[:-2] + p[2:]), axis=1)  # [T-2] m
    dt = 1.0 / float(fps)
    return (2.0 * dev / (dt * dt)) / _G


def make_result(
    name: str,
    *,
    value,
    passed: bool,
    score: float,
    severity: str = SOFT,
    time: list[list[int]] | None = None,
    comment: str = "",
    status: Status | None = None,
    trace: list[str] | None = None,
    meta: dict | None = None,
) -> CheckResult:
    if status is None:
        status = Status.PASS if passed else Status.FAIL
    return CheckResult(
        name=name,
        status=status,
        value=value,
        passed=passed,
        score=score,
        time=time or None,
        comment=comment,
        severity=severity,
        trace=trace or [],
        meta=meta or {},
    )


def skip(name: str, comment: str, severity: str = SOFT) -> CheckResult:
    return CheckResult(
        name=name, status=Status.SKIP, value=None, passed=True, score=1.0, comment=comment, severity=severity
    )


def step_mask_to_intervals(mask, T: int, offset: int = 1) -> list[list[int]] | None:
    """A boolean mask over a differenced series (length ``T-offset``) → frame"""
    mask = np.asarray(mask, dtype=bool)
    full = np.zeros(T, dtype=bool)
    n = min(len(mask), T - offset)
    if n > 0:
        full[offset : offset + n] = mask[:n]
    iv = mask_to_intervals(full.tolist())
    return iv or None


def locomotion_mask(episode, tracks) -> np.ndarray:
    """Per-frame boolean: frames in a LOCOMOTION phase (walking/turning) where"""
    import numpy as np

    T = episode.T
    ph = episode.phases
    if ph:
        m = np.zeros(T, dtype=bool)
        for name, span in ph.items():
            if str(name).upper() in ("SIT", "STAND", "SETTLE"):
                continue
            try:
                a, b = int(span[0]), int(span[1])
            except (TypeError, IndexError, ValueError):
                continue
            m[max(0, a) : min(T, b)] = True
        return m
    if tracks is not None and getattr(tracks, "floor_z_foot", None) is not None:
        floor = np.nanmedian(tracks.floor_z_foot, axis=1)
    else:
        floor = 0.0
    return (episode.root_xyz[:, 2] - floor) > 0.70
