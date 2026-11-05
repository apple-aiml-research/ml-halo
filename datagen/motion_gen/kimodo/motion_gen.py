#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Constants and pure-numpy helpers shared by the sit segment chain."""

from __future__ import annotations

import os

import numpy as np


PROMPTS = {
    "WALK": "A person walks",
    "TURN": os.environ.get("TURN_PROMPT", "A person turns around slowly"),
    "SIT": "A person sits on a chair",
}

PLAN_3 = [("WALK", 0, 1), ("TURN", 1, 2), ("SIT", 2, 4)]


def seam_blend(qpos: np.ndarray, seams, k: int = 7) -> np.ndarray:
    """Linearly cross-fade root pos/quat + joints across ±k frames of each seam"""
    for s in seams:
        lo, hi = s - k, s + k
        if lo - 1 < 0 or hi >= len(qpos):
            continue
        n = hi - lo + 1
        p0, p1 = qpos[lo - 1, :3].copy(), qpos[hi, :3].copy()
        q0, q1 = qpos[lo - 1, 3:7].copy(), qpos[hi, 3:7].copy()
        if float(np.dot(q0, q1)) < 0:
            q1 = -q1
        a, bvec = qpos[lo - 1, 7:36].copy(), qpos[hi, 7:36].copy()
        for j, idx in enumerate(range(lo, hi + 1)):
            w = (j + 1) / (n + 1)
            qpos[idx, :3] = (1 - w) * p0 + w * p1
            ql = (1 - w) * q0 + w * q1
            nrm = np.linalg.norm(ql)
            qpos[idx, 3:7] = ql / nrm if nrm > 1e-8 else q0
            qpos[idx, 7:36] = (1 - w) * a + w * bvec
    return qpos
