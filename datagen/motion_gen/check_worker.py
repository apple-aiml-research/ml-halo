#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Lightweight calibrated-checker worker for parallel best-of-N filtering."""

import numpy as np
from motion_gen.motion_check.episode import MotionEpisode
from motion_gen.motion_check.config import CheckConfig
from motion_gen.motion_check.run import run_episode

_CFG = {}


def check_one(args):
    """args = (idx, qpos[T,36] float32, phases dict, tier). Returns a compact dict"""
    idx, qpos, phases, tier = args
    cfg = _CFG.get(tier)
    if cfg is None:
        cfg = _CFG[tier] = CheckConfig.for_task("sit_on_chair", tier=tier)
    ep = MotionEpisode(
        episode_id=f"c{idx}",
        qpos=np.asarray(qpos, dtype=np.float64),
        task_type="sit_on_chair",
        meta={"task_type": "sit_on_chair", "phases": phases},
    )
    rep = run_episode(ep, cfg, use_mujoco=True)
    d = rep.to_dict()
    return {
        "idx": idx,
        "passed": bool(rep.verdict.accept),
        "blocking": list(rep.verdict.blocking),
        "score": float(rep.verdict.score),
        "checker_id": d.get("checker_id"),
        "checks": {r.name: r.to_dict() for r in rep.results},
    }
