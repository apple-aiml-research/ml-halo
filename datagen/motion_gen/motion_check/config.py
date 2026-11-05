#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Thresholds + weights for the motion quality checker."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any

_CONFIG_DIR = os.environ.get(
    "MOTION_CHECK_CONFIG_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", "motion_check"),
)

@dataclass
class CheckConfig:
    """All checker thresholds (SI units: metres, seconds, radians)."""

    fps: float = 30.0

    quality_tier: str = ""

    foot_sink_tol_m: float = 0.03
    foot_sink_fail_frac: float = 0.02  # fail if >2% of frames sink
    foot_float_tol_m: float = 0.06
    foot_float_min_run: int = 6  # sustained frames to flag a float
    foot_contact_h_m: float = 0.03
    foot_skate_vert_vel_mps: float = 0.05
    foot_skate_min_contact_run: int = 4
    foot_skate_speed_mps: float = 0.15
    foot_skate_fail_frac: float = 0.03  # fail if >3% of contact-frames skate
    foot_skate_severe_mps: float = 0.35
    stance_width_max_m: float = 0.35
    stance_width_min_m: float = 0.05  # crossed / near-zero (feet on top of each other)
    stance_width_fail_frac: float = 0.10  # flag if >10% of grounded frames out of band
    feet_close_min_m: float = 0.11
    step_clearance_min_m: float = 0.065
    step_clearance_walk_z: float = 0.60  # root-above-floor gate for "walking" (vs sitting)
    leg_swing_max_m: float = 0.15
    walk_pause_min_s: float = 1.0
    walk_pause_step_h_m: float = 0.035
    walk_pause_speed_mps: float = 0.10
    waist_bend_max_deg: float = 18.0
    flight_min_m: float = 0.02
    flight_min_run: int = 2

    dyn_jump_prominence_m: float = 0.10
    dyn_speed_root_mps: float = 2.5  # excessive root horizontal speed
    dyn_speed_joint_radps: float = 12.0  # excessive joint angular speed
    dyn_teleport_step_m: float = 0.35  # per-frame root jump = teleport
    dyn_backforth_reversals: int = 3  # heading reversals to flag
    dyn_jitter_jerk_mps3: float = 40.0  # root jerk RMS for "jitter"
    dyn_jitter_fail: float = 80.0  # hard-ish jitter level
    dyn_hop_prominence_m: float = 0.07
    dyn_hop_repeat_prom_m: float = 0.045
    dyn_hop_repeat_n: int = 2
    dyn_turn_rate_dps: float = 250.0
    arm_speed_p95_radps: float = 1.1
    crab_lateral_frac: float = 0.32
    crab_min_speed_mps: float = 0.15  # only count meaningfully-moving frames
    dyn_drift_up_m: float = 0.04
    dyn_drift_win_s: float = 0.7
    dyn_drift_horiz_max_m: float = 0.12
    dyn_drift_max_m: float = 0.18  # ignore full stand-ups (larger rises)

    joint_over_limit_tol_deg: float = 3.0
    joint_flip_rad: float = 1.0
    root_curv_g_max: float = 15.0
    dead_travel_m: float = 0.10
    float_gate_contact_m: float = 0.02

    coll_self_pen_tol_m: float = 0.015
    coll_self_cutoff_m: float = 0.12  # mj_geomDistance search cutoff
    coll_scene_pen_tol_m: float = 0.01  # robot↔scene penetration to flag

    sit_root_z_lo: float = 0.35
    sit_root_z_hi: float = 0.62
    sit_settle_frames: int = 15  # last N frames must be a stable sit
    sit_root_z_var_tol: float = 0.03  # max root-z std over the settle window
    sit_seat_contact_tol_m: float = 0.08

    weights: dict = field(default_factory=dict)
    accept_min_score: float = 0.0
    severity_overrides: dict = field(default_factory=dict)
    hints: dict = field(default_factory=dict)

    def update(self, d: dict[str, Any]) -> "CheckConfig":
        """In-place overlay of a flat dict of field → value. Unknown keys are"""
        known = {f.name for f in fields(self)}
        for k, v in (d or {}).items():
            if k not in known:
                print(f"[motion_gen.motion_check.config] ignoring unknown key {k!r}", flush=True)
                continue
            cur = getattr(self, k)
            if isinstance(cur, dict) and isinstance(v, dict):
                cur.update(v)
            else:
                setattr(self, k, v)
        return self

    @classmethod
    def for_task(cls, task_type: str | None, config_dir: str | None = None, tier: str | None = None) -> "CheckConfig":
        """Build defaults, overlay ``default.yaml`` then the sit base (for sit"""
        cfg = cls()
        cdir = config_dir or _CONFIG_DIR
        task = str(task_type or "")
        names = ["default"]
        if task in _SIT_TASKS:
            names.append("sit")  # shared sit base (band + foot relax)
        if task and task not in names:
            names.append(task)  # task-specific overrides (may layer on sit)
        if tier:
            names.append(f"tier_{tier}")
            cfg.quality_tier = tier
        for name in names:
            if not name:
                continue
            p = os.path.join(cdir, f"{name}.yaml")
            if os.path.isfile(p):
                try:
                    import yaml

                    with open(p) as f:
                        cfg.update(yaml.safe_load(f) or {})
                except ImportError:
                    print("[motion_gen.motion_check.config] pyyaml unavailable; using defaults", flush=True)
                    break
        return cfg

_SIT_TASKS = ("sit", "sit_on_chair", "sit_on_furniture")
