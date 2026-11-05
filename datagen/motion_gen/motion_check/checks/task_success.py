#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Task-success checkers — did the robot accomplish the task? Keyed by"""

from __future__ import annotations

import numpy as np

from motion_gen.motion_check.checks._util import graded_score, make_result, skip
from motion_gen.motion_check.checks.base import CheckContext, MotionQualityChecker, register
from motion_gen.motion_check.config import _SIT_TASKS
from motion_gen.motion_check.schema import HARD, SOFT


def check_sits_at_end(ctx: CheckContext):
    """The robot ends seated and settled: the final root z drops into the seated"""
    ep, cfg = ctx.episode, ctx.cfg
    root_z = float(ep.qpos[-1, 2])
    n = min(cfg.sit_settle_frames, ep.T)
    window = ep.qpos[-n:, 2]
    var = float(np.std(window))
    in_band = cfg.sit_root_z_lo <= root_z <= cfg.sit_root_z_hi
    settled = var <= cfg.sit_root_z_var_tol
    passed = bool(in_band and settled)
    parts = []
    if not in_band:
        parts.append(f"final root z {root_z:.2f} m outside seated band [{cfg.sit_root_z_lo}, {cfg.sit_root_z_hi}]")
    if not settled:
        parts.append(
            f"root z still moving over last {n} frames (std {var * 100:.1f} cm > {cfg.sit_root_z_var_tol * 100:.0f} cm)"
        )
    trace = [
        f"final root z {root_z:.2f} m in seated band "
        f"[{cfg.sit_root_z_lo}, {cfg.sit_root_z_hi}] → {'ok' if in_band else 'NO'}",
        f"settle std over last {n} frames {var * 100:.1f} cm "
        f"(tol {cfg.sit_root_z_var_tol * 100:.0f} cm) → {'ok' if settled else 'NO'}",
        f"{'PASS' if passed else 'FAIL'}",
    ]
    return [
        make_result(
            "sits_at_end",
            value={"root_z_end": round(root_z, 3), "settle_std": round(var, 4)},
            passed=passed,
            score=1.0 if passed else 0.0,
            severity=HARD,
            comment="" if passed else "; ".join(parts),
            trace=trace,
            meta={"root_z_end_m": round(root_z, 3), "settle_frames": n, "root_z_std_m": round(var, 4)},
        )
    ]


def check_seat_contact(ctx: CheckContext):
    """The pelvis actually rests on the chair (not floating above / clipping"""
    ep, tr, cfg = ctx.episode, ctx.tracks, ctx.cfg
    if tr is None or getattr(tr, "scene_min_dist", None) is None:
        return [skip("seat_contact", "no scene distances (floor-only scene)")]
    n = min(cfg.sit_settle_frames, ep.T)
    gap = float(np.nanmin(tr.scene_min_dist[-n:]))  # closest robot→scene point at rest
    passed = gap <= cfg.sit_seat_contact_tol_m
    trace = [
        f"nearest robot↔scene gap over last {n} frames {gap * 100:+.0f} cm "
        f"(seat-contact tol {cfg.sit_seat_contact_tol_m * 100:.0f} cm)",
        f"{'pelvis rests on the seat' if passed else 'hovering above the seat'} → {'PASS' if passed else 'FAIL'}",
    ]
    return [
        make_result(
            "seat_contact",
            value=round(gap, 4),
            passed=passed,
            score=graded_score(gap, cfg.sit_seat_contact_tol_m, 3 * cfg.sit_seat_contact_tol_m),
            severity=SOFT,
            comment=""
            if passed
            else f"pelvis not resting on the seat at end (nearest scene gap "
            f"{gap * 100:+.0f} cm > {cfg.sit_seat_contact_tol_m * 100:.0f} cm)",
            trace=trace,
            meta={"scene_gap_end_m": round(gap, 4), "settle_frames": n},
        )
    ]


@register
class SitsAtEnd(MotionQualityChecker):
    name = "sits_at_end"
    severity = HARD
    tasks = _SIT_TASKS
    description = "Robot ends seated (root z in the seated band) and settled."

    def check(self, ctx: CheckContext):
        return check_sits_at_end(ctx)[0]


@register
class SeatContact(MotionQualityChecker):
    name = "seat_contact"
    severity = SOFT
    needs_tracks = True
    tasks = _SIT_TASKS
    description = "Pelvis rests on the chair seat at the end (needs scene tracks)."

    def check(self, ctx: CheckContext):
        return check_seat_contact(ctx)[0]

