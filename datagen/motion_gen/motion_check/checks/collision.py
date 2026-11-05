#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Collision checkers — consume the signed-distance series in ``Tracks`` (all"""

from __future__ import annotations

import numpy as np

from motion_gen.motion_check.checks._util import graded_score
from motion_gen.motion_check.checks.base import CheckContext, MotionQualityChecker, register
from motion_gen.motion_check.schema import HARD, mask_to_intervals


@register
class SelfCollision(MotionQualityChecker):
    name = "self_collision"
    severity = HARD
    needs_tracks = True
    description = "Distinct limbs (arm↔arm/leg/torso/head, leg↔leg) do not interpenetrate."

    def check(self, ctx: CheckContext):
        tr, cfg = ctx.tracks, ctx.cfg
        if tr.self_min_dist is None:
            return self.skip("no self-distance series")
        dist = tr.self_min_dist  # signed; <0 = penetration
        tol = cfg.coll_self_pen_tol_m
        mask = dist < -tol
        max_pen = max(0.0, float(-np.min(dist)))
        passed = not bool(mask.any())
        pair = (tr.self_worst or {}).get("pair")
        trace = [
            f"min cross-limb signed distance {(-max_pen) * 100:+.1f} cm"
            + (f" ({pair[0]}↔{pair[1]})" if pair else "")
            + f" at frame {(tr.self_worst or {}).get('frame', '?')}",
            f"penetration tolerance {tol * 100:.1f} cm (hull-aware — MuJoCo geoms are convex hulls)",
            f"frames interpenetrating: {int(mask.sum())} → {'PASS' if passed else 'FAIL'}",
        ]
        return self.result(
            passed=passed,
            score=1.0 if passed else graded_score(max_pen, tol, tol + 0.05),
            value=round(max_pen, 4),
            time=mask_to_intervals(mask.tolist()),
            comment=""
            if passed
            else f"limbs interpenetrate up to {max_pen * 100:.1f} cm"
            + (f" — worst {pair[0]}↔{pair[1]}" if pair else ""),
            trace=trace,
            meta={"max_penetration_m": round(max_pen, 4), "worst": tr.self_worst},
        )


@register
class SceneCollision(MotionQualityChecker):
    name = "scene_collision"
    severity = HARD
    needs_tracks = True
    description = "Robot does not penetrate non-target scene or prop geometry."

    def check(self, ctx: CheckContext):
        tr, cfg = ctx.tracks, ctx.cfg
        if tr.scene_min_dist is None:
            return self.skip("MuJoCo scene has no furniture/prop geometry (floor-only); floor covered by foot_sink")
        dist = tr.scene_min_dist
        tol = cfg.coll_scene_pen_tol_m
        mask = dist < -tol
        max_pen = max(0.0, float(-np.min(dist)))
        passed = not bool(mask.any())
        pair = (tr.scene_worst or {}).get("pair")
        trace = [
            f"min robot↔scene signed distance {(-max_pen) * 100:+.1f} cm" + (f" ({pair[0]}↔{pair[1]})" if pair else ""),
            f"penetration tolerance {tol * 100:.1f} cm",
            f"frames penetrating: {int(mask.sum())} → {'PASS' if passed else 'FAIL'}",
        ]
        return self.result(
            passed=passed,
            score=1.0 if passed else graded_score(max_pen, tol, tol + 0.05),
            value=round(max_pen, 4),
            time=mask_to_intervals(mask.tolist()),
            comment=""
            if passed
            else f"robot penetrates scene up to {max_pen * 100:.1f} cm"
            + (f" — worst {pair[0]}↔{pair[1]}" if pair else ""),
            trace=trace,
            meta={"max_penetration_m": round(max_pen, 4), "worst": tr.scene_worst},
        )
