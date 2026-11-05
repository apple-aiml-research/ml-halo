#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Discrete defect checks — pure numpy on the raw qpos, no MuJoCo needed."""

from __future__ import annotations

import numpy as np

from motion_gen.motion_check.checks._util import _G as _CURV_G
from motion_gen.motion_check.checks._util import graded_score, root_curvature_g, step_mask_to_intervals
from motion_gen.motion_check.checks.base import CheckContext, MotionQualityChecker, register
from motion_gen.motion_check.schema import HARD, SOFT, mask_to_intervals

_G1_JOINT_LIMITS = [
    ("left_hip_pitch", -2.5307, 2.8798),
    ("left_hip_roll", -0.5236, 2.9671),
    ("left_hip_yaw", -2.7576, 2.7576),
    ("left_knee", -0.087267, 2.8798),
    ("left_ankle_pitch", -0.87267, 0.5236),
    ("left_ankle_roll", -0.2618, 0.2618),
    ("right_hip_pitch", -2.5307, 2.8798),
    ("right_hip_roll", -2.9671, 0.5236),
    ("right_hip_yaw", -2.7576, 2.7576),
    ("right_knee", -0.087267, 2.8798),
    ("right_ankle_pitch", -0.87267, 0.5236),
    ("right_ankle_roll", -0.2618, 0.2618),
    ("waist_yaw", -2.618, 2.618),
    ("waist_roll", -0.52, 0.52),
    ("waist_pitch", -0.52, 0.52),
    ("left_shoulder_pitch", -3.0892, 2.6704),
    ("left_shoulder_roll", -1.5882, 2.2515),
    ("left_shoulder_yaw", -2.618, 2.618),
    ("left_elbow", -1.0472, 2.0944),
    ("left_wrist_roll", -1.97222, 1.97222),
    ("left_wrist_pitch", -1.61443, 1.61443),
    ("left_wrist_yaw", -1.61443, 1.61443),
    ("right_shoulder_pitch", -3.0892, 2.6704),
    ("right_shoulder_roll", -2.2515, 1.5882),
    ("right_shoulder_yaw", -2.618, 2.618),
    ("right_elbow", -1.0472, 2.0944),
    ("right_wrist_roll", -1.97222, 1.97222),
    ("right_wrist_pitch", -1.61443, 1.61443),
    ("right_wrist_yaw", -1.61443, 1.61443),
]
_JOINT_NAMES = [n for n, _, _ in _G1_JOINT_LIMITS]
_JOINT_LO = np.array([lo for _, lo, _ in _G1_JOINT_LIMITS])
_JOINT_HI = np.array([hi for _, _, hi in _G1_JOINT_LIMITS])


@register
class JointOverLimit(MotionQualityChecker):
    name = "joint_over_limit"
    severity = HARD
    description = "No joint angle exceeds its mechanical stop (beyond a small tolerance)."

    def check(self, ctx: CheckContext):
        ep, cfg = ctx.episode, ctx.cfg
        j = ep.joints
        if j.shape[1] < len(_JOINT_NAMES):
            return self.skip(f"expected {len(_JOINT_NAMES)} joints, got {j.shape[1]}")
        j = j[:, : len(_JOINT_NAMES)]
        over = np.maximum(0.0, np.maximum(_JOINT_LO - j, j - _JOINT_HI))  # [T,29] rad
        worst_per_joint = over.max(axis=0)  # [29] rad
        ji = int(np.argmax(worst_per_joint))
        worst_rad = float(worst_per_joint[ji])
        worst_deg = np.degrees(worst_rad)
        tol_deg = cfg.joint_over_limit_tol_deg
        over_deg = np.degrees(over)
        flagged = (over_deg > tol_deg).any(axis=1)
        passed = worst_deg <= tol_deg
        maxes = np.degrees(worst_per_joint)
        tiers = {f">{t}deg": int((maxes > t).sum()) for t in (0.5, 1, 3, 10, 20)}
        per_joint = {_JOINT_NAMES[k]: round(float(maxes[k]), 2) for k in range(len(_JOINT_NAMES)) if maxes[k] > 0.5}
        wname = _JOINT_NAMES[ji]
        at = int(np.argmax(over[:, ji])) if worst_rad > 0 else 0
        trace = [
            f"worst joint overshoot: {wname} by {worst_deg:.1f}° "
            f"(range [{np.degrees(_JOINT_LO[ji]):.1f}, {np.degrees(_JOINT_HI[ji]):.1f}]°) at frame {at}",
            f"joints over by >0.5/1/3/10/20°: {tiers['>0.5deg']}/{tiers['>1deg']}/"
            f"{tiers['>3deg']}/{tiers['>10deg']}/{tiers['>20deg']}",
            f"tolerance {tol_deg:.1f}° (sub-tol saturation, e.g. ankle_roll, is cosmetic) → "
            f"{'PASS' if passed else 'FAIL'}",
        ]
        return self.result(
            passed=passed,
            score=1.0 if passed else graded_score(worst_deg, tol_deg, tol_deg + 20.0),
            value=round(worst_deg, 2),
            time=mask_to_intervals(flagged.tolist()),
            comment="" if passed else f"{wname} exceeds its limit by {worst_deg:.1f}° — infeasible pose",
            trace=trace,
            meta={
                "worst_joint": wname,
                "worst_deg": round(worst_deg, 2),
                "tol_deg": tol_deg,
                "tiers": tiers,
                "per_joint_deg": per_joint,
            },
        )


@register
class JointFlip(MotionQualityChecker):
    name = "joint_flip"
    severity = HARD
    description = "No single joint jumps more than ~1 rad between consecutive frames (angle wrap)."

    def check(self, ctx: CheckContext):
        ep, cfg = ctx.episode, ctx.cfg
        if ep.T < 2:
            return self.skip("sequence too short")
        d = np.abs(np.diff(ep.joints, axis=0))  # [T-1, njoints] rad/frame
        thr = cfg.joint_flip_rad
        step_max = d.max(axis=1) if d.size else np.zeros(0)
        mask = step_max > thr
        mx = float(step_max.max()) if step_max.size else 0.0
        at = int(np.argmax(step_max)) + 1 if step_max.size else 0
        ji = int(np.argmax(d[at - 1])) if step_max.size else 0
        jname = _JOINT_NAMES[ji] if ji < len(_JOINT_NAMES) else f"joint{ji}"
        passed = not bool(mask.any())
        wrap = abs(mx - 2 * np.pi) < 0.3  # ~2π signature = an angle wrap
        trace = [
            f"largest single-joint step: {jname} {mx:.2f} rad ({np.degrees(mx):.0f}°) at frame {at}",
            f"threshold {thr:.1f} rad/frame" + ("  (≈2π → angle-wrap teleport)" if wrap else ""),
            f"{int(mask.sum())} flip frame(s) → {'PASS' if passed else 'FAIL'}",
        ]
        return self.result(
            passed=passed,
            score=1.0 if passed else graded_score(mx, thr, 2 * np.pi),
            value=round(mx, 3),
            time=step_mask_to_intervals(mask, ep.T),
            comment=""
            if passed
            else f"{jname} jumps {mx:.2f} rad in one frame — "
            + ("angle-wrap teleport" if wrap else "discontinuous joint motion"),
            trace=trace,
            meta={"flip_joint": jname, "max_rad": round(mx, 3), "is_wrap": bool(wrap)},
        )


@register
class RootCurvature(MotionQualityChecker):
    name = "root_curvature"
    severity = HARD
    description = "Pelvis path doesn't bend implausibly sharply (bounded acceleration)."

    def check(self, ctx: CheckContext):
        ep, cfg = ctx.episode, ctx.cfg
        if ep.T < 3:
            return self.skip("sequence too short")
        dt = 1.0 / ep.fps
        g = root_curvature_g(ep.root_xyz, ep.fps)  # [T-2] in g
        thr = cfg.root_curv_g_max
        mask = g > thr
        mx = float(g.max()) if g.size else 0.0
        at = int(np.argmax(g)) + 1 if g.size else 0
        passed = not bool(mask.any())
        dev_max = mx * _CURV_G * dt * dt / 2.0
        full = np.zeros(ep.T, dtype=bool)
        full[1:-1] = mask
        trace = [
            f"peak root-path curvature {mx:.1f} g at frame {at} "
            f"(deviation {dev_max * 100:.1f} cm from the neighbour midpoint)",
            f"threshold {thr:.0f} g (LAFAN human motion tops ~18 g; >15 g is an artifact)",
            f"{int(mask.sum())} frame(s) over → {'PASS' if passed else 'FAIL'}",
        ]
        return self.result(
            passed=passed,
            score=1.0 if passed else graded_score(mx, thr, 3 * thr),
            value=round(mx, 1),
            time=mask_to_intervals(full.tolist()),
            comment="" if passed else f"root path bends at {mx:.0f} g — non-physical acceleration (frame {at})",
            trace=trace,
            meta={"peak_g": round(mx, 1), "threshold_g": thr},
        )


@register
class NonFinite(MotionQualityChecker):
    name = "non_finite"
    severity = HARD
    description = "qpos contains no NaN or infinity (corrupt-file guard)."

    def check(self, ctx: CheckContext):
        ep = ctx.episode
        finite = np.isfinite(ep.qpos)
        n_bad = int((~finite).sum())
        passed = n_bad == 0
        first = None
        if not passed:
            fr, col = np.argwhere(~finite)[0]
            first = (int(fr), int(col))
        trace = [
            f"non-finite (NaN/inf) entries in qpos[{ep.T},{ep.qpos.shape[1]}]: {n_bad}",
            f"first at (frame, col) = {first}" if first else "all finite",
            f"{'PASS' if passed else 'FAIL'}",
        ]
        return self.result(
            passed=passed,
            score=1.0 if passed else 0.0,
            value=n_bad,
            time=None,
            comment="" if passed else f"corrupt: {n_bad} NaN/inf value(s) in qpos (first at frame {first[0]})",
            trace=trace,
            meta={"n_nonfinite": n_bad, "first_frame_col": first},
        )


@register
class DeadClip(MotionQualityChecker):
    name = "dead_clip"
    severity = SOFT
    description = "The robot travels somewhere (isn't stuck in one spot)."

    def check(self, ctx: CheckContext):
        ep, cfg = ctx.episode, ctx.cfg
        if ep.T < 2:
            return self.skip("sequence too short")
        xy = ep.root_xyz[:, :2]
        excursion = float(np.linalg.norm(xy - xy[0], axis=1).max())  # farthest from start
        thr = cfg.dead_travel_m
        passed = excursion >= thr
        trace = [
            f"max root-xy excursion from start: {excursion * 100:.1f} cm",
            f"threshold {thr * 100:.0f} cm (below → the clip goes nowhere)",
            f"{'moves' if passed else 'DEAD'} → {'PASS' if passed else 'FAIL'}",
        ]
        return self.result(
            passed=passed,
            score=1.0 if passed else graded_score(thr - excursion, 0.0, thr),
            value=round(excursion, 3),
            time=None,
            comment="" if passed else f"dead clip: root moves only {excursion * 100:.0f} cm end to end",
            trace=trace,
            meta={"excursion_m": round(excursion, 3), "threshold_m": thr},
        )
