#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Dynamics checkers — pure numpy on the raw qpos, no MuJoCo needed."""

from __future__ import annotations

import numpy as np

from motion_gen.motion_check.checks._util import graded_score, step_mask_to_intervals
from motion_gen.motion_check.checks.base import CheckContext, MotionQualityChecker, register
from motion_gen.motion_check.schema import HARD, SOFT, Status, frames_to_intervals, mask_to_intervals


@register
class Teleport(MotionQualityChecker):
    name = "teleport"
    severity = HARD
    description = "Root position must not jump discontinuously between frames."

    def check(self, ctx: CheckContext):
        ep, cfg = ctx.episode, ctx.cfg
        if ep.T < 2:
            return self.skip("sequence too short")
        step = np.linalg.norm(np.diff(ep.root_xyz, axis=0), axis=1)  # m/frame
        thr = cfg.dyn_teleport_step_m
        mask = step > thr
        mx, at = float(step.max()), int(np.argmax(step)) + 1
        passed = not bool(mask.any())
        trace = [
            f"per-frame root displacement: max {mx * 100:.1f} cm/frame ({mx * ep.fps:.1f} m/s) at frame {at}",
            f"threshold {thr * 100:.0f} cm/frame ({thr * ep.fps:.1f} m/s)",
            f"{'0' if passed else int(mask.sum())} frame(s) exceed → "
            f"{'PASS' if passed else 'FAIL'}" + (f"  (seams at {ep.seams})" if ep.seams else ""),
        ]
        return self.result(
            passed=passed,
            score=1.0 if passed else 0.0,
            value=round(mx, 4),
            time=step_mask_to_intervals(mask, ep.T),
            comment="" if passed else f"root jumps up to {mx:.2f} m/frame — discontinuous (likely a bad segment seam)",
            trace=trace,
            meta={"max_m_per_frame": round(mx, 4), "seams": ep.seams},
        )


@register
class ExcessiveSpeed(MotionQualityChecker):
    name = "excessive_speed"
    severity = SOFT
    description = "Root translation and joint rotation stay within plausible G1 speeds."

    def check(self, ctx: CheckContext):
        ep, cfg = ctx.episode, ctx.cfg
        if ep.T < 2:
            return self.skip("sequence too short")
        fps = ep.fps
        root_sp = np.linalg.norm(np.diff(ep.root_xyz[:, :2], axis=0), axis=1) * fps
        joint_sp = np.abs(np.diff(ep.joints, axis=0)) * fps
        jmax_t = joint_sp.max(axis=1) if joint_sp.size else np.zeros(0)
        rmax = float(root_sp.max()) if root_sp.size else 0.0
        jmax = float(jmax_t.max()) if jmax_t.size else 0.0
        rthr, jthr = cfg.dyn_speed_root_mps, cfg.dyn_speed_joint_radps
        mask = root_sp > rthr
        if jmax_t.size:
            mask = mask | (jmax_t > jthr)
        passed = rmax <= rthr and jmax <= jthr
        score = min(graded_score(rmax, rthr, 2 * rthr), graded_score(jmax, jthr, 2 * jthr))
        trace = [
            f"max root speed {rmax:.2f} m/s vs limit {rthr} m/s → {'ok' if rmax <= rthr else 'OVER'}",
            f"max joint speed {jmax:.1f} rad/s vs limit {jthr} rad/s → {'ok' if jmax <= jthr else 'OVER'}",
            f"{'PASS' if passed else 'FAIL'} (score {score:.2f})",
        ]
        return self.result(
            passed=passed,
            score=score,
            value={"root_mps": round(rmax, 3), "joint_radps": round(jmax, 3)},
            time=step_mask_to_intervals(mask, ep.T),
            comment="" if passed else f"moves too fast: root {rmax:.2f} m/s, joint {jmax:.1f} rad/s",
            trace=trace,
            meta={"root_thresh_mps": rthr, "joint_thresh_radps": jthr},
        )


@register
class SuddenJump(MotionQualityChecker):
    name = "sudden_jump"
    severity = HARD
    description = "Root must not ballistically pop up (rise then fall back)."

    def check(self, ctx: CheckContext):
        ep, cfg = ctx.episode, ctx.cfg
        if ep.T < 5:
            return self.skip("sequence too short")
        rz = ep.root_xyz[:, 2]
        W = max(3, int(round(0.33 * ep.fps)))
        prom = np.zeros(ep.T)
        for t in range(ep.T):
            lo, hi = rz[max(0, t - W) : t], rz[t + 1 : t + 1 + W]
            if lo.size and hi.size:
                prom[t] = rz[t] - max(lo.min(), hi.min())
        thr = cfg.dyn_jump_prominence_m
        mask = prom > thr
        mx, at = float(prom.max()), int(np.argmax(prom))
        passed = not bool(mask.any())
        trace = [
            f"peak prominence (rise above baseline that later reverses) over a "
            f"{W}-frame window: max {mx * 100:.0f} cm at frame {at}",
            f"threshold {thr * 100:.0f} cm — a monotonic stand-up scores ~0",
            f"{'no ballistic pop' if passed else f'pop of {mx * 100:.0f} cm'} → {'PASS' if passed else 'FAIL'}",
        ]
        return self.result(
            passed=passed,
            score=1.0 if passed else graded_score(mx, thr, thr + 0.20),
            value=round(mx, 3),
            time=mask_to_intervals(mask.tolist()),
            comment="" if passed else f"root pops up {mx * 100:.0f} cm then drops back — an unphysical jump",
            trace=trace,
            meta={"peak_prominence_m": round(mx, 3), "threshold_m": thr},
        )


@register
class SmallHop(MotionQualityChecker):
    name = "small_hop"
    severity = SOFT  # promoted to HARD for sit_on_chair (fires only on bad there)
    description = "Root shouldn't hop/bounce (a single sub-jump pop, or repeated bounces)."

    def check(self, ctx: CheckContext):
        from scipy.signal import find_peaks

        ep, cfg = ctx.episode, ctx.cfg
        if ep.T < 5:
            return self.skip("sequence too short")
        rz = ep.root_xyz[:, 2]
        pk, pr = find_peaks(rz, prominence=cfg.dyn_hop_repeat_prom_m)
        proms = pr["prominences"] if len(pk) else np.zeros(0)
        single = [
            (int(pk[j]), float(proms[j]))
            for j in range(len(pk))
            if cfg.dyn_hop_prominence_m <= proms[j] < cfg.dyn_jump_prominence_m
        ]
        n_bounce = int((proms > cfg.dyn_hop_repeat_prom_m).sum())
        repeated = n_bounce >= cfg.dyn_hop_repeat_n
        mx = float(proms.max()) if proms.size else 0.0
        passed = not (single or repeated)
        mask = np.zeros(ep.T, dtype=bool)
        if not passed:  # mark a small window around each bounce peak
            thr = cfg.dyn_hop_repeat_prom_m if repeated else cfg.dyn_hop_prominence_m
            for j in range(len(pk)):
                if proms[j] > thr:
                    mask[max(0, pk[j] - 2) : min(ep.T, pk[j] + 3)] = True
        trace = [
            f"pelvis bounce peaks (find_peaks prominence): {n_bounce} > "
            f"{cfg.dyn_hop_repeat_prom_m * 100:.0f} cm, biggest {mx * 100:.0f} cm",
            f"flag if a single bounce in [{cfg.dyn_hop_prominence_m * 100:.0f},"
            f"{cfg.dyn_jump_prominence_m * 100:.0f}) cm OR >= {cfg.dyn_hop_repeat_n} bounces "
            f"(good gait bob is one <=6.5 cm bob)",
            f"single={bool(single)} repeated={repeated} → {'PASS' if passed else 'FAIL'}",
        ]
        return self.result(
            passed=passed,
            score=1.0 if passed else graded_score(mx, cfg.dyn_hop_prominence_m, cfg.dyn_jump_prominence_m),
            value=round(mx, 3),
            time=mask_to_intervals(mask.tolist()),
            comment=""
            if passed
            else (
                f"root hops {n_bounce}× (repeated bounces)"
                if repeated
                else f"root does a small hop of {mx * 100:.0f} cm"
            ),
            trace=trace,
            meta={"peak_hop_m": round(mx, 3), "n_bounce": n_bounce},
        )


@register
class CrabWalk(MotionQualityChecker):
    name = "crab_walk"
    severity = SOFT  # promoted to HARD for sit_on_chair
    description = "Locomotion is mostly forward, not sideways (no crab / side-walking)."

    def check(self, ctx: CheckContext):
        ep, cfg = ctx.episode, ctx.cfg
        if ep.T < 3:
            return self.skip("sequence too short")
        q = ep.root_quat_wxyz
        yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        fwd = np.stack([np.cos(yaw), np.sin(yaw)], axis=1)[:-1]
        lat = np.stack([-np.sin(yaw), np.cos(yaw)], axis=1)[:-1]
        dv = np.diff(ep.root_xyz[:, :2], axis=0) * ep.fps  # root planar velocity
        fspd, lspd = np.abs((dv * fwd).sum(1)), np.abs((dv * lat).sum(1))
        moving = (fspd + lspd) > cfg.crab_min_speed_mps
        if moving.sum() < 5:
            return self.skip("not enough locomotion to judge direction")
        latfrac = float((lspd[moving] / (fspd[moving] + lspd[moving] + 1e-6)).mean())
        thr = cfg.crab_lateral_frac
        passed = latfrac <= thr
        trace = [
            f"mean lateral fraction of moving translation: {latfrac:.2f}",
            f"threshold {thr:.2f} (forward walks stay <=0.30)",
            f"{'forward' if passed else 'crab/side-walk'} → {'PASS' if passed else 'FAIL'}",
        ]
        return self.result(
            passed=passed,
            score=1.0 if passed else graded_score(latfrac, thr, 0.7),
            value=round(latfrac, 3),
            time=None,
            comment="" if passed else f"crab/side-walk: {latfrac * 100:.0f}% of motion is lateral",
            trace=trace,
            meta={"lateral_fraction": round(latfrac, 3)},
        )


@register
class UpperBodySpeed(MotionQualityChecker):
    name = "upper_body_speed"
    severity = SOFT  # promoted to HARD for sit_on_chair
    description = "Upper body (shoulders) moves at a natural speed — no jerky/frantic arms."
    _SHOULDERS = [15, 16, 17, 22, 23, 24]  # L/R shoulder pitch,roll,yaw in joints[0:29]

    def check(self, ctx: CheckContext):
        ep, cfg = ctx.episode, ctx.cfg
        if ep.T < 3:
            return self.skip("sequence too short")
        v = np.abs(np.diff(ep.joints[:, self._SHOULDERS], axis=0)) * ep.fps  # rad/s
        p95, mx = float(np.percentile(v, 95)), float(v.max())
        thr = cfg.arm_speed_p95_radps
        passed = p95 <= thr
        fast = np.zeros(ep.T, dtype=bool)
        fast[1:] = (v > thr).any(axis=1)
        trace = [
            f"shoulder angular speed: p95 {p95:.2f}, peak {mx:.2f} rad/s",
            f"threshold p95 <= {thr:.2f} rad/s (calm arms; good motions <=0.96)",
            f"{'natural' if passed else 'jerky/frantic arms'} → {'PASS' if passed else 'FAIL'}",
        ]
        return self.result(
            passed=passed,
            score=1.0 if passed else graded_score(p95, thr, 3 * thr),
            value=round(p95, 3),
            time=mask_to_intervals(fast.tolist()),
            comment="" if passed else f"jerky upper body: shoulder p95 speed {p95:.2f} rad/s",
            trace=trace,
            meta={"p95_radps": round(p95, 3), "peak_radps": round(mx, 3)},
        )


@register
class TurnRate(MotionQualityChecker):
    name = "turn_rate"
    severity = SOFT
    description = "Turn (yaw) angular speed stays within a plausible range."

    def check(self, ctx: CheckContext):
        ep, cfg = ctx.episode, ctx.cfg
        if ep.T < 2:
            return self.skip("sequence too short")
        q = ep.root_quat_wxyz
        yaw = np.unwrap(np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)))
        rate = np.abs(np.diff(yaw)) * ep.fps * 180.0 / np.pi  # deg/s
        thr = cfg.dyn_turn_rate_dps
        mask = rate > thr
        mx, at = float(rate.max()), int(np.argmax(rate)) + 1
        passed = not bool(mask.any())
        trace = [
            f"peak yaw rate {mx:.0f} deg/s at frame {at}",
            f"threshold {thr:.0f} deg/s (plausible turns peak ~230)",
            f"{int(mask.sum())} frame(s) over → {'PASS' if passed else 'FAIL'}",
        ]
        return self.result(
            passed=passed,
            score=1.0 if passed else graded_score(mx, thr, 2 * thr),
            value=round(mx, 1),
            time=step_mask_to_intervals(mask, ep.T),
            comment="" if passed else f"turns too fast: peak yaw rate {mx:.0f} deg/s",
            trace=trace,
            meta={"peak_dps": round(mx, 1), "threshold_dps": thr},
        )


@register
class VerticalDrift(MotionQualityChecker):
    name = "vertical_drift"
    severity = SOFT
    description = "Root shouldn't rise 'without effective movement' (up while barely translating)."

    def check(self, ctx: CheckContext):
        ep, cfg = ctx.episode, ctx.cfg
        W = max(3, int(round(cfg.dyn_drift_win_s * ep.fps)))
        if ep.T < W + 1:
            return self.skip("sequence too short")
        rz = ep.root_xyz[:, 2]
        xy = ep.root_xyz[:, :2]
        up = cfg.dyn_drift_up_m
        drift = np.zeros(ep.T)  # per-window rise attributed to start frame
        for s in range(ep.T - W):
            e = s + W
            dz = rz[e] - rz[s]
            dxy = float(np.linalg.norm(xy[e] - xy[s]))
            monotonic = float((np.diff(rz[s : e + 1]) > -0.003).mean())  # mostly-rising
            s0, e2 = max(0, s - W), min(ep.T - 1, e + W)
            span_rise = rz[e2] - rz[s0]
            if (
                dxy < cfg.dyn_drift_horiz_max_m
                and up < dz < cfg.dyn_drift_max_m
                and span_rise < cfg.dyn_drift_max_m
                and monotonic > 0.8
            ):
                drift[s] = dz
        mask = drift > up
        mx, at = float(drift.max()), int(np.argmax(drift))
        passed = not bool(mask.any())
        trace = [
            f"scanned {W}-frame ({cfg.dyn_drift_win_s:.1f}s) windows for a monotonic root rise "
            f"while translating < {cfg.dyn_drift_horiz_max_m * 100:.0f} cm horizontally",
            f"biggest such rise: {mx * 100:.1f} cm (starts frame {at}); "
            f"threshold {up * 100:.0f} cm, ignores full stand-ups > {cfg.dyn_drift_max_m * 100:.0f} cm",
            f"{int(mask.sum())} drift window(s) → {'PASS' if passed else 'FAIL'}",
        ]
        return self.result(
            passed=passed,
            score=1.0 if passed else graded_score(mx, up, cfg.dyn_drift_max_m),
            value=round(mx, 3),
            time=mask_to_intervals(mask.tolist()),
            comment="" if passed else f"root drifts up {mx * 100:.0f} cm without effective (horizontal) movement",
            trace=trace,
            meta={"max_drift_m": round(mx, 3)},
        )


@register
class BackAndForth(MotionQualityChecker):
    name = "back_and_forth"
    severity = SOFT
    description = "Root should not repeatedly reverse travel direction (aimless wander)."

    def check(self, ctx: CheckContext):
        ep, cfg = ctx.episode, ctx.cfg
        if ep.T < 4:
            return self.skip("sequence too short")
        vel = np.diff(ep.root_xyz[:, :2], axis=0)
        speed = np.linalg.norm(vel, axis=1)
        reversals: list[int] = []
        last = None
        for i in range(len(vel)):
            if speed[i] < 0.02:  # ~0.6 m/s = "moving"
                continue
            head = np.arctan2(vel[i, 1], vel[i, 0])
            if last is not None:
                d = abs((head - last + np.pi) % (2 * np.pi) - np.pi)
                if d > np.deg2rad(120):
                    reversals.append(i + 1)
            last = head
        path_len = float(speed.sum())
        net = float(np.linalg.norm(ep.root_xyz[-1, :2] - ep.root_xyz[0, :2]))
        ratio = path_len / max(net, 1e-6)
        n = len(reversals)
        passed = n < cfg.dyn_backforth_reversals
        trace = [
            f"travel-direction reversals (>120°): {n}  (threshold {cfg.dyn_backforth_reversals})",
            f"path length {path_len:.2f} m vs net displacement {net:.2f} m (ratio {ratio:.1f})",
            f"{'PASS' if passed else 'FAIL'}",
        ]
        return self.result(
            passed=passed,
            score=1.0 if passed else graded_score(n, cfg.dyn_backforth_reversals, 2 * cfg.dyn_backforth_reversals),
            value={"reversals": n, "path_ratio": round(ratio, 2)},
            time=frames_to_intervals(reversals) if reversals else None,
            comment="" if passed else f"{n} direction reversals — robot moves back and forth",
            trace=trace,
            meta={"path_len_m": round(path_len, 2), "net_disp_m": round(net, 2)},
        )


@register
class Jitter(MotionQualityChecker):
    name = "jitter"
    severity = SOFT
    description = "Root motion is smooth (low high-frequency jerk)."

    def check(self, ctx: CheckContext):
        ep, cfg = ctx.episode, ctx.cfg
        if ep.T < 4:
            return self.skip("sequence too short")
        fps = ep.fps

        def rms_jerk(x):
            j = np.diff(x, n=3, axis=0) * (fps**3)
            return float(np.sqrt(np.mean(j**2))) if j.size else 0.0

        root_jerk, joint_jerk = rms_jerk(ep.root_xyz), rms_jerk(ep.joints)
        warn, fail = cfg.dyn_jitter_jerk_mps3, cfg.dyn_jitter_fail
        if root_jerk >= fail:
            status, passed = Status.FAIL, False
        elif root_jerk >= warn:
            status, passed = Status.WARN, True
        else:
            status, passed = Status.PASS, True
        trace = [
            f"root jerk RMS {root_jerk:.0f} m/s³ (joint jerk {joint_jerk:.0f} rad/s³)",
            f"warn ≥ {warn}, fail ≥ {fail}",
            f"{status.value.upper()}",
        ]
        return self.result(
            passed=passed,
            score=graded_score(root_jerk, warn, fail),
            status=status,
            value=round(root_jerk, 2),
            comment="" if status == Status.PASS else f"root jerk RMS {root_jerk:.0f} m/s³ — high-frequency vibration",
            trace=trace,
            meta={"root_jerk_mps3": round(root_jerk, 2), "joint_jerk_rad_s3": round(joint_jerk, 2)},
        )
