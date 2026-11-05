#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Foot checkers — consume the numpy ``Tracks`` (sole heights, floor, foot"""

from __future__ import annotations

import numpy as np

from motion_gen.motion_check.checks._util import graded_score, locomotion_mask
from motion_gen.motion_check.checks.base import CheckContext, MotionQualityChecker, register
from motion_gen.motion_check.schema import HARD, SOFT, mask_to_intervals


def _gap(tr) -> np.ndarray:
    """Per-frame, per-foot sole height above the floor ([T,2]); <0 = sunk."""
    return tr.foot_sole_z - tr.floor_z_foot


def _clip_floats(tr, cfg) -> float | None:
    """If the clip floats (the LOWER foot never comes within"""
    if not np.isfinite(tr.foot_sole_z).any():
        return None
    with np.errstate(invalid="ignore"):
        lowest = np.nanmin(_gap(tr), axis=1)  # lower foot per frame
    min_gap = float(np.nanmin(lowest))
    return min_gap if min_gap > cfg.float_gate_contact_m + 1e-4 else None


@register
class FootSink(MotionQualityChecker):
    name = "foot_sink"
    severity = HARD
    needs_tracks = True
    description = "No foot sole penetrates the (ray-cast) floor beyond tolerance."

    def check(self, ctx: CheckContext):
        tr, cfg = ctx.tracks, ctx.cfg
        if not np.isfinite(tr.foot_sole_z).any():
            return self.skip("no foot-sole geometry")
        fl = _clip_floats(tr, cfg)
        if fl is not None:
            return self.skip(f"clip floats (+{fl * 100:.0f} cm) — penetration inconclusive until grounded")
        gap = _gap(tr)
        with np.errstate(invalid="ignore"):
            deepest = np.nanmin(gap, axis=1)  # most-sunk foot per frame
        sunk = deepest < -cfg.foot_sink_tol_m
        frac = float(sunk.mean())
        max_sink = max(0.0, float(-np.nanmin(deepest)))
        passed = frac <= cfg.foot_sink_fail_frac
        score = graded_score(max_sink, cfg.foot_sink_tol_m, cfg.foot_sink_tol_m + 0.07)
        floor = float(np.nanmedian(tr.floor_z_foot))
        trace = [
            f"ray-cast floor under feet z ≈ {floor:.3f} m",
            f"deepest sole gap {(-max_sink) * 100:+.1f} cm (at frame {int(np.nanargmin(deepest))})",
            f"frames with a foot below floor-{cfg.foot_sink_tol_m * 100:.0f}cm: "
            f"{int(sunk.sum())}/{tr.T} ({frac * 100:.1f}%)",
            f"fail if >{cfg.foot_sink_fail_frac * 100:.0f}% of frames sink → {'PASS' if passed else 'FAIL'}",
        ]
        return self.result(
            passed=passed,
            score=score,
            value=round(max_sink, 4),
            time=mask_to_intervals(sunk.tolist()),
            comment=""
            if passed
            else f"foot sinks up to {max_sink * 100:.1f} cm below floor in {frac * 100:.0f}% of frames",
            trace=trace,
            meta={
                "max_sink_m": round(max_sink, 4),
                "frac_frames": round(frac, 4),
                "plant_offset_max_m": round(float(np.max(tr.plant_offset)), 4) if tr.plant_offset is not None else None,
            },
        )


@register
class FootFloat(MotionQualityChecker):
    name = "foot_float"
    severity = SOFT
    needs_tracks = True
    description = "Both feet not airborne together for a sustained run (robot floats)."

    def check(self, ctx: CheckContext):
        tr, cfg = ctx.tracks, ctx.cfg
        if not np.isfinite(tr.foot_sole_z).any():
            return self.skip("no foot-sole geometry")
        gap = _gap(tr)
        with np.errstate(invalid="ignore"):
            lowest = np.nanmin(gap, axis=1)  # lower foot per frame
        airborne = lowest > cfg.foot_float_tol_m
        intervals = mask_to_intervals(airborne.tolist()) or []
        sustained = [iv for iv in intervals if iv[1] - iv[0] + 1 >= cfg.foot_float_min_run]
        frac = float(airborne.mean())
        max_h = float(np.nanmax(lowest)) if airborne.any() else 0.0
        passed = len(sustained) == 0
        trace = [
            f"lower foot above floor by up to {max(0.0, max_h) * 100:.0f} cm",
            f"frames with BOTH feet >{cfg.foot_float_tol_m * 100:.0f}cm up: {int(airborne.sum())} "
            f"({frac * 100:.0f}%); sustained runs (≥{cfg.foot_float_min_run}f): {len(sustained)}",
            f"{'PASS' if passed else 'FAIL'}",
        ]
        return self.result(
            passed=passed,
            score=1.0 if passed else graded_score(frac, 0.0, 0.30),
            value=round(max(0.0, max_h), 4),
            time=sustained or None,
            comment=""
            if passed
            else f"both feet airborne (up to {max_h * 100:.0f} cm) for {len(sustained)} run(s) — robot floats",
            trace=trace,
            meta={"frac_frames_airborne": round(frac, 4)},
        )


@register
class StanceWidth(MotionQualityChecker):
    name = "stance_width"
    severity = SOFT
    needs_tracks = True
    description = "Lateral foot separation (stance width) stays in a plausible band."

    def check(self, ctx: CheckContext):
        ep, tr, cfg = ctx.episode, ctx.tracks, ctx.cfg
        if tr.foot_pos is None or ep.T < 2:
            return self.skip("no foot positions")
        q = ep.root_quat_wxyz
        yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        lat = np.stack([-np.sin(yaw), np.cos(yaw)], axis=1)  # [T,2]
        lr = tr.foot_pos[:, 0, :2] - tr.foot_pos[:, 1, :2]  # [T,2]
        width = np.abs(np.sum(lr * lat, axis=1))  # [T] lateral separation
        gap = _gap(tr)
        grounded = np.nanmax(gap, axis=1) < 0.10  # both feet near floor
        sel = width[grounded] if grounded.sum() >= 5 else width
        med, p90, mx = float(np.median(sel)), float(np.percentile(sel, 90)), float(sel.max())
        p10, mn = float(np.percentile(sel, 10)), float(sel.min())
        lo, hi = cfg.stance_width_min_m, cfg.stance_width_max_m
        wide = grounded & (width > hi)
        narrow = grounded & (width < lo)
        oob = wide | narrow
        frac = float(oob.sum()) / max(1, int(grounded.sum()))
        passed = frac <= cfg.stance_width_fail_frac
        sides = []
        if wide.any():
            sides.append(f"too wide (p90 {p90 * 100:.0f} cm > {hi * 100:.0f}) — legs splayed")
        if narrow.any():
            sides.append(f"too narrow (p10 {p10 * 100:.0f} cm < {lo * 100:.0f}) — ankles too close")
        trace = [
            f"lateral stance width over grounded frames: p10 {p10 * 100:.0f}, median "
            f"{med * 100:.0f}, p90 {p90 * 100:.0f} cm (min {mn * 100:.0f}, max {mx * 100:.0f})",
            f"plausible band {lo * 100:.0f}-{hi * 100:.0f} cm",
            f"out-of-band grounded frames: {int(oob.sum())} ({frac * 100:.0f}%) "
            f"[{int(wide.sum())} wide, {int(narrow.sum())} narrow] → "
            f"{'PASS' if passed else 'FAIL'}",
        ]
        wide_sc = graded_score(p90, hi, hi + 0.20) if wide.any() else 1.0
        narrow_sc = graded_score(lo - p10, 0.0, lo) if narrow.any() else 1.0
        return self.result(
            passed=passed,
            score=1.0 if passed else min(wide_sc, narrow_sc),
            value=round(p90 if wide.sum() >= narrow.sum() else p10, 3),
            time=mask_to_intervals(oob.tolist()),
            comment="" if passed else "stance " + "; ".join(sides),
            trace=trace,
            meta={
                "median_m": round(med, 3),
                "p10_m": round(p10, 3),
                "p90_m": round(p90, 3),
                "min_m": round(mn, 3),
                "max_m": round(mx, 3),
                "band_m": [lo, hi],
                "n_wide": int(wide.sum()),
                "n_narrow": int(narrow.sum()),
            },
        )


@register
class WalkPause(MotionQualityChecker):
    name = "walk_pause"
    severity = SOFT  # promoted to HARD for sit_on_chair
    needs_tracks = True
    description = "No pausing / standing mid-walk (gait stalls: translating but not stepping)."

    def check(self, ctx: CheckContext):
        ep, tr, cfg = ctx.episode, ctx.tracks, ctx.cfg
        if tr.foot_pos is None or ep.T < 5:
            return self.skip("no foot tracks / too short")
        if not ep.phases:
            return self.skip("needs WALK/TURN/SIT phase metadata to isolate locomotion")
        loco = locomotion_mask(ep, tr)  # WALK/TURN only (SIT excluded)
        if loco.sum() < 5:
            return self.skip("no locomotion phase")
        fh = np.nanmax(tr.foot_sole_z - tr.floor_z_foot, axis=1)  # higher foot
        v = np.concatenate([[0.0], np.linalg.norm(np.diff(ep.root_xyz[:, :2], axis=0), axis=1) * ep.fps])
        pause = loco & (fh < cfg.walk_pause_step_h_m) & (v > cfg.walk_pause_speed_mps)
        runs = mask_to_intervals(pause.tolist())
        min_frames = int(round(cfg.walk_pause_min_s * ep.fps))
        bad = [[s, e] for s, e in runs if e - s + 1 >= min_frames]
        longest = max((e - s + 1 for s, e in runs), default=0)
        passed = not bad
        trace = [
            f"longest no-step-while-gliding run in WALK/TURN: {longest} frames ({longest / ep.fps:.2f}s)",
            f"pause if >= {min_frames} frames ({cfg.walk_pause_min_s:.1f}s) of locomotion + "
            f"translating(>{cfg.walk_pause_speed_mps * 100:.0f}cm/s) + not-stepping(<{cfg.walk_pause_step_h_m * 100:.0f}cm)",
            f"{'no pause' if passed else f'{len(bad)} pause window(s)'} → {'PASS' if passed else 'FAIL'}",
        ]
        return self.result(
            passed=passed,
            score=1.0 if passed else graded_score(longest, min_frames, 3 * min_frames),
            value=round(longest / ep.fps, 2),
            time=bad or None,
            comment="" if passed else f"pauses/glides mid-walk for {longest / ep.fps:.1f}s without stepping",
            trace=trace,
            meta={"longest_pause_s": round(longest / ep.fps, 2)},
        )


@register
class WalkTorsoBend(MotionQualityChecker):
    name = "walk_torso_bend"
    severity = SOFT  # promoted to HARD for sit_on_chair
    needs_tracks = True  # needs tracks only for the phase-fallback floor; qpos carries the angle
    description = "Torso shouldn't bend forward during walking (spine flexion in the WALK/TURN phase)."

    def check(self, ctx: CheckContext):
        ep, tr, cfg = ctx.episode, ctx.tracks, ctx.cfg
        if ep.T < 3:
            return self.skip("sequence too short")
        if not ep.phases:
            return self.skip("needs WALK/TURN/SIT phase metadata to isolate locomotion")
        loco = locomotion_mask(ep, tr)
        if loco.sum() < 5:
            return self.skip("no locomotion phase")
        waist = np.degrees(ep.joints[:, 14])  # waist_pitch (spine forward flexion)
        gated = np.where(loco, waist, -99.0)
        thr = cfg.waist_bend_max_deg
        mask = gated > thr
        mx, at = float(gated.max()), int(np.argmax(gated))
        passed = not bool(mask.any())
        trace = [
            f"max forward spine flexion (waist_pitch) during WALK/TURN: {mx:.0f} deg at frame {at}",
            f"threshold {thr:.0f} deg (a normal sit-lean is confined to the SIT phase)",
            f"{'upright' if passed else f'bends {mx:.0f} deg mid-walk'} → {'PASS' if passed else 'FAIL'}",
        ]
        return self.result(
            passed=passed,
            score=1.0 if passed else graded_score(mx, thr, 3 * thr),
            value=round(mx, 1),
            time=mask_to_intervals(mask.tolist()),
            comment="" if passed else f"torso bends {mx:.0f} deg forward while walking (frame {at})",
            trace=trace,
            meta={"max_waist_pitch_deg": round(mx, 1)},
        )


@register
class WalkLegAngle(MotionQualityChecker):
    name = "walk_leg_angle"
    severity = SOFT  # promoted to HARD for sit_on_chair
    needs_tracks = True
    description = "Swing leg shouldn't lift too high (excessive foot/knee swing during walking)."

    def check(self, ctx: CheckContext):
        ep, tr, cfg = ctx.episode, ctx.tracks, ctx.cfg
        if tr.foot_pos is None or ep.T < 5:
            return self.skip("no foot tracks / too short")
        walk = locomotion_mask(ep, tr)  # WALK/TURN only (SIT excluded)
        fh = np.nanmax(tr.foot_sole_z - tr.floor_z_foot, axis=1)  # higher foot per frame
        fh_walk = np.where(walk, fh, -9.0)
        thr = cfg.leg_swing_max_m
        mask = fh_walk > thr
        mx, at = float(fh_walk.max()), int(np.argmax(fh_walk))
        passed = not bool(mask.any())
        trace = [
            f"max swing-foot height while walking: {mx * 100:.1f} cm at frame {at}",
            f"threshold {thr * 100:.0f} cm (a normal step lifts <=11 cm)",
            f"{'ok' if passed else f'leg swings too high ({mx * 100:.0f} cm)'} → {'PASS' if passed else 'FAIL'}",
        ]
        return self.result(
            passed=passed,
            score=1.0 if passed else graded_score(mx, thr, 3 * thr),
            value=round(mx, 3),
            time=mask_to_intervals(mask.tolist()),
            comment="" if passed else f"leg swings too high: foot lifts {mx * 100:.0f} cm (frame {at})",
            trace=trace,
            meta={"max_swing_foot_m": round(mx, 3), "frame": at},
        )


@register
class StepClearance(MotionQualityChecker):
    name = "step_clearance"
    severity = SOFT  # promoted to HARD for sit_on_chair
    needs_tracks = True
    description = "Swing feet must lift clear of the floor (no low shuffling / mincing steps)."

    def check(self, ctx: CheckContext):
        from scipy.signal import find_peaks

        ep, tr, cfg = ctx.episode, ctx.tracks, ctx.cfg
        if tr.foot_pos is None or ep.T < 10:
            return self.skip("no foot tracks / too short")
        floor_med = np.nanmedian(tr.floor_z_foot, axis=1)
        walk = (ep.root_xyz[:, 2] - floor_med) > cfg.step_clearance_walk_z
        if walk.sum() < 0.3 * ep.T:
            return self.skip("little locomotion (mostly sitting/standing)")
        heights = []
        for k in range(2):  # swing-foot peak heights per foot
            fz = np.where(walk, tr.foot_sole_z[:, k] - tr.floor_z_foot[:, k], 0.0)
            pk, _ = find_peaks(fz, prominence=0.015, height=0.01)
            heights += list(fz[pk])
        if len(heights) < 4:
            return self.skip("too few steps to judge clearance")
        p75 = float(np.percentile(heights, 75))
        med = float(np.median(heights))
        thr = cfg.step_clearance_min_m
        passed = p75 >= thr
        mask = np.zeros(ep.T, dtype=bool)
        for k in range(2):
            fz = np.where(walk, tr.foot_sole_z[:, k] - tr.floor_z_foot[:, k], 0.0)
            pk, _ = find_peaks(fz, prominence=0.015, height=0.01)
            for p in pk:
                if fz[p] < thr:
                    mask[max(0, p - 3) : min(ep.T, p + 4)] = True
        trace = [
            f"swing-foot peak clearance over walking phase: median {med * 100:.0f}, "
            f"p75 {p75 * 100:.0f} cm ({len(heights)} steps)",
            f"threshold p75 >= {thr * 100:.0f} cm (clean walks lift to ~7-8 cm)",
            f"{'clears' if passed else 'shuffles/minces (low steps)'} → {'PASS' if passed else 'FAIL'}",
        ]
        return self.result(
            passed=passed,
            score=1.0 if passed else graded_score(thr - p75, 0.0, thr),
            value=round(p75, 3),
            time=mask_to_intervals(mask.tolist()),
            comment=""
            if passed
            else f"low/shuffling steps: p75 swing clearance {p75 * 100:.0f} cm (< {thr * 100:.0f})",
            trace=trace,
            meta={"p75_clearance_m": round(p75, 3), "median_clearance_m": round(med, 3), "n_steps": len(heights)},
        )


@register
class FlightPhase(MotionQualityChecker):
    name = "flight_phase"
    severity = SOFT  # promoted to HARD for sit_on_chair
    needs_tracks = True
    description = "No flight phase while walking (both feet off the floor = a hop)."

    def check(self, ctx: CheckContext):
        ep, tr, cfg = ctx.episode, ctx.tracks, ctx.cfg
        if tr.foot_pos is None or ep.T < 5:
            return self.skip("no foot tracks / too short")
        floor_med = np.nanmedian(tr.floor_z_foot, axis=1)
        walk = (ep.root_xyz[:, 2] - floor_med) > cfg.step_clearance_walk_z
        lower = np.nanmin(tr.foot_sole_z - tr.floor_z_foot, axis=1)  # the LOWER foot
        airborne = walk & (lower > cfg.flight_min_m)  # both feet up while walking
        flight = np.zeros(ep.T, dtype=bool)
        for s, e in mask_to_intervals(airborne.tolist()):
            if e - s + 1 >= cfg.flight_min_run:
                flight[s : e + 1] = True
        mx = float(np.nanmax(np.where(walk, lower, -9)))
        passed = not bool(flight.any())
        trace = [
            f"lower-foot (both-feet-airborne) clearance while walking: max {mx * 100:.1f} cm",
            f"flight if lower foot > {cfg.flight_min_m * 100:.0f} cm for >= "
            f"{cfg.flight_min_run} frames (good walks keep a stance foot down, <=1.7 cm)",
            f"{int(flight.sum())} flight frame(s) → {'PASS' if passed else 'FAIL'}",
        ]
        return self.result(
            passed=passed,
            score=1.0 if passed else graded_score(mx, cfg.flight_min_m, 0.10),
            value=round(mx, 3),
            time=mask_to_intervals(flight.tolist()),
            comment="" if passed else f"hop: both feet airborne (lower foot up {mx * 100:.0f} cm)",
            trace=trace,
            meta={"max_lower_foot_m": round(mx, 3)},
        )


@register
class FeetClose(MotionQualityChecker):
    name = "feet_close"
    severity = SOFT  # promoted to HARD for sit_on_chair (fires only on bad there)
    needs_tracks = True
    description = "The two feet shouldn't get abnormally close / collide."

    def check(self, ctx: CheckContext):
        tr, cfg = ctx.tracks, ctx.cfg
        if tr.foot_pos is None or tr.T < 1:
            return self.skip("no foot positions")
        d = np.linalg.norm(tr.foot_pos[:, 0, :2] - tr.foot_pos[:, 1, :2], axis=1)
        thr = cfg.feet_close_min_m
        mask = d < thr
        mn, at = float(d.min()), int(np.argmin(d))
        passed = not bool(mask.any())
        trace = [
            f"min horizontal inter-foot distance: {mn * 100:.1f} cm at frame {at}",
            f"threshold {thr * 100:.0f} cm (plausible motions stay > ~13 cm apart)",
            f"{int(mask.sum())} frame(s) closer → {'PASS' if passed else 'FAIL'}",
        ]
        return self.result(
            passed=passed,
            score=1.0 if passed else graded_score(thr - mn, 0.0, thr),
            value=round(mn, 3),
            time=mask_to_intervals(mask.tolist()),
            comment="" if passed else f"feet get too close: min {mn * 100:.0f} cm apart (frame {at})",
            trace=trace,
            meta={"min_foot_dist_m": round(mn, 3), "frame": at},
        )


def _planted_slip(tr, cfg):
    """(planted[T-1,2], skating[T-1,2], slip[T-1,2] m/s, p95_slip) for a whole-foot"""
    gap_hi = (tr.foot_sole_z_max - tr.floor_z_foot) if tr.foot_sole_z_max is not None else _gap(tr)
    flat_low = gap_hi < cfg.foot_contact_h_m
    qualified = np.zeros_like(flat_low)
    for i in range(2):
        for s, e in mask_to_intervals(flat_low[:, i].tolist()):
            if e - s + 1 >= cfg.foot_skate_min_contact_run:
                qualified[s : e + 1, i] = True
    slip = np.linalg.norm(np.diff(tr.foot_pos[:, :, :2], axis=0), axis=2) * tr.fps
    vert = np.abs(np.diff(tr.foot_sole_z, axis=0)) * tr.fps
    planted = qualified[:-1] & qualified[1:] & (vert < cfg.foot_skate_vert_vel_mps)
    skating = planted & (slip > cfg.foot_skate_speed_mps)
    ps = slip[planted] if planted.any() else np.zeros(1)
    return planted, skating, slip, float(np.percentile(ps, 95))


@register
class FootSkate(MotionQualityChecker):
    name = "foot_skate"
    severity = HARD
    needs_tracks = True
    description = "A flat-planted foot must not slide (ice-skate) along the floor."

    def check(self, ctx: CheckContext):
        tr, cfg = ctx.tracks, ctx.cfg
        if tr.T < 2 or not np.isfinite(tr.foot_sole_z).any():
            return self.skip("no foot tracks / too short")
        fl = _clip_floats(tr, cfg)
        if fl is not None:
            return self.skip(f"clip floats (+{fl * 100:.0f} cm) — slip inconclusive until grounded")
        planted, skating, slip, p95 = _planted_slip(tr, cfg)
        n_planted, n_skate = int(planted.sum()), int(skating.sum())
        frac = (n_skate / n_planted) if n_planted else 0.0
        ps = slip[planted] if (n_planted and planted.any()) else np.zeros(1)
        med, mx = float(np.median(ps)), float(ps.max())
        passed = frac <= cfg.foot_skate_fail_frac
        score = 1.0 if passed else graded_score(p95, cfg.foot_skate_speed_mps, 4 * cfg.foot_skate_speed_mps)
        frame_mask = np.zeros(tr.T, dtype=bool)
        frame_mask[1:] = skating.any(axis=1)
        trace = [
            f"planted-foot frames (whole foot <{cfg.foot_contact_h_m * 100:.0f}cm, "
            f"≥{cfg.foot_skate_min_contact_run}f run, low vert-vel): {n_planted}",
            f"planted slip: median {med * 100:.0f}, p95 {p95 * 100:.0f}, peak {mx * 100:.0f} cm/s",
            f"frames slipping >{cfg.foot_skate_speed_mps * 100:.0f}cm/s: {n_skate} ({frac * 100:.0f}% of planted)",
            f"fail if >{cfg.foot_skate_fail_frac * 100:.0f}% → {'PASS' if passed else 'FAIL'}",
        ]
        return self.result(
            passed=passed,
            score=score,
            value=round(p95, 3),
            time=mask_to_intervals(frame_mask.tolist()),
            comment=""
            if passed
            else f"planted foot slides: p95 {p95 * 100:.0f} cm/s, peak {mx * 100:.0f} cm/s "
            f"in {frac * 100:.0f}% of planted frames",
            trace=trace,
            meta={
                "p95_slip_mps": round(p95, 3),
                "median_slip_mps": round(med, 3),
                "max_slip_mps": round(mx, 3),
                "frac_planted_skating": round(frac, 4),
                "n_planted_steps": n_planted,
            },
        )


@register
class FootSkateSevere(MotionQualityChecker):
    name = "foot_skate_severe"
    severity = HARD  # egregious skate blocks on EVERY tier (not softened for sit)
    needs_tracks = True
    description = "No egregious planted-foot skating (well above the tolerated slip level)."

    def check(self, ctx: CheckContext):
        tr, cfg = ctx.tracks, ctx.cfg
        if tr.T < 2 or not np.isfinite(tr.foot_sole_z).any():
            return self.skip("no foot tracks / too short")
        fl = _clip_floats(tr, cfg)
        if fl is not None:
            return self.skip(f"clip floats (+{fl * 100:.0f} cm) — slip inconclusive until grounded")
        planted, skating, slip, p95 = _planted_slip(tr, cfg)
        thr = cfg.foot_skate_severe_mps
        passed = p95 <= thr
        frame_mask = np.zeros(tr.T, dtype=bool)
        frame_mask[1:] = planted.any(axis=1) & (slip.max(axis=1) > thr)
        trace = [
            f"planted-slip p95 {p95 * 100:.0f} cm/s",
            f"egregious-skate threshold {thr * 100:.0f} cm/s (tolerated slip tops ~33)",
            f"{'ok' if passed else 'EGREGIOUS'} → {'PASS' if passed else 'FAIL'}",
        ]
        return self.result(
            passed=passed,
            score=1.0 if passed else graded_score(p95, thr, 3 * thr),
            value=round(p95, 3),
            time=mask_to_intervals(frame_mask.tolist()),
            comment="" if passed else f"egregious foot skating: planted slip p95 {p95 * 100:.0f} cm/s",
            trace=trace,
            meta={"p95_slip_mps": round(p95, 3), "severe_mps": thr},
        )
