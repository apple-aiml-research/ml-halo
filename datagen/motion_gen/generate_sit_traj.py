#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Motion generation for the sit-on-chair task (stage 2 of docs/data_pipeline.md)."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

# Run as a script, sys.path[0] is motion_gen/, whose kimodo/ would shadow the kimodo package: use datagen/ instead.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != _HERE]
sys.path.insert(0, os.path.dirname(_HERE))

from motion_gen.tasks.sit import ARM_CHAIR_USDZ, SitBatchConfig, SitTask, canonical_scene  # noqa: E402
from motion_gen.kimodo.keyframes.sit_keyframes import build_sit_kfs_from_spec  # noqa: E402
from motion_gen.kimodo.motion_gen import (  # noqa: E402
    PLAN_3,
)

_ANCHOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kimodo", "keyframes")


def single_kf_path(name: str) -> str:
    """Resolve ``single_kf_<name>.npz``: the standing anchor, holding the pose in both"""
    repo = os.path.abspath(os.path.join(_ANCHOR_DIR, f"single_kf_{name}.npz"))
    if os.path.isfile(repo):
        return repo
    raise SystemExit(f"[sit] --single-standing-kf {name}: no anchor at {repo}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200, help="episodes for this shard")
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="base seed; per-episode seeds are derived from it, --shard-id and the episode index",
    )
    ap.add_argument(
        "--n-max-retry",
        type=int,
        default=5,
        help="max walk resamples per episode. Each round draws a fresh "
        "batch of walks and re-runs the TURN/SIT stages on the "
        "winner (a bad walk-end can make the tail unwinnable).",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=16,
        help="walk best-of-B: B seed-varied walks from one batched "
        "forward pass, ranked by _rank_key. TURN and SIT are "
        "single candidates chained off the winner, so 'the turn "
        "that feeds this sit' is unambiguous.",
    )
    ap.add_argument(
        "--stage-retry", type=int, default=2, help="per-stage retries for TURN and SIT before the WALK is resampled"
    )
    ap.add_argument(
        "--check-workers", type=int, default=16, help="processes for the motion_check gate (1 = in-process)"
    )
    ap.add_argument("--tier", default="industry", help="motion_check tier for the gate (academic / industry / strict)")
    ap.add_argument("--steps", type=int, default=200, help="Kimodo denoising steps per segment")
    ap.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    ap.add_argument("--shard-id", type=int, default=0, help="gen shard index; episode ids are shard_id*1000 + i")
    ap.add_argument("--max-resample", type=int, default=40, help="seed retries when a seed yields no valid approach")
    ap.add_argument("--cam-kind", default="fisheye", help="ego camera kind recorded in the episode spec")
    ap.add_argument(
        "--robot-variant",
        default="g1_29dof_thor",
        help="robot variant recorded in the episode spec (THOR helmet + OAK)",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="resume: skip episodes whose out/episode_<gid>/ already has "
        "episode_spec.json + g1_qpos.npy (pre-populate from S3 for "
        "preemption-safe resume)",
    )
    ap.add_argument("--vdr-level", choices=["light"], default="light", help="render-time sensor artifacts (light only)")
    ap.add_argument(
        "--asset-color", choices=["default"], default="default", help="chair appearance (native material only)"
    )
    ap.add_argument(
        "--chair-set",
        choices=["single", "pair"],
        default="pair",
        help="chair draw: pair = the paper data's per-episode 50/50 draw (default; only the arm chair ships, so "
        "both slots are arm_chair and the draw keeps the paper's camera DR); single = no draw, arm_chair.",
    )
    ap.add_argument(
        "--sit-nf", type=int, default=50, help="sit segment frame budget. Lower = less pre-descent standing."
    )
    ap.add_argument(
        "--seam-k",
        type=int,
        default=3,
        help="seam-blend half-width in frames across segment joins. Lower = "
        "shorter frozen band; the staged chain's raw joins are already "
        "under 0.3 cm.",
    )
    ap.add_argument(
        "--single-standing-kf",
        default="standpose",
        help="standing pose (motion_gen/kimodo/keyframes/single_kf_<name>.npz) used "
        "for both the walk's FullBody anchor and the turn-end keyframe kf[2]; "
        "'none' uses Kimodo's own standing pose for the walk and the "
        "keyframe's own pose. Two different sources differ by ~6 deg mean / "
        "~21 deg max per joint and give a ~6 deg pelvis-pitch step at the "
        "walk -> turn join. 'eval' = the closed-loop eval start pose.",
    )
    ap.add_argument(
        "--turn-constraint",
        default="plateau",
        choices=["fullbody", "heading", "hybrid", "plateau"],
        help="turn constraint set (motion_gen/kimodo/segments.py::gen_segment). "
        "plateau = dense root hold + sparse swept heading + a terminal "
        "heading plateau: turn-end yaw error 2.2 deg vs 8.6 deg for fullbody "
        "(the 2 endpoint keyframes).",
    )
    ap.add_argument(
        "--no-turn-pin-start",
        dest="turn_pin_start",
        action="store_false",
        help="do not pin the turn's frame 0 to the walk's actual last frame. "
        "Pinned (default): early-turn slide 3.6 cm vs 16.7 cm, turn-end yaw "
        "error 2.5 vs 8.6 deg.",
    )
    ap.add_argument(
        "--turn-nf",
        type=int,
        default=50,
        help="turn frame budget (50 = 1.67 s at 30 fps); 0 derives it from the yaw "
        "sweep at 2.5 deg/frame (45-97 frames)",
    )
    ap.add_argument(
        "--turn-end-back-cm",
        type=float,
        default=20.0,
        help="move the turn-end keyframe this far back toward the seat, folding a "
        "step back into the pivot so the sit covers less backward translation "
        "while lowering. Moving only this keyframe costs no placement yield "
        "(shrinking the stand gap to 0.35 m placed 0/9).",
    )
    ap.add_argument(
        "--no-tf32", dest="tf32", action="store_false", help="disable TF32 matmuls and cudnn.benchmark for Kimodo"
    )
    ap.add_argument(
        "--no-text-cache", dest="text_cache", action="store_false", help="do not cache text-encoder outputs per prompt"
    )
    ap.add_argument(
        "--no-rank-by-skate",
        dest="rank_by_skate",
        action="store_false",
        help="rank walk candidates by (hard failures, -score) only",
    )
    ap.add_argument("--out", default="/tmp/sit_syn", help="output dir; episodes are written to <out>/episode_<gid>/")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.environ["TURN_PIN_START"] = "1" if a.turn_pin_start else "0"
    global _RANK_BY_SKATE
    _RANK_BY_SKATE = a.rank_by_skate
    scenes = ["canonical"]
    scene_occ = {"canonical": canonical_scene()}
    seg_plan = PLAN_3
    # The paper data drew one of two chairs per episode; only the arm chair ships, so both slots of the pair render it.
    # Keeping the draw keeps every episode's camera DR (the next draws of that stream) equal to the paper's.
    chair_pool = (ARM_CHAIR_USDZ, ARM_CHAIR_USDZ) if a.chair_set == "pair" else ()
    cfg = SitBatchConfig(
        vdr_level=a.vdr_level,
        asset_color=a.asset_color,
        chair_asset_pool=chair_pool,
    )
    task = SitTask(cam_kind=a.cam_kind, robot_variant=a.robot_variant)
    SHARD_STRIDE = 1_000_000

    gen_ctx = _init_kimodo(a)

    n_done = 0
    for local_ep in range(a.n):
        gid = a.shard_id * 1000 + local_ep
        ep_dir = os.path.join(a.out, f"episode_{gid:04d}")
        if (
            a.skip_existing
            and os.path.isfile(os.path.join(ep_dir, "episode_spec.json"))
            and os.path.isfile(os.path.join(ep_dir, "g1_qpos.npy"))
        ):
            n_done += 1
            print(f"[sit] ep{gid}: resume-skip (already present)", flush=True)
            continue
        spec = None
        base = a.seed + a.shard_id * SHARD_STRIDE + local_ep * (a.max_resample + 1)
        scene_dir = scenes[gid % len(scenes)]
        for k in range(a.max_resample + 1):
            spec = task.build_scenario(scene_occ[scene_dir], base + k, cfg)
            if spec is not None:
                break
        if spec is None:
            print(
                f"[sit] ep{gid}: no valid placement in {os.path.basename(scene_dir)} "
                f"after {a.max_resample} tries; skipping",
                flush=True,
            )
            continue

        kf = build_sit_kfs_from_spec(spec)
        if a.single_standing_kf != "none":
            _q = np.load(single_kf_path(a.single_standing_kf))["qpos"].astype(np.float64)
            assert _q.shape == (36,), _q.shape
            kf[2] = _q
        dense_plan = np.column_stack([np.asarray(spec.path_xy), np.asarray(spec.path_yaw)]) if spec.path_xy else None

        qpos, bounds, meta = _generate_gated(gen_ctx, a, seg_plan, spec, kf, gid, scene_occ[scene_dir].scene_id)

        sit_lo, sit_hi = bounds.get("SIT", [-1, -1])
        spec.sit_start_frame, spec.sit_end_frame = int(sit_lo), int(sit_hi)

        meta["vdr_level"] = a.vdr_level
        meta["asset_color"] = a.asset_color
        meta["domain_randomization"] = True
        meta["prompt"] = spec.prompt
        meta["duration_s"] = spec.duration_s
        meta["lighting_variation"] = spec.lighting_variation
        meta["cam_artifacts"] = spec.cam_artifacts

        os.makedirs(ep_dir, exist_ok=True)
        np.save(os.path.join(ep_dir, "g1_qpos.npy"), qpos.astype(np.float32))
        if dense_plan is not None:
            np.save(os.path.join(ep_dir, "plan.npy"), dense_plan.astype(np.float32))
        json.dump(meta, open(os.path.join(ep_dir, "meta.json"), "w"), indent=2)
        spec.dump(os.path.join(ep_dir, "episode_spec.json"))
        print(
            f"[sit] ep{gid} DONE scene={meta['scene']} frames={len(qpos)} "
            f"accept={meta.get('final_gen_accepted')} -> {ep_dir}",
            flush=True,
        )
        n_done += 1

    print(f"[sit] SHARD {a.shard_id} DONE ({n_done}/{a.n} eps) -> {a.out}", flush=True)
    if gen_ctx is not None and gen_ctx.get("pool") is not None:
        gen_ctx["pool"].shutdown(wait=True)
    sys.stdout.flush()
    os._exit(0)


def _init_kimodo(a):
    import concurrent.futures as cf
    import multiprocessing as mp

    os.environ.setdefault("TEXT_ENCODER_MODE", "local")
    import torch

    if a.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    from motion_gen.kimodo.model import load_model_and_skel, skel_conv_singleton, CachedTextEncoder
    from motion_gen.kimodo.motion_gen import seam_blend as _seam_blend, PROMPTS
    from motion_gen.kimodo.segments import gen_segment
    from motion_gen.kimodo.walk import walk_qpos_from_traj, _g1_standing_keyframe
    from kimodo.exports.mujoco import MujocoQposConverter
    from motion_gen.motion_check.config import CheckConfig

    model, skel, converter, standing, dev = load_model_and_skel(dev=a.device)
    if a.text_cache:
        model.text_encoder = CachedTextEncoder(model.text_encoder)
    conv = MujocoQposConverter(skel_conv_singleton())
    if a.single_standing_kf != "none":
        _p = single_kf_path(a.single_standing_kf)
        standing_kf = _g1_standing_keyframe(skel, model=model, cache_path=_p)
        print(f"[sit] single standing kf {a.single_standing_kf} -> WALK anchor + kf[2] from {_p}", flush=True)
    else:
        standing_kf = _g1_standing_keyframe(skel, model=model)  # for walk_qpos_from_traj
    pool = None
    if a.check_workers > 1:
        pool = cf.ProcessPoolExecutor(max_workers=a.check_workers, mp_context=mp.get_context("spawn"))
    return dict(
        model=model,
        skel=skel,
        conv=conv,
        dev=dev,
        prompts=PROMPTS,
        gen_seg=gen_segment,
        walk_traj=walk_qpos_from_traj,
        standing_kf=standing_kf,
        seam=_seam_blend,
        cfg=CheckConfig.for_task("sit_on_chair", tier=a.tier),
        pool=pool,
    )


_RANK_BY_SKATE = True  # --no-rank-by-skate


def _rank_key(rep: dict):
    """Best-of-N ordering key: fewest hard failures, then least foot skate, then score."""
    blocking = len(rep.get("blocking", ()))
    score = -float(rep.get("score", 0.0))
    if not _RANK_BY_SKATE:
        return (blocking, score)
    fs = rep.get("checks", {}).get("foot_skate", {}).get("meta", {})
    skate = fs.get("frac_planted_skating")
    skate = float("inf") if skate is None else float(skate)
    return (blocking, skate, score)


_END_OF_CLIP_CHECKS = ("sits_at_end", "seat_contact")


def _prefix_blocking(rep: dict) -> list:
    """``rep``'s hard failures minus the ones only the full clip can satisfy."""
    return [b for b in rep.get("blocking", ()) if b not in _END_OF_CLIP_CHECKS]


def _prefix_key(rep: dict):
    """:func:`_rank_key` on a prefix verdict (end-of-clip checks dropped)."""
    return _rank_key({**rep, "blocking": _prefix_blocking(rep)})


def _generate_gated(ctx, a, seg_plan, spec, kf, gid, scene_id):
    """Staged walk → turn → sit best-of-N."""
    import time
    from motion_gen.check_worker import check_one
    from motion_gen.kimodo.keyframes.sit_keyframes import set_root_pose, _yaw_of
    from kimodo.tools import seed_everything

    model, skel, conv, dev = ctx["model"], ctx["skel"], ctx["conv"], ctx["dev"]

    plan = np.column_stack([np.asarray(spec.path_xy), np.asarray(spec.path_yaw)])
    approach_face = float(spec.path_yaw[-1])
    seat_xy = np.asarray(spec.seat_target_xyz[:2])
    seated_yaw = float(spec.seat_yaw_rad)
    base_seed = (a.seed * 100 + a.shard_id) * 100000 + gid * 100

    turn_mode = a.turn_constraint
    teb = a.turn_end_back_cm / 100.0
    back_dir = np.array([math.cos(seated_yaw), math.sin(seated_yaw)])

    def _check(idx, qp, phases):
        """One calibrated check. The pool exists for the WALK batch; a single"""
        return check_one((idx, qp, phases, a.tier))

    def _check_many(items):
        return list(ctx["pool"].map(check_one, items)) if ctx["pool"] is not None else [check_one(t) for t in items]

    rec = dict(walk_rounds=0, turn_tries=0, sit_tries=0, walk_batch=int(a.batch))
    n_cand = n_pass = 0
    chosen = None
    soft = None  # best (key, ...) seen across all rounds
    for rnd in range(a.n_max_retry + 1):
        rec["walk_rounds"] = rnd + 1
        t0 = time.time()
        seed_everything(base_seed + rnd * 1_000_000)
        walks = ctx["walk_traj"](
            plan[:, 0],
            plan[:, 1],
            plan[:, 2],
            float(spec.robot.yaw_rad),
            model,
            skel,
            ctx["standing_kf"],
            end_yaw=approach_face,
            num_denoising_steps=a.steps,
            device=dev,
            num_samples=int(a.batch),
        )
        if walks.ndim == 2:  # num_samples==1 → [T,36]
            walks = walks[None]
        wreps = _check_many([(i, walks[i], {"WALK": [0, walks.shape[1]]}, a.tier) for i in range(len(walks))])
        wi = min(range(len(wreps)), key=lambda i: _prefix_key(wreps[i]))
        walk = np.asarray(walks[wi])
        rec["walk_gen_s"] = round(time.time() - t0, 2)
        rec["walk_pick_idx"] = int(wi)
        rec["walk_pick_score"] = float(wreps[wi]["score"])
        rec["walk_pick_blocking"] = _prefix_blocking(wreps[wi])
        rec["walk_n_pass"] = int(sum(1 for r in wreps if not _prefix_blocking(r)))

        def _skate(r):
            m = r.get("checks", {}).get("foot_skate", {}).get("meta", {})
            return m.get("frac_planted_skating")

        rec["walk_batch_skate"] = [_skate(r) for r in wreps]
        rec["walk_pick_skate"] = _skate(wreps[wi])
        walk_end = walk[-1].copy()

        turn_end = set_root_pose(kf[2], walk_end[:2] - teb * back_dir, seated_yaw)
        turn_nf = (
            a.turn_nf
            if a.turn_nf > 0
            else max(int(abs(_yaw_of(turn_end[3:7]) - _yaw_of(walk_end[3:7])) / math.radians(2.5)), 30)
        )
        best_t = None
        t0 = time.time()
        for r in range(a.stage_retry + 1):
            rec["turn_tries"] += 1
            s = base_seed + rnd * 1_000_000 + 500 + r
            seed_everything(s)
            turn = ctx["gen_seg"](
                model,
                skel,
                conv,
                ctx["prompts"]["TURN"],
                turn_nf,
                walk_end,
                turn_end,
                s,
                dev,
                a.steps,
                turn_mode=turn_mode,
            )
            cat = np.concatenate([walk, turn], axis=0)
            rp = _check(0, cat, {"WALK": [0, len(walk)], "TURN": [len(walk), len(cat)]})
            tb = _prefix_blocking(rp)
            n_cand += 1
            n_pass += int(not tb)
            key = _prefix_key(rp)
            if best_t is None or key < best_t[0]:
                best_t = (key, turn, rp, tb)
            if not tb:
                break
        rec["turn_gen_s"] = round(time.time() - t0, 2)
        turn, tblock = best_t[1], best_t[3]
        rec["turn_passed"] = not tblock
        rec["turn_blocking"] = tblock
        if tblock and rnd < a.n_max_retry:
            print(f"[sit] ep{gid}: TURN failed {tblock} -> resampling WALK", flush=True)
            continue

        sit_start = np.asarray(turn[-1], dtype=np.float64).copy()
        seated_kf = set_root_pose(kf[4], seat_xy, seated_yaw)
        best_s = None
        t0 = time.time()
        for r in range(a.stage_retry + 1):
            rec["sit_tries"] += 1
            s = base_seed + rnd * 1_000_000 + 900 + r
            seed_everything(s)
            sit = ctx["gen_seg"](
                model,
                skel,
                conv,
                ctx["prompts"]["SIT"],
                int(a.sit_nf),
                sit_start,
                seated_kf,
                s,
                dev,
                a.steps,
                turn_mode="fullbody",
            )
            cat = np.concatenate([walk, turn, sit], axis=0)
            ph = {
                "WALK": [0, len(walk)],
                "TURN": [len(walk), len(walk) + len(turn)],
                "SIT": [len(walk) + len(turn), len(cat)],
            }
            rp = _check(0, cat, ph)
            n_cand += 1
            n_pass += int(rp["passed"])
            key = _rank_key(rp)
            if best_s is None or key < best_s[0]:
                best_s = (key, sit, rp, cat, ph)
            if rp["passed"]:
                break
        rec["sit_gen_s"] = round(time.time() - t0, 2)
        skey, sit, srep, cat, ph = best_s
        rec["sit_passed"] = bool(srep["passed"])
        rec["sit_blocking"] = list(srep["blocking"])
        if soft is None or skey < soft[0]:
            soft = (skey, walk, turn, sit, cat, ph, srep, dict(rec))
        if srep["passed"]:
            chosen = (skey, walk, turn, sit, cat, ph, srep, dict(rec))
            break
        if rnd < a.n_max_retry:
            print(f"[sit] ep{gid}: SIT failed {srep['blocking']} -> resampling WALK", flush=True)
    if chosen is None:
        chosen = soft
    _, walk, turn, sit, cat, ph, kept, rec = chosen

    seams = [int(len(walk)), int(len(walk) + len(turn))]
    qpos = ctx["seam"](cat.copy(), seams, k=int(a.seam_k)) if a.seam_k > 0 else cat.copy()
    bounds = {k: [int(v[0]), int(v[1])] for k, v in ph.items()}
    meta = {
        "task_type": "sit_on_chair",
        "phases": bounds,
        "seams": seams,
        "recipe": "sit_syn_staged_walk_turn_sit",
        "tier": a.tier,
        "segments": 3,
        "global_ep": gid,
        "scene": scene_id,
        "batch": a.batch,
        "n_max_retry": a.n_max_retry,
        "rounds_used": rec["walk_rounds"],
        "sit_nf": int(a.sit_nf),
        "seam_k": int(a.seam_k),
        "turn_nf": int(len(turn)),
        "turn_constraint": turn_mode,
        "turn_end_back_cm": float(a.turn_end_back_cm),
        "standing_kf": None if a.single_standing_kf == "none" else a.single_standing_kf,
        "n_candidates": n_cand,
        "n_passed": int(n_pass),
        "final_gen_accepted": bool(kept["passed"]),
        "kept_score": kept["score"],
        "kept_blocking": kept["blocking"],
        "final_checks": kept["checks"],
        "total_frames": int(len(qpos)),
        "staged": rec,
    }
    return qpos, bounds, meta


if __name__ == "__main__":
    main()
