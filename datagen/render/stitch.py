#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""S3-side dataset stitch for the direct-write render layout — zero bulk download.

Each render shard writes its parquets + mp4s with global episode indices straight into the
dataset prefix and its meta into `meta/shard_NNN/{episodes.jsonl,tasks.jsonl,info.json}`.
The stitch concatenates the per-shard meta into the top-level `meta/` (KB of data, seconds),
then checks the meta episode set equals the data/ + videos/ file sets (listings only).

Dataset layout:
    s3://bucket/<dataset>/
        data/episode_{global:06d}.parquet
        videos/observation.images.fpv_cam/episode_{global:06d}.mp4
        videos/observation.images.tpv_cam/episode_{global:06d}.mp4
        meta/{info.json, episodes.jsonl, tasks.jsonl}, meta/shard_NNN/

Usage:
    python -m render.stitch --mode stitch --source-prefix $HALO_S3_ROOT/halo_sit_n50k
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

# Resolve render.* / common.* without datagen/ on PYTHONPATH (the script is also run as a
# plain `python render/stitch.py` without PYTHONPATH set).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.append(_REPO)
from render.dataset_writer import build_g1_features_block as _g1_features_block  # noqa: E402
from common.storage import s3  # noqa: E402


def _ls(prefix: str, recursive: bool = False) -> subprocess.CompletedProcess:
    """`s3 ls [--recursive] prefix` (common/storage.py) as a CompletedProcess."""
    try:
        lines = s3().ls_lines(prefix, recursive=recursive)
        return subprocess.CompletedProcess(["ls", prefix], 0, "\n".join(lines) + "\n", "")
    except subprocess.CalledProcessError as e:
        return subprocess.CompletedProcess(["ls", prefix], e.returncode or 1, "", e.stderr or "")


def s3_episode_index_set(prefix: str, ext: str) -> set[int] | None:
    """Global episode-index set from `episode_{NNNNNN}.<ext>` files under `prefix`
    (recursive), None if the listing failed. Used by the stitch integrity check — S3 ls
    only, no download."""
    import re
    r = _ls(prefix.rstrip("/") + "/", recursive=True)
    if r.returncode != 0:
        print(f"[ls FAIL] {prefix}\n{r.stderr}", file=sys.stderr)
        return None
    pat = re.compile(rf"episode_(\d+)\.{re.escape(ext)}\b")
    return {int(m.group(1)) for m in pat.finditer(r.stdout)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["stitch"], default="stitch",
                    help="stitch (the only mode): concatenate the per-shard meta jsonls.")
    ap.add_argument("--source-prefix", required=True,
                    help="Dataset prefix containing meta/shard_NNN/ subdirs.")
    ap.add_argument("--dest",
                    help="Optional, defaults to --source-prefix (in-place).")
    ap.add_argument("--keep-local", action="store_true",
                    help="Don't delete the local meta staging dir on success.")
    ap.add_argument("--workdir", default=None,
                    help="Local staging dir. Defaults to a tempdir.")
    ap.add_argument("--expected-shards", type=int, default=None,
                    help="Fail if the number of per-shard meta dirs found "
                         "!= this (catches a stitch run before late/preempted shards uploaded).")
    ap.add_argument("--no-verify", action="store_true",
                    help="Skip the post-stitch integrity check that the meta "
                         "episode set equals the data/ + videos/ file sets on S3.")
    _run_stitch(ap.parse_args())


def _run_stitch(args):
    """Concatenate per-shard meta jsonls into top-level meta/ — for direct-write layout."""
    dest = (args.dest or args.source_prefix).rstrip("/")
    work = args.workdir or tempfile.mkdtemp(prefix="dataset_stitch_")
    os.makedirs(work, exist_ok=True)

    print(f"=== dataset stitch (direct-write) ===")
    print(f"  dest: {dest}")
    print(f"  workdir: {work}")

    # 1. Pull the per-shard meta tree (small, but ~3 files x hundreds of shards on one
    #    prefix). S3 throttles rapid GET bursts on one prefix (AWS_ERROR_S3_SLOW_DOWN), and a
    #    failed `cp --recursive` starts over on retry. `sync` (skips files already pulled)
    #    in a backoff retry loop fetches only what is missing on each pass, so the pull
    #    completes while the prefix is rate-limited.
    print(f"--- pulling per-shard meta ---")
    meta_local = os.path.join(work, "meta")
    os.makedirs(meta_local, exist_ok=True)
    err = None
    for _attempt in range(8):
        try:
            s3().sync(f"{dest}/meta/", meta_local + "/")
            err = None
            break
        except subprocess.CalledProcessError as e:
            err = e
        print(f"  [meta pull] attempt {_attempt + 1}/8 incomplete (throttle?); "
              f"retrying after backoff", flush=True)
        time.sleep(15 * (_attempt + 1))
    if err is not None:
        raise SystemExit(f"[fatal] failed to pull {dest}/meta/ after retries: "
                         f"{(err.stderr or '')[-400:]}")

    meta_root = os.path.join(work, "meta")
    shard_dirs = sorted(d for d in os.listdir(meta_root)
                        if d.startswith("shard_") and os.path.isdir(os.path.join(meta_root, d)))
    print(f"  found {len(shard_dirs)} per-shard meta dirs")
    if not shard_dirs:
        raise SystemExit(f"[fatal] no shard_*/ dirs under {dest}/meta/")
    # Refuse to stitch before late or preempted shards have uploaded their meta (otherwise
    # their episodes are missing from the stitched dataset).
    if args.expected_shards is not None and len(shard_dirs) != args.expected_shards:
        raise SystemExit(
            f"[fatal] found {len(shard_dirs)} per-shard meta dirs but --expected-shards="
            f"{args.expected_shards}. Some shards have not uploaded meta yet (preempted/late)? "
            f"Re-run once all {args.expected_shards} are present.")

    # 2. Concatenate.
    all_episodes = []
    info_template = None
    all_tasks = set()
    for sd in shard_dirs:
        ep_path = os.path.join(meta_root, sd, "episodes.jsonl")
        if os.path.exists(ep_path):
            with open(ep_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_episodes.append(json.loads(line))
        if info_template is None:
            info_path = os.path.join(meta_root, sd, "info.json")
            if os.path.exists(info_path):
                with open(info_path) as f:
                    info_template = json.load(f)
        tasks_path = os.path.join(meta_root, sd, "tasks.jsonl")
        if os.path.exists(tasks_path):
            with open(tasks_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_tasks.add(json.loads(line)["task"])

    all_episodes.sort(key=lambda e: e["episode_index"])
    total_frames = sum(e.get("length", 0) for e in all_episodes)
    gaps = 0
    if all_episodes:
        gaps = (all_episodes[-1]["episode_index"] - all_episodes[0]["episode_index"] + 1) - len(all_episodes)
    print(f"  stitched {len(all_episodes)} episodes, {total_frames} frames, "
          f"{len(all_tasks)} unique tasks, {gaps} index gaps")

    # 3. Write top-level meta jsonls.
    out_dir = os.path.join(work, "out_meta")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "episodes.jsonl"), "w") as f:
        for ep in all_episodes:
            f.write(json.dumps(ep) + "\n")
    sorted_tasks = sorted(all_tasks) or ["walk_and_sit"]
    with open(os.path.join(out_dir, "tasks.jsonl"), "w") as f:
        for i, t in enumerate(sorted_tasks):
            f.write(json.dumps({"task_index": i, "task": t}) + "\n")
    info = info_template or {}
    info["total_episodes"] = len(all_episodes)
    info["total_frames"] = total_frames
    info["total_tasks"] = len(sorted_tasks)
    info["total_chunks"] = 1
    info["chunks_size"] = len(all_episodes)
    info["splits"] = {"train": f"0:{len(all_episodes)}"}
    # Backfill the LeRobot-style `features` block if the per-shard info.json has none
    # (datasets written without dataset_writer.build_g1_features_block). eval_openloop reads
    # features.action.names and raise KeyError without it.
    if "features" not in info:
        info["features"] = _g1_features_block()
    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    # 4. Upload top-level meta. Per-shard meta subdirs stay in place for provenance.
    print(f"--- uploading stitched meta to {dest}/meta/ ---")
    for fname in ("episodes.jsonl", "tasks.jsonl", "info.json"):
        s3().cp(os.path.join(out_dir, fname), f"{dest}/meta/{fname}")
    print(f"=== done: {dest} ===")
    print(f"  {len(all_episodes)} episodes (gaps={gaps}), {total_frames} frames")

    # 5. Integrity check: the stitched meta episode set must equal the data/ + videos/
    # file sets. Catches a partial stitch (missing shards) or data / meta / video naming
    # mismatches; listings only, no download.
    if not args.no_verify:
        print("--- integrity check (meta vs data vs videos) ---")
        meta_set = {e["episode_index"] for e in all_episodes}
        data_set = s3_episode_index_set(f"{dest}/data", "parquet")
        fpv_set = s3_episode_index_set(
            f"{dest}/videos/observation.images.fpv_cam", "mp4")
        tpv_set = s3_episode_index_set(
            f"{dest}/videos/observation.images.tpv_cam", "mp4")
        listings = (("data", data_set), ("fpv", fpv_set), ("tpv", tpv_set))
        print(f"  meta={len(meta_set)} " + " ".join(
            f"{k}={'ls failed' if v is None else len(v)}" for k, v in listings))
        problems = []
        # an empty or failed listing is a failure, not a skipped comparison
        for name, got in listings:
            if meta_set and not got:
                problems.append(f"{name}: " + ("listing failed (or empty prefix)"
                                               if got is None else "no files"))
        if data_set and meta_set != data_set:
            problems.append(f"meta vs data: {len(data_set - meta_set)} in data-not-meta, "
                            f"{len(meta_set - data_set)} in meta-not-data "
                            f"(e.g. {sorted(data_set ^ meta_set)[:5]})")
        if fpv_set and data_set and fpv_set != data_set:
            problems.append(f"fpv vs data: {len(data_set ^ fpv_set)} symmetric-diff")
        if tpv_set and data_set and tpv_set != data_set:
            problems.append(f"tpv vs data: {len(data_set ^ tpv_set)} symmetric-diff")
        if problems:
            for p in problems:
                print(f"  [FAIL] {p}", file=sys.stderr)
            raise SystemExit(
                "[fatal] integrity check FAILED — meta/data/video episode sets disagree. "
                "Do not train on the stitched meta until this is resolved.")
        print(f"  [PASS] meta == data == fpv == tpv ({len(meta_set)} episodes)")
    if not args.keep_local and not args.workdir:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
