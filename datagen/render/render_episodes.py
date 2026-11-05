#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Render one scene's episodes (THOR G1, OAK fisheye as FPV + fixed TPV) and write them as dataset episodes."""

from __future__ import annotations

import argparse
import glob
import os
import sys

# Run as a script, sys.path[0] is render/, whose mujoco/ would shadow the mujoco package: use datagen/ instead.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != _HERE]
sys.path.insert(0, os.path.dirname(_HERE))

from render.driver_common import resolve_episode_specs  # noqa: E402


def main():
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True, choices=["isaac", "mujoco"], help="Rendering backend.")
    ap.add_argument("--npy-dir", required=True, help="Dir of episode_*/ spec dirs (episode_spec.json + g1_qpos.npy).")
    ap.add_argument(
        "--scene-dir", required=True, help="Infinigen scene dir (asset/scan.usdc|obj + scene_analysis.json)."
    )
    ap.add_argument("--scene-key", default="", help="The scene's key under HALO_S3_ROOT, recorded as scene_s3_path.")
    ap.add_argument(
        "--output-dir", required=True, help="Output root: per-episode renders in episodes/, the dataset in dataset/."
    )
    ap.add_argument("--shard-idx", type=int, default=0, help="Render shard index (part of the per-episode seeds).")
    ap.add_argument("--render-batch", default="local", help="Batch tag (part of the per-episode seeds).")
    ap.add_argument("--max-trajs", type=int, default=0, help="Cap episodes in this run (0 = all).")
    ap.add_argument("--target-fps", type=int, default=30, help="mp4 frame rate and parquet timestamps.")
    ap.set_defaults(headless=True, enable_cameras=True, width=848, height=480)  # AppLauncher; FPV/TPV size
    args = ap.parse_args()

    # Read by the dataset writer: Isaac renders the OAK as a native fisheye, MuJoCo as a pinhole that the writer warps.
    os.environ["OAK_NATIVE_FISHEYE"] = "1" if args.backend == "isaac" else "0"

    out = os.path.expanduser(args.output_dir)
    episodes_root, dataset_dir = (os.path.join(out, "episodes"), os.path.join(out, "dataset"))
    os.makedirs(episodes_root, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    npy_paths = sorted(glob.glob(os.path.join(os.path.expanduser(args.npy_dir), "episode_*", "g1_qpos.npy")))
    if args.max_trajs > 0:
        npy_paths = npy_paths[: args.max_trajs]
    print(f"[shard {args.shard_idx}] {len(npy_paths)} sit episodes (backend={args.backend})")
    spec_paths = resolve_episode_specs(npy_paths)
    if not spec_paths:
        raise SystemExit(
            f"[render] FATAL: shard {args.shard_idx} has 0 episodes (episode_spec.json + g1_qpos.npy) under {args.npy_dir}"
        )
    if args.backend == "isaac":
        from render.isaac.driver import run_isaac

        run_isaac(args, spec_paths, episodes_root, dataset_dir)
    else:
        from render.mujoco.driver import run_mujoco

        run_mujoco(args, spec_paths, episodes_root, dataset_dir)


if __name__ == "__main__":
    main()
