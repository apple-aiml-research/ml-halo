#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Isaac Sim render loop for ``render_episodes.py --backend isaac``."""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from render.dataset_writer import write_dataset_episode, write_dataset_shard_meta
from render.driver_common import (
    apply_cam_artifacts_to_frames,
    episode_is_complete,
    episode_meta,
    episode_motion_flow,
    dataset_episode_index,
    load_scene,
    place_episode,
    seed_for_episode,
)

CAMFX_WORKERS = 12


def _finalize(args, dataset_dir, scene_stem, frames, ep_dir, spec, qpos, ep_name, t_render, ep_idx):
    """cam_fx on the OAK frames (TPV stays clean), mp4 encode, meta.json, dataset write."""
    import imageio.v2 as imageio

    art = dict(spec.cam_artifacts or {})
    if art and frames["oak_rgb"]:
        seed = seed_for_episode(args.render_batch, args.shard_idx, ep_name)
        flow = episode_motion_flow(art, qpos)
        frames["oak_rgb"] = apply_cam_artifacts_to_frames(frames["oak_rgb"], art, seed, flow, CAMFX_WORKERS)
    t1 = time.time()
    for key, name in (("tpv_rgb", "tpv.mp4"), ("oak_rgb", "oak.mp4")):
        if frames[key]:
            imageio.mimsave(os.path.join(ep_dir, name), frames[key], fps=args.target_fps, macro_block_size=1)
    meta = episode_meta(spec, qpos, ep_name, scene_stem, source="data_gen_isaac_go_to")
    meta["timing"] = {"render_s": t_render, "encode_wait_s": time.time() - t1, "mode": "imageio"}
    with open(os.path.join(ep_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    n_actual = write_dataset_episode(ep_dir, dataset_dir, ep_idx, meta, fps=args.target_fps)
    return {"ep_idx": ep_idx, "ep_name": ep_name, "n_frames": n_actual, "meta": meta}


def run_isaac(args, spec_paths, episodes_root, dataset_dir):
    """Render every ``(spec_path, npy_path)`` episode of this shard with Isaac."""
    from isaaclab.app import AppLauncher

    scene_dir = os.path.expanduser(args.scene_dir)
    scene_stem = os.path.basename(scene_dir.rstrip("/"))
    _app = AppLauncher(args).app  # starts Kit; must precede the isaaclab.sim imports

    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext

    from render.isaac.episode_objects import (
        clear_episode_prims,
        place_episode_objects,
        prespawn_prop_pool,
        set_episode_lighting,
    )
    from render.isaac.replay import run_simulator
    from render.isaac.scene_builder import design_scene, set_robot_colors
    from render.synthetic_scene.build_infinigen_usd import build_infinigen_scene
    from motion_gen.episode_spec import EpisodeSpec

    sim = SimulationContext(sim_utils.SimulationCfg(device="cuda:0"))
    entities = design_scene()
    build_infinigen_scene(sim_utils, scene_dir)
    scene = load_scene(scene_dir)
    specs = [EpisodeSpec.load(s) for s, _ in spec_paths]
    prop_pool = prespawn_prop_pool(sim_utils, sorted({p.asset_path for s in specs for p in s.placements}))
    sim.reset()
    set_robot_colors()

    overlap_pool = ThreadPoolExecutor(max_workers=1)  # finalize episode i while rendering i+1
    pending = []
    for ei, ((spec_path, npy_path), spec) in enumerate(zip(spec_paths, specs)):
        ep_name = os.path.basename(os.path.dirname(spec_path))
        ep_dir = os.path.join(episodes_root, ep_name)
        ep_idx = dataset_episode_index(ep_name)
        if episode_is_complete(ep_dir):
            print(f"  [{ei + 1}/{len(spec_paths)}] {ep_name} — resume skip")
            with open(os.path.join(ep_dir, "meta.json")) as f:
                meta = json.load(f)
            n_actual = write_dataset_episode(ep_dir, dataset_dir, ep_idx, meta, fps=args.target_fps)
            pending.append({"ep_idx": ep_idx, "ep_name": ep_name, "n_frames": n_actual, "meta": meta})
            clear_episode_prims()
            continue

        os.makedirs(ep_dir, exist_ok=True)
        qpos = np.load(npy_path).astype(np.float64)
        spec, qpos, tpv_eye, tpv_target = place_episode(args, spec, qpos, ep_name, scene)
        place_episode_objects(spec, prop_pool)
        set_episode_lighting(sim_utils, spec)

        t0 = time.time()
        frames: dict[str, list] = {"tpv_rgb": [], "oak_rgb": []}
        sinks = {k: (lambda fid, rgb, k=k: frames[k].append(rgb.copy())) for k in frames}
        run_simulator(
            sim, entities, qpos, np.asarray(spec.target_xyz), ep_dir, tpv_eye, tpv_target, sinks, fps=args.target_fps
        )
        t_render = time.time() - t0
        print(f"  [{ei + 1}/{len(spec_paths)}] {ep_name} ({qpos.shape[0]} f) render={t_render:.1f}s")
        pending.append(
            overlap_pool.submit(
                _finalize, args, dataset_dir, scene_stem, frames, ep_dir, spec, qpos, ep_name, t_render, ep_idx
            )
        )
        clear_episode_prims()

    finished = [p.result() if hasattr(p, "result") else p for p in pending]
    overlap_pool.shutdown(wait=True)
    write_dataset_shard_meta(dataset_dir, finished, shard_subpath=f"shard_{args.shard_idx:03d}", fps=args.target_fps)
    print(f"[shard {args.shard_idx}] done — {len(finished)} episodes → {dataset_dir}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
