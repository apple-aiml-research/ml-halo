#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""MuJoCo render loop for ``render_episodes.py --backend mujoco``."""

from __future__ import annotations

import json
import os
import time

import numpy as np

from render.dataset_writer import FPS, write_dataset_episode, write_dataset_shard_meta
from render.cameras import load_oak_intrinsics
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


def apply_cam_artifacts_to_mp4(mp4_path: str, art: dict, seed: int, fps: int, flow) -> None:
    """Read an mp4, apply the seeded cam_fx to every frame, rewrite it in place."""
    if not art or not os.path.isfile(mp4_path):
        return
    try:
        import imageio.v2 as imageio

        frames = [np.asarray(fr)[..., :3] for fr in imageio.mimread(mp4_path, memtest=False)]
        if frames:
            out = apply_cam_artifacts_to_frames(frames, art, seed, flow, workers=1)
            imageio.mimsave(mp4_path, out, fps=fps, macro_block_size=1)
    except Exception as e:  # FX must never break a render
        print(f"  [cam_fx] WARN {os.path.basename(mp4_path)}: {e}", flush=True)


def _add_backing_floor(xml_str, floor_z):
    """Render-only backing floor 2 cm under the room floor and a flat grey skybox (no void through mesh gaps)."""
    sky = (
        '    <texture name="_bg_sky" type="skybox" builtin="flat" '
        'rgb1="0.55 0.55 0.58" rgb2="0.55 0.55 0.58" width="8" height="8"/>\n'
        '    <material name="_bg_floor_mat" rgba="0.45 0.42 0.38 1"/>\n'
    )
    flr = (
        f'    <geom name="_bg_floor" type="plane" size="60 60 0.1" pos="0 0 {float(floor_z) - 0.02:.4f}" '
        f'material="_bg_floor_mat" contype="0" conaffinity="0"/>\n'
    )
    xml_str = xml_str.replace("</asset>", sky + "  </asset>", 1)
    return xml_str.replace("</worldbody>", flr + "  </worldbody>", 1)


def run_mujoco(args, spec_paths, episodes_root, dataset_dir):
    """Render every ``(spec_path, npy_path)`` episode of this shard with MuJoCo."""
    from render.mujoco.episode_objects import add_objects_to_xml, set_obj_cache_dir
    from render.mujoco.replay import init_worker, render_episode
    from render.synthetic_scene.build_infinigen_mjcf import build_scene_with_robot
    from motion_gen.episode_spec import EpisodeSpec

    scene_dir = os.path.expanduser(args.scene_dir)
    scene_stem = os.path.basename(scene_dir.rstrip("/"))
    workdir = os.path.join(os.path.expanduser(args.output_dir), "_mjscene")
    pr = load_oak_intrinsics()["pinhole_render"]
    oak_w, oak_h = pr["width"], pr["height"]
    xml_path = build_scene_with_robot(
        scene_dir, workdir, offwidth=max(1600, args.width, oak_w), offheight=max(1000, args.height, oak_h)
    )
    base_xml = open(xml_path).read()
    scene = load_scene(scene_dir)
    set_obj_cache_dir(workdir)

    finished = []
    for ei, (spec_path, npy_path) in enumerate(spec_paths):
        ep_name = os.path.basename(os.path.dirname(spec_path))
        ep_dir = os.path.join(episodes_root, ep_name)
        spec = EpisodeSpec.load(spec_path)
        qpos = np.load(npy_path).astype(np.float64)
        fx_seed = seed_for_episode(args.render_batch, args.shard_idx, ep_name)
        if episode_is_complete(ep_dir):
            print(f"  [{ei + 1}/{len(spec_paths)}] {ep_name} — resume skip")
        else:
            os.makedirs(ep_dir, exist_ok=True)
            spec, qpos, tpv_eye, tpv_target = place_episode(args, spec, qpos, ep_name, scene)
            xml_str = _add_backing_floor(add_objects_to_xml(base_xml, spec), scene["floor_z"])
            ep_xml = os.path.join(workdir, f"episode_{ei:04d}_scene.xml")
            with open(ep_xml, "w") as f:
                f.write(xml_str)
            init_worker(ep_xml, args.width, args.height, oak_w, oak_h)
            t0 = time.time()
            n, timing = render_episode(qpos, ep_dir, FPS, tpv_eye, tpv_target)
            art = dict(spec.cam_artifacts or {})
            apply_cam_artifacts_to_mp4(
                os.path.join(ep_dir, "fpv.mp4"), art, fx_seed, args.target_fps, episode_motion_flow(art, qpos)
            )
            print(
                f"  [{ei + 1}/{len(spec_paths)}] {ep_name} ({n} f) render={timing['render_s']:.1f}s "
                f"encode={timing['encode_wait_s']:.1f}s wall={time.time() - t0:.1f}s"
            )
            meta = episode_meta(spec, qpos, ep_name, scene_stem, source="data_gen_mujoco_go_to")
            meta["timing"] = timing
            with open(os.path.join(ep_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

        with open(os.path.join(ep_dir, "meta.json")) as f:
            meta = json.load(f)
        art = dict(spec.cam_artifacts or {})
        ep_idx = dataset_episode_index(ep_name)
        n_actual = write_dataset_episode(ep_dir, dataset_dir, ep_idx, meta, fps=args.target_fps)
        # OAK is the FPV: the warped fisheye in the dataset fpv_cam slot gets the cam_fx.
        oak_final = os.path.join(dataset_dir, "videos", "observation.images.fpv_cam", f"episode_{ep_idx:06d}.mp4")
        apply_cam_artifacts_to_mp4(oak_final, art, fx_seed, args.target_fps, episode_motion_flow(art, qpos))
        finished.append({"ep_idx": ep_idx, "ep_name": ep_name, "n_frames": n_actual, "meta": meta})

    write_dataset_shard_meta(dataset_dir, finished, shard_subpath=f"shard_{args.shard_idx:03d}", fps=args.target_fps)
    print(f"[shard {args.shard_idx}] done — {len(finished)} episodes → {dataset_dir}")
