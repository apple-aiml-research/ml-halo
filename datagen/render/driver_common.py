#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Backend-independent per-episode helpers for the render drivers."""

from __future__ import annotations

import hashlib
import os
import re

import numpy as np

REQUIRED_FILES = ("meta.json", "traj.npz", "tpv.mp4", "oak.mp4")  # written by both backends


def seed_for_episode(batch_id: str, shard_idx: int, ep_name: str) -> int:
    """Deterministic 31-bit seed (placement, cam_fx) from (render batch, shard, episode name)."""
    h = hashlib.sha256(f"{batch_id}::{shard_idx:03d}::{ep_name}".encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def episode_is_complete(ep_dir: str, min_bytes: int = 256) -> bool:
    """True iff every file in REQUIRED_FILES exists in ``ep_dir`` with at least ``min_bytes``."""
    for f in REQUIRED_FILES:
        try:
            if os.path.getsize(os.path.join(ep_dir, f)) < min_bytes:
                return False
        except OSError:
            return False
    return True


def resolve_episode_specs(npy_paths):
    """Pair each ``episode_*/g1_qpos.npy`` with the ``episode_spec.json`` next to it."""
    out = []
    for npy in npy_paths:
        spec = os.path.join(os.path.dirname(npy), "episode_spec.json")
        if not os.path.isfile(spec):
            print(f"  [render] WARN missing episode_spec.json next to {npy}; skipping", flush=True)
            continue
        out.append((spec, npy))
    return out


def record_infinigen_scene(spec, scene_dir, floor_z, scene_key=""):
    """Write the room an episode was placed into onto its (rendered) spec."""
    scene_dir = os.path.expanduser(scene_dir).rstrip("/")
    spec.floor_z = float(floor_z)
    spec.scene_dir = ""  # host-local staging dir; the room is scene_s3_path under the data root
    spec.scene_id = os.path.basename(scene_dir)
    spec.scene_s3_path = scene_key or ""
    spec.scene_source = "infinigen"
    spec.robot.z = float(floor_z)
    for p in spec.placements:  # repo-relative chair path (render/assets.py resolves by file name)
        p.asset_path = f"assets/objects/chairs/{os.path.basename(p.asset_path)}"
    return spec


def place_episode(args, spec, qpos, ep_name, scene):
    """se2_fit-place one episode into the scene and pick its TPV; returns ``(spec, qpos, tpv_eye, tpv_target)``."""
    from render.synthetic_scene import robot_placement as rp
    from render.tpv import select_infinigen_tpv

    analysis, floor_z = scene["analysis"], scene["floor_z"]
    seed = seed_for_episode(args.render_batch, args.shard_idx, ep_name)
    spec, qpos, fallback = rp.se2_fit_place(spec, qpos, analysis, scene["clearance"], scene["cell"], floor_z, seed)
    if fallback:
        print(f"    [sit/se2_fit] {ep_name}: no path-free placement; falling back to static scene spawn")
    tpv_eye, tpv_target = select_infinigen_tpv(analysis, qpos[:, :2], floor_z)
    record_infinigen_scene(spec, args.scene_dir, floor_z, args.scene_key)
    return spec, qpos, tpv_eye, tpv_target


def load_scene(scene_dir):
    """scene_analysis.json of ``scene_dir`` plus its floor_z and render-time clearance grid."""
    import json

    from render.synthetic_scene.robot_placement import build_scene_clearance

    analysis = json.load(open(os.path.join(scene_dir, "scene_analysis.json")))
    occ, clearance = build_scene_clearance(analysis)
    cell = float(analysis.get("occupancy", {}).get("cell", 0.10))
    print(f"[infinigen] clearance grid {clearance.shape} n_free={int((~occ).sum())} cell={cell}")
    return {"analysis": analysis, "floor_z": float(analysis["floor_z"]), "clearance": clearance, "cell": cell}


def episode_meta(spec, qpos, ep_name, scene_stem, source):
    """Per-episode meta dict carried into the dataset writer."""
    return {
        "scene": scene_stem,
        "seq_name": ep_name,
        "valid_frames": int(qpos.shape[0]),
        "episode_name": ep_name,
        "source": source,
        "episode_spec": spec.to_dict(),
    }


def dataset_episode_index(ep_name):
    """Dataset-wide episode index: the trailing integer of the spec dir name (``episode_<gid>``)."""
    m = re.search(r"(\d+)$", ep_name)
    if m is None:
        raise RuntimeError(f"no trailing episode index in ep_name={ep_name!r}")
    return int(m.group(1))


def episode_motion_flow(art: dict, qpos) -> list:
    """Per-frame egomotion (mag_px, angle_rad) driving cam_fx motion blur; ``[]`` without motion blur."""
    if art and float(art.get("motion_blur_gain", 0.0)) > 0:
        from render.cam_fx import camera_yaw_pitch_from_qpos, egomotion_image_flow

        return egomotion_image_flow(*camera_yaw_pitch_from_qpos(qpos))
    return []


def apply_cam_artifacts_to_frames(frames, art, seed, flow, workers):
    """Apply seeded cam_fx to one camera's frame list; with ``workers > 1`` the draws are serial, the apply parallel."""
    from render.cam_fx import apply_cam_artifacts, draw_cam_artifact_randoms

    mot = [(flow[i] if i < len(flow) else None) for i in range(len(frames))]
    rng = np.random.default_rng(seed)
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        H, W = frames[0].shape[:2]
        draws = [draw_cam_artifact_randoms(art, rng, H, W, motion=mot[i]) for i in range(len(frames))]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(
                ex.map(
                    lambda i: apply_cam_artifacts(frames[i], art, rng=None, motion=mot[i], draws=draws[i]),
                    range(len(frames)),
                )
            )
    return [apply_cam_artifacts(fr, art, rng, motion=mot[i]) for i, fr in enumerate(frames)]
