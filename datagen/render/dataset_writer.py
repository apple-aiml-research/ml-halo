#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Training-dataset episode writer + per-shard meta writer.

Dataset layout (consumed by downstream training):

    <dataset_dir>/
      data/episode_NNNNNN.parquet
      videos/observation.images.fpv_cam/episode_NNNNNN.mp4
      videos/observation.images.tpv_cam/episode_NNNNNN.mp4
      meta/{info.json, episodes.jsonl, tasks.jsonl}

Both Isaac and MuJoCo render drivers share these writers — episodes from
either backend are interchangeable on disk.

Direct-write mode: when DIRECT_WRITE_DEST is set, every shard writes dataset
files (with global episode indices) straight to a shared dataset prefix —
no separate merge step. Per-shard meta lives at meta/shard_NNN/, stitched
by `render.stitch` at the end.

"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np


FPS = 30


# G1 29-DoF joint names, in the qpos[7:36] order produced by Kimodo / the MuJoCo /
# Isaac retarget. Same list as render.cameras.MUJOCO_JOINT_NAMES, duplicated so this
# module does not import OpenCV / MuJoCo.
G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def build_g1_features_block() -> dict:
    """LeRobot-style `features` block for G1 29-DoF datasets, stored in meta/info.json.

    generator/scripts/eval_openloop.py reads
    `info["features"]["action"]["names"]` to map model output indices back to MuJoCo
    joint names. Both the writer (render) and the stitcher (which backfills datasets
    whose per-shard info.json lacks it) write this block.
    """
    return {
        "action": {
            "dtype": "float32",
            "shape": [len(G1_JOINT_NAMES)],
            "names": list(G1_JOINT_NAMES),
        },
        "observation.joint_pos": {
            "dtype": "float32",
            "shape": [len(G1_JOINT_NAMES)],
            "names": list(G1_JOINT_NAMES),
        },
    }


# ----------------------------------------------------------------------------
# Per-episode dataset writer
# ----------------------------------------------------------------------------
def write_dataset_episode(ep_dir: str, dataset_dir: str, ep_idx: int, meta: dict,
                         fps: int = FPS) -> int:
    """One episode → dataset/data/episode_NNNNNN.parquet + dataset/videos/.../*.mp4.

    Returns the number of rows written to the parquet. Callers use this (not
    traj.npz's `num_frames`) for meta.length, so a truncated render (preemption
    mid-write, rolling-upload race) still gives a consistent dataset instead of a
    meta / parquet mismatch that breaks downstream samplers.

    `fps` sets the timestamp column (`i / fps`), not the mp4 frame rate (set when
    the renderer writes the fpv/tpv mp4s).

    Assumes ep_dir already has fpv.mp4, tpv.mp4, traj.npz, meta.json (assembled
    by the render driver).

    OAK: oak.mp4 fills the observation.images.fpv_cam slot; OAK_NATIVE_FISHEYE=1 (Isaac) copies it, =0 (MuJoCo)
    cv2.fisheye-warps the pinhole oak.mp4.
    """
    import pandas as pd

    traj = np.load(os.path.join(ep_dir, "traj.npz"))
    n_frames = int(traj.get("num_frames", traj["qpos"].shape[0]))
    qpos = traj["qpos"][:n_frames]

    pelvis_pos = qpos[:, :3].astype(np.float32)
    pelvis_quat = qpos[:, 3:7].astype(np.float32)
    joint_angles = qpos[:, 7:].astype(np.float32)
    action_deltas = np.zeros_like(joint_angles)
    action_deltas[:-1] = joint_angles[1:] - joint_angles[:-1]

    ident_q = np.array([1, 0, 0, 0], dtype=np.float32)
    # Task-level meta: when meta carries an `episode_spec` block (staged renders write
    # one), use its task_type / target / scene_id; otherwise use the lab sit-on-chair
    # values.
    ep_spec = meta.get("episode_spec")
    if ep_spec:
        task_type = ep_spec["task_type"]
        target_object_id = ep_spec.get("target_object_id", "")
        target_xyz = list(ep_spec.get("target_xyz", [0.0, 0.0, 0.0]))
        scene_id = ep_spec.get("scene_id", "")
        objective_category = target_object_id or "object"
        verb_category = task_type
        task_name = f"{task_type}_{objective_category}_{meta.get('seq_name', '')}".rstrip("_")
    else:
        # Episodes without an episode_spec keep the original sit columns.
        objective_category = meta.get("furniture_type", "chair")
        verb_category = "sit on furniture"
        task_name = f"{objective_category}_{meta.get('seq_name', 'sit')}"
    rows = []
    for i in range(n_frames):
        bp = np.empty(30, dtype=object)
        bq = np.empty(30, dtype=object)
        bp[0] = pelvis_pos[i]
        bq[0] = pelvis_quat[i]
        for b in range(1, 30):
            bp[b] = np.zeros(3, dtype=np.float32)
            bq[b] = ident_q.copy()
        row = {
            "action": action_deltas[i],
            "observation.joint_pos": joint_angles[i],
            "observation.body_pos": bp,
            "observation.body_quat": bq,
            "info.task_name": task_name,
            "info.objective_category": objective_category,
            "info.verb_category": verb_category,
        }
        if ep_spec:
            row.update({
                "info.task_type": task_type,
                "info.target_object_id": target_object_id,
                "info.target_xyz": np.asarray(target_xyz, dtype=np.float32),
                "info.scene_id": scene_id,
            })
        row.update({
            "timestamp": np.float32(i / fps),
            "frame_index": np.int64(i),
            "episode_index": np.int64(ep_idx),
            "index": np.int64(i),
            "task_index": np.int64(0),
        })
        rows.append(row)

    data_dir = os.path.join(dataset_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    parquet_path = os.path.join(data_dir, f"episode_{ep_idx:06d}.parquet")
    pd.DataFrame(rows).to_parquet(parquet_path)

    # Source of truth for episode length is the parquet itself, not n_frames
    # (which came from traj.npz before any potential write-time truncation).
    import pyarrow.parquet as pq
    n_actual = pq.read_metadata(parquet_path).num_rows
    if n_actual != n_frames:
        print(f"[dataset_writer] WARNING ep_{ep_idx:06d}: traj.num_frames={n_frames} "
              f"but parquet has {n_actual} rows — using {n_actual} for meta", flush=True)

    # Store the full EpisodeSpec next to the parquet so closed-loop eval can rebuild
    # the scene (placements, target, planned path, lighting, robot spawn) without
    # re-sampling from a seed. Lab sit-on-chair episodes have no episode_spec block;
    # they get a minimal sidecar so the layout is uniform.
    ep_spec_block = meta.get("episode_spec")
    if ep_spec_block:
        ep_specs_dir = os.path.join(dataset_dir, "episode_specs")
        os.makedirs(ep_specs_dir, exist_ok=True)
        with open(os.path.join(ep_specs_dir, f"episode_{ep_idx:06d}.json"), "w") as f:
            json.dump(ep_spec_block, f, indent=2)

    for cam, src_name in [("fpv_cam", "fpv.mp4"), ("tpv_cam", "tpv.mp4")]:
        src = os.path.join(ep_dir, src_name)
        if not os.path.isfile(src):
            continue
        dst_dir = os.path.join(dataset_dir, "videos", f"observation.images.{cam}")
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(dst_dir, f"episode_{ep_idx:06d}.mp4"))

    # OAK: Isaac renders the calibrated fisheye natively (copied); MuJoCo's pinhole oak.mp4 is warped here, once.
    oak_mp4 = os.path.join(ep_dir, "oak.mp4")
    if os.path.isfile(oak_mp4):
        oak_dst_dir = os.path.join(dataset_dir, "videos", "observation.images.fpv_cam")
        os.makedirs(oak_dst_dir, exist_ok=True)
        oak_dst = os.path.join(oak_dst_dir, f"episode_{ep_idx:06d}.mp4")
        if meta.get("oak_already_warped", False) or os.environ.get("OAK_NATIVE_FISHEYE", "1") == "1":
            shutil.copy2(oak_mp4, oak_dst)
        else:
            from render.cameras import load_oak_intrinsics
            from render.fisheye_warp import warp_pinhole_mp4_to_fisheye_mp4

            warp_pinhole_mp4_to_fisheye_mp4(oak_mp4, oak_dst, load_oak_intrinsics(), fps=FPS)
    return n_actual


# ----------------------------------------------------------------------------
# Per-shard dataset meta
# ----------------------------------------------------------------------------
def _task_str(e):
    """Language label of one episode: "<task_type> <target>" for staged episodes,
    "sit on <furniture>" for lab-sofa episodes (no episode_spec)."""
    ep_spec = e["meta"].get("episode_spec")
    if ep_spec:
        return f"{ep_spec['task_type']} {ep_spec.get('target_object_id', 'object')}"
    return f"sit on {e['meta'].get('furniture_type', 'chair')}"


def write_dataset_shard_meta(
    dataset_dir: str,
    episodes: list[dict],
    shard_subpath: str | None = None,
    fps: int = FPS,
) -> None:
    """Write per-shard meta for this batch of episodes.

    When ``shard_subpath`` is provided (e.g. "shard_007"), per-shard meta files
    are written under ``meta/<shard_subpath>/`` instead of ``meta/``. Used by
    the direct-write production pattern where many shards push to the same
    dataset prefix; a tiny post-job stitcher then concatenates the per-shard
    jsonls into the canonical top-level meta/episodes.jsonl.

    `fps` is recorded in info.json's `fps` field — the dataset's nominal rate.
    """
    if shard_subpath:
        meta_dir = os.path.join(dataset_dir, "meta", shard_subpath)
    else:
        meta_dir = os.path.join(dataset_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    total_frames = sum(e["n_frames"] for e in episodes)

    with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
        for e in episodes:
            f.write(json.dumps({
                "episode_index": e["ep_idx"],
                "tasks": [_task_str(e)],
                "length": e["n_frames"],
            }) + "\n")

    tasks = sorted({_task_str(e) for e in episodes})
    with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
        for i, t in enumerate(tasks):
            f.write(json.dumps({"task_index": i, "task": t}) + "\n")

    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump({
            "codebase_version": "egomotion-gen-0.1.0",
            "robot_type": "unitree_g1_29dof",
            "total_episodes": len(episodes),
            "total_frames": total_frames,
            "fps": fps,
            "data_path": "data/episode_{episode_index:06d}.parquet",
            "video_path": "videos/{video_key}/episode_{episode_index:06d}.mp4",
            "features": build_g1_features_block(),
        }, f, indent=2)

