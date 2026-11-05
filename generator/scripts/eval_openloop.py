#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
Open-loop evaluation script for generator.

Loads one episode from training data, runs model inference for each
valid start frame, compares predicted trajectories with ground truth,
and produces a video/GIF visualization with 3D skeleton rendering.

Usage (run from generator/):
    python scripts/eval_openloop.py \
        --checkpoint checkpoints/best_model.pt \
        --config configs/qwen3_5_2b_dit_vision_finetune.yaml \
        --episode_idx 0
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add generator root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from PIL import Image
import io

from configs.config import Config, load_config
from models.motion_generator import MotionGenerator, MotionGeneratorConfig
from data.video_reader import iter_frames
from data.transforms import get_vlm_image_transform
from evaluation.metrics import compute_all_metrics, MetricTracker

# Default path to the G1 MuJoCo scene (relative to generator/)
DEFAULT_MJCF_PATH = str(Path(__file__).parent.parent.parent / "assets" / "robots" / "g1" / "mjcf" / "scene_g1_29dof.xml")

# MuJoCo rendering size per skeleton panel
RENDER_SIZE = 480


def create_model_config(config: Config) -> MotionGeneratorConfig:
    """Create model config from training config.

    Shared with scripts/train.py via model_config_from, so every model option
    the checkpoint was trained with is reproduced here.
    """
    from models.motion_generator import model_config_from
    return model_config_from(config)


def load_model(checkpoint_path, config, device):
    """Load trained model from checkpoint."""
    model_config = create_model_config(config)
    model = MotionGenerator(model_config)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    res = model.load_state_dict(state_dict, strict=False)
    # strict=False is needed because frozen VLM weights come from HuggingFace, not
    # the checkpoint. But anything ELSE that fails to match means the config does
    # not describe the model that was trained — and loading it anyway silently
    # evaluates a different model (this is how cols8+pos checkpoints were
    # evaluated with a mean-pooled token and no positional embedding). Refuse.
    unexpected = [k for k in res.unexpected_keys if not k.startswith("_normalizer.")]
    missing = [k for k in res.missing_keys if not k.startswith("vlm_encoder.")]
    if unexpected or missing:
        raise RuntimeError(
            "checkpoint does not match the config it is being loaded with:\n"
            f"  in checkpoint but not in model: {unexpected[:10]}\n"
            f"  in model but not in checkpoint: {missing[:10]}\n"
            "Use the training config of the run that produced this checkpoint.")
    model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "unknown")
    best_loss = checkpoint.get("best_val_loss", "N/A")
    print(f"Loaded checkpoint from epoch {epoch} (best_val_loss={best_loss})")
    if model.normalizer is not None:
        print("Motion normalizer loaded from checkpoint")

    return model


def episode_root(data_dir, cache_dir, episode_idx):
    """Local directory holding the episode's files. For an s3:// dataset, its
    meta + episode files are fetched into the cache copy first."""
    from data import remote
    root = remote.local_dir(data_dir, cache_dir)
    if remote.is_s3(data_dir):
        for rel in ("meta/info.json", "meta/episodes.jsonl",
                    f"data/episode_{episode_idx:06d}.parquet",
                    f"videos/observation.images.fpv_cam/episode_{episode_idx:06d}.mp4"):
            remote.download(f"{data_dir.rstrip('/')}/{rel}", root / rel)
    return root


def load_episode(data_dir, episode_idx):
    """Load all data for a single episode."""
    data_dir = Path(data_dir)

    # by episode_index, not line number: datagen's indices are gapped (shard * 1000 + i)
    episodes = {}
    with open(data_dir / "meta" / "episodes.jsonl") as f:
        for line in f:
            e = json.loads(line)
            episodes[e["episode_index"]] = e
    if episode_idx not in episodes:
        raise KeyError(f"episode {episode_idx} is not in {data_dir / 'meta' / 'episodes.jsonl'}")

    episode_meta = episodes[episode_idx]
    episode_length = episode_meta["length"]
    language_cmd = episode_meta["tasks"][0]

    from data.dataset import read_episode_parquet
    data = read_episode_parquet(data_dir / "data" / f"episode_{episode_idx:06d}.parquet")

    video_path = (
        data_dir / "videos" / "observation.images.fpv_cam"
        / f"episode_{episode_idx:06d}.mp4"
    )
    return data, video_path, language_cmd, episode_length


def extract_gt_trajectory(data, start_frame, trajectory_length=30):
    """Ground-truth target window: exactly the training target (data/traj_extract.py),
    including tail padding by repeating the last real frame."""
    from data.traj_extract import extract_trajectory_vectorized
    return extract_trajectory_vectorized(data, start_frame, trajectory_length)


def extract_gt_absolute_state(data, frame_idx):
    """Extract absolute joint positions and pelvis state at a frame."""
    joint_pos = np.array(data["observation.joint_pos"][frame_idx], dtype=np.float32)
    body_pos = np.array(data["observation.body_pos"][frame_idx], dtype=np.float32)
    pelvis_pos = body_pos[0]
    body_quat = np.array(data["observation.body_quat"][frame_idx], dtype=np.float32)
    pelvis_quat = body_quat[0]
    return joint_pos, pelvis_pos, pelvis_quat


def pred_to_absolute(pred, init_pelvis_pos, init_pelvis_quat):
    """
    Convert predicted trajectory (absolute joints, body-frame relative pelvis)
    to absolute world-frame coordinates for visualization.

    Args:
        pred: [T, 36] tensor — absolute joints, body-frame relative pelvis pos/quat
        init_pelvis_pos: [3] numpy — pelvis position at t0 (world frame)
        init_pelvis_quat: [4] numpy — pelvis quaternion at t0 (world frame)

    Returns:
        (abs_joints [T, 29], abs_pelvis_pos [T, 3], abs_pelvis_quat [T, 4])
    """
    from data.quaternion import quat_multiply, quat_rotate_vector, yaw_of, yaw_quat

    abs_joints = pred[:, :29].numpy()

    # Inverse YAW-ONLY gauge: re-apply +yaw0 only (mirror of qz_inv in the extractor).
    yaw0 = yaw_of(init_pelvis_quat)
    qz = yaw_quat(yaw0)

    # De-yawed relative pelvis pos -> world: pos_world = pos_t0 + Rz(yaw0) @ pos_rel
    rel_pelvis_pos = pred[:, 29:32].numpy()
    qz_bcast = np.broadcast_to(qz, rel_pelvis_pos.shape[:-1] + (4,))
    world_displacement = quat_rotate_vector(qz_bcast, rel_pelvis_pos)
    abs_pelvis_pos = world_displacement + init_pelvis_pos[np.newaxis, :]

    # Relative pelvis quat -> absolute: q_world = qz(yaw0) * q_rel  (re-apply yaw only;
    # the pitch/roll are already carried in q_rel)
    rel_pelvis_quat = pred[:, 32:36].numpy()
    abs_pelvis_quat = quat_multiply(
        np.broadcast_to(qz, rel_pelvis_quat.shape),
        rel_pelvis_quat,
    )

    return abs_joints, abs_pelvis_pos, abs_pelvis_quat


def compute_frame_errors(pred, gt):
    """Compute per-frame errors for display."""
    mpjpe = torch.abs(pred[:, :29] - gt[:, :29]).mean().item()
    pelvis_l2 = torch.norm(pred[:, 29:32] - gt[:, 29:32], dim=-1).mean().item()
    return mpjpe, pelvis_l2


# ── MuJoCo skeleton rendering ──────────────────────────────────────────────


def build_training_to_mj_map(mj_model, joint_names):
    """Build mapping from training joint order to MuJoCo qpos indices."""
    training_to_mj = []
    for jname in joint_names:
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise ValueError(f"Joint '{jname}' not found in MuJoCo model")
        training_to_mj.append(mj_model.jnt_qposadr[jid])
    return training_to_mj


def set_mujoco_state(mj_model, mj_data, joint_angles, pelvis_pos, pelvis_quat,
                     training_to_mj):
    """Set MuJoCo state and run forward kinematics."""
    mj_data.qpos[:] = 0
    mj_data.qpos[0:3] = pelvis_pos
    mj_data.qpos[3:7] = pelvis_quat
    for i, qpos_idx in enumerate(training_to_mj):
        mj_data.qpos[qpos_idx] = joint_angles[i]
    mujoco.mj_forward(mj_model, mj_data)


def render_robot(renderer, mj_data, camera):
    """Render current MuJoCo state to RGB array."""
    renderer.update_scene(mj_data, camera=camera)
    return renderer.render().copy()


def make_camera(pelvis_pos, distance=3.0, azimuth=135, elevation=-20):
    """Create a MuJoCo camera tracking the pelvis."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = pelvis_pos
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


# ── Stick figure skeleton overlay ───────────────────────────────────────────

# Skeleton bones defined as (parent_body, child_body) pairs
SKELETON_BONES = [
    # Left leg
    ("pelvis", "left_hip_pitch_link"),
    ("left_hip_pitch_link", "left_hip_yaw_link"),
    ("left_hip_yaw_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_pitch_link"),
    ("left_ankle_pitch_link", "left_ankle_roll_link"),
    # Right leg
    ("pelvis", "right_hip_pitch_link"),
    ("right_hip_pitch_link", "right_hip_yaw_link"),
    ("right_hip_yaw_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_pitch_link"),
    ("right_ankle_pitch_link", "right_ankle_roll_link"),
    # Torso
    ("pelvis", "torso_link"),
    # Head (approximate with torso top)
    ("torso_link", "left_shoulder_pitch_link"),
    ("torso_link", "right_shoulder_pitch_link"),
    # Left arm
    ("left_shoulder_pitch_link", "left_shoulder_yaw_link"),
    ("left_shoulder_yaw_link", "left_elbow_link"),
    ("left_elbow_link", "left_wrist_roll_link"),
    ("left_wrist_roll_link", "left_wrist_yaw_link"),
    # Right arm
    ("right_shoulder_pitch_link", "right_shoulder_yaw_link"),
    ("right_shoulder_yaw_link", "right_elbow_link"),
    ("right_elbow_link", "right_wrist_roll_link"),
    ("right_wrist_roll_link", "right_wrist_yaw_link"),
]


def get_body_positions(mj_model, mj_data):
    """Extract all body positions as a dict after FK has been run.

    Skips the world body (index 0) which is always at the origin.
    """
    positions = {}
    for i in range(1, mj_model.nbody):  # skip world body at index 0
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
        positions[name] = mj_data.xpos[i].copy()
    return positions


def draw_stick_figure(ax, positions, color, alpha=1.0, linewidth=1.5):
    """Draw a stick figure skeleton on a matplotlib axis.

    Uses XZ plane (front view): X = horizontal, Z = vertical.
    """
    for parent, child in SKELETON_BONES:
        if parent in positions and child in positions:
            p = positions[parent]
            c = positions[child]
            ax.plot([p[0], c[0]], [p[2], c[2]],
                    color=color, alpha=alpha, linewidth=linewidth)


def render_frame(
    rgb_frame,
    gt_abs_joints,
    gt_pelvis_pos,
    gt_pelvis_quat,
    pred_abs_joints,
    pred_pelvis_pos,
    pred_pelvis_quat,
    frame_idx,
    language_cmd,
    mpjpe,
    pelvis_l2,
    renderer,
    mj_model,
    mj_data,
    training_to_mj,
    pred_abs_joints_all,
    pred_pelvis_pos_all,
    pred_pelvis_quat_all,
    gt_abs_joints_all,
    gt_pelvis_pos_all,
    gt_pelvis_quat_all,
    plan_half_span=None,
):
    """
    Render a single evaluation frame as a PIL Image.

    Layout: [RGB input] [GT skeleton] [Predicted skeleton] [GT motion plan] [Pred motion plan]
    """
    # Render GT skeleton
    set_mujoco_state(mj_model, mj_data, gt_abs_joints, gt_pelvis_pos, gt_pelvis_quat,
                     training_to_mj)
    cam_gt = make_camera(gt_pelvis_pos)
    gt_render = render_robot(renderer, mj_data, cam_gt)

    # Render predicted skeleton (t=0)
    set_mujoco_state(mj_model, mj_data, pred_abs_joints, pred_pelvis_pos, pred_pelvis_quat,
                     training_to_mj)
    cam_pred = make_camera(pred_pelvis_pos)
    pred_render = render_robot(renderer, mj_data, cam_pred)

    # Compute body positions for all 30 timesteps (prediction)
    T = len(pred_abs_joints_all)
    pred_body_positions = []
    for t in range(T):
        set_mujoco_state(
            mj_model, mj_data,
            pred_abs_joints_all[t], pred_pelvis_pos_all[t], pred_pelvis_quat_all[t],
            training_to_mj,
        )
        pred_body_positions.append(get_body_positions(mj_model, mj_data))

    # Compute body positions for all 30 timesteps (ground truth)
    T_gt = len(gt_abs_joints_all)
    gt_body_positions = []
    for t in range(T_gt):
        set_mujoco_state(
            mj_model, mj_data,
            gt_abs_joints_all[t], gt_pelvis_pos_all[t], gt_pelvis_quat_all[t],
            training_to_mj,
        )
        gt_body_positions.append(get_body_positions(mj_model, mj_data))

    # Compose with matplotlib
    fig = plt.figure(figsize=(25, 5), dpi=100)
    gs = GridSpec(1, 5, wspace=0.05)

    # Panel 1: RGB input
    ax_rgb = fig.add_subplot(gs[0, 0])
    rgb_np = rgb_frame.permute(1, 2, 0).numpy()
    rgb_np = np.clip(rgb_np, 0, 1)
    ax_rgb.imshow(rgb_np)
    ax_rgb.set_title("Input RGB", fontsize=10)
    ax_rgb.set_xticks([])
    ax_rgb.set_yticks([])

    # Panel 2: Predicted motion plan overlay (stick figures)
    ax_pred_plan = fig.add_subplot(gs[0, 1])
    ax_pred_plan.set_facecolor('#1a1a2e')
    cmap = plt.cm.coolwarm
    for t in reversed(range(T)):
        frac = t / max(T - 1, 1)
        color = cmap(frac)
        alpha = 1.0 - 0.7 * frac
        draw_stick_figure(ax_pred_plan, pred_body_positions[t], color, alpha=alpha, linewidth=2.0)
    ax_pred_plan.set_title("Pred Plan", fontsize=10)
    ax_pred_plan.set_aspect('equal')
    if plan_half_span is not None:
        all_x = []
        all_z = []
        for bp in pred_body_positions:
            for name in bp:
                all_x.append(bp[name][0])
                all_z.append(bp[name][2])
        cx = (min(all_x) + max(all_x)) / 2.0
        cz = (min(all_z) + max(all_z)) / 2.0
        ax_pred_plan.set_xlim(cx - plan_half_span, cx + plan_half_span)
        ax_pred_plan.set_ylim(cz - plan_half_span, cz + plan_half_span)
    ax_pred_plan.set_xticks([])
    ax_pred_plan.set_yticks([])

    # Panel 3: GT motion plan overlay (stick figures)
    ax_gt_plan = fig.add_subplot(gs[0, 2])
    ax_gt_plan.set_facecolor('#1a1a2e')
    for t in reversed(range(T_gt)):
        frac = t / max(T_gt - 1, 1)
        color = cmap(frac)
        alpha = 1.0 - 0.7 * frac
        draw_stick_figure(ax_gt_plan, gt_body_positions[t], color, alpha=alpha, linewidth=2.0)
    ax_gt_plan.set_title("GT Plan", fontsize=10)
    ax_gt_plan.set_aspect('equal')
    if plan_half_span is not None:
        all_x = []
        all_z = []
        for bp in gt_body_positions:
            for name in bp:
                all_x.append(bp[name][0])
                all_z.append(bp[name][2])
        cx = (min(all_x) + max(all_x)) / 2.0
        cz = (min(all_z) + max(all_z)) / 2.0
        ax_gt_plan.set_xlim(cx - plan_half_span, cx + plan_half_span)
        ax_gt_plan.set_ylim(cz - plan_half_span, cz + plan_half_span)
    ax_gt_plan.set_xticks([])
    ax_gt_plan.set_yticks([])

    # Panel 4: Predicted skeleton (MuJoCo)
    ax_pred = fig.add_subplot(gs[0, 3])
    ax_pred.imshow(pred_render)
    ax_pred.set_title("Prediction (MuJoCo)", fontsize=10)
    ax_pred.set_xticks([])
    ax_pred.set_yticks([])

    # Panel 5: GT skeleton (MuJoCo)
    ax_gt = fig.add_subplot(gs[0, 4])
    ax_gt.imshow(gt_render)
    ax_gt.set_title("Ground Truth (MuJoCo)", fontsize=10)
    ax_gt.set_xticks([])
    ax_gt.set_yticks([])

    fig.suptitle(
        f'Frame {frame_idx} | MPJPE={mpjpe:.4f} | Pelvis L2={pelvis_l2:.4f} | '
        f'Task: "{language_cmd}"',
        fontsize=9, y=0.02,
    )

    # Render to PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    pil_img = Image.open(buf).copy()
    buf.close()
    return pil_img


def evaluate_episode(
    model,
    data,
    video_path,
    language_cmd,
    episode_length,
    image_transform,
    device,
    joint_names,
    mj_model,
    mj_data,
    renderer,
    training_to_mj,
    trajectory_length=30,
    max_start_frames=0,
):
    """Run open-loop evaluation on all valid start frames of an episode."""
    # One eval window per frame in the episode — matches training's
    # data/dataset.py:_build_sample_index exactly (every frame is a valid start;
    # windows past the episode end are tail-padded by repeating the last real
    # frame in extract_gt_trajectory).
    start_frames = list(range(0, episode_length))
    # On long episodes, running the VLM + sampling + render on EVERY start frame
    # can overrun the trainer's 30-min subprocess cap. When max_start_frames>0,
    # evenly subsample to that many windows (representative metrics + a gif in a
    # fraction of the time). Default 0 = all frames.
    if max_start_frames and len(start_frames) > max_start_frames:
        stride = -(-len(start_frames) // max_start_frames)  # ceil division
        start_frames = start_frames[::stride]

    # Decode the video once, keeping the start frames ([3, H, W] uint8).
    wanted = set(start_frames)
    frames = {i: f for i, f in enumerate(iter_frames(video_path)) if i in wanted}

    all_predictions = []
    all_ground_truths = []
    gif_frames = []
    metric_tracker = MetricTracker()

    # First pass: run inference and compute consistent window size for motion plan panel
    print("Pass 1: Running inference and computing motion plan bounds...")
    frame_data_cache = []
    # Track the max skeleton extent (body span at a single timestep)
    max_body_x_extent = 0.0
    max_body_z_extent = 0.0

    for start_frame in tqdm(start_frames, desc="Inference"):
        rgb_raw = frames[start_frame].float() / 255.0
        rgb_input = image_transform(rgb_raw).unsqueeze(0).to(device)

        gt_abs_joints, gt_pelvis_pos, gt_pelvis_quat = extract_gt_absolute_state(
            data, start_frame)

        gt = extract_gt_trajectory(data, start_frame, trajectory_length)

        # prev_chunk inpainting: condition on the true first 0.5s of the future
        # plan (normalized to match the training space). The model hard-fixes
        # these frames so the prediction's prefix matches the GT prefix exactly.
        predict_kwargs = {}
        if model.inpaint_prev_chunk:
            P = model.prev_chunk_frames
            prev_chunk = gt[:P].clone()
            if model.normalizer is not None:
                prev_chunk = model.normalizer.normalize(prev_chunk)
            predict_kwargs["prev_chunk"] = prev_chunk.to(device)

        pred = model.predict(rgb_input, language_cmd, **predict_kwargs)
        pred = pred.cpu()
        if pred.dim() == 3:
            pred = pred.squeeze(0)

        all_predictions.append(pred)
        all_ground_truths.append(gt)

        metrics = compute_all_metrics(pred.unsqueeze(0), gt.unsqueeze(0))
        metric_tracker.update(metrics, count=1)
        mpjpe, pelvis_l2 = compute_frame_errors(pred, gt)

        pred_abs_joints, pred_pelvis_pos, pred_pelvis_quat = pred_to_absolute(
            pred, gt_pelvis_pos, gt_pelvis_quat)

        # Extract GT absolute states for full trajectory-length window (pad tail
        # by clamping to last frame, same as extract_gt_trajectory).
        gt_abs_joints_seq = []
        gt_pelvis_pos_seq = []
        gt_pelvis_quat_seq = []
        for t in range(trajectory_length):
            fi = min(start_frame + t, episode_length - 1)
            j, p, q = extract_gt_absolute_state(data, fi)
            gt_abs_joints_seq.append(j)
            gt_pelvis_pos_seq.append(p)
            gt_pelvis_quat_seq.append(q)
        gt_abs_joints_seq = np.stack(gt_abs_joints_seq)
        gt_pelvis_pos_seq = np.stack(gt_pelvis_pos_seq)
        gt_pelvis_quat_seq = np.stack(gt_pelvis_quat_seq)

        # Compute body positions for all T timesteps, track max body extent
        T = len(pred_abs_joints)
        for t in range(T):
            # Prediction extents
            set_mujoco_state(
                mj_model, mj_data,
                pred_abs_joints[t], pred_pelvis_pos[t], pred_pelvis_quat[t],
                training_to_mj,
            )
            bp = get_body_positions(mj_model, mj_data)
            xs = [bp[n][0] for n in bp]
            zs = [bp[n][2] for n in bp]
            max_body_x_extent = max(max_body_x_extent, max(xs) - min(xs))
            max_body_z_extent = max(max_body_z_extent, max(zs) - min(zs))

            # GT extents
            set_mujoco_state(
                mj_model, mj_data,
                gt_abs_joints_seq[t], gt_pelvis_pos_seq[t], gt_pelvis_quat_seq[t],
                training_to_mj,
            )
            bp = get_body_positions(mj_model, mj_data)
            xs = [bp[n][0] for n in bp]
            zs = [bp[n][2] for n in bp]
            max_body_x_extent = max(max_body_x_extent, max(xs) - min(xs))
            max_body_z_extent = max(max_body_z_extent, max(zs) - min(zs))

        frame_data_cache.append({
            'rgb_raw': rgb_raw,
            'gt_abs_joints': gt_abs_joints,
            'gt_pelvis_pos': gt_pelvis_pos,
            'gt_pelvis_quat': gt_pelvis_quat,
            'pred_abs_joints': pred_abs_joints,
            'pred_pelvis_pos': pred_pelvis_pos,
            'pred_pelvis_quat': pred_pelvis_quat,
            'gt_abs_joints_seq': gt_abs_joints_seq,
            'gt_pelvis_pos_seq': gt_pelvis_pos_seq,
            'gt_pelvis_quat_seq': gt_pelvis_quat_seq,
            'mpjpe': mpjpe,
            'pelvis_l2': pelvis_l2,
            'start_frame': start_frame,
        })

    # Use the max body extent (with generous padding) as consistent window size
    # This ensures the skeleton fills most of the panel
    plan_half_span = max(max_body_x_extent, max_body_z_extent) / 2.0 * 1.5  # 50% padding

    # Second pass: render frames with consistent window size
    print("Pass 2: Rendering frames...")
    for cached in tqdm(frame_data_cache, desc="Rendering"):
        frame_img = render_frame(
            cached['rgb_raw'],
            cached['gt_abs_joints'], cached['gt_pelvis_pos'], cached['gt_pelvis_quat'],
            cached['pred_abs_joints'][0], cached['pred_pelvis_pos'][0], cached['pred_pelvis_quat'][0],
            cached['start_frame'], language_cmd,
            cached['mpjpe'], cached['pelvis_l2'],
            renderer, mj_model, mj_data, training_to_mj,
            cached['pred_abs_joints'], cached['pred_pelvis_pos'], cached['pred_pelvis_quat'],
            cached['gt_abs_joints_seq'], cached['gt_pelvis_pos_seq'], cached['gt_pelvis_quat_seq'],
            plan_half_span=plan_half_span,
        )
        gif_frames.append(frame_img)

    all_pred = torch.stack(all_predictions)
    all_gt = torch.stack(all_ground_truths)

    avg_metrics = metric_tracker.compute()
    batch_metrics = compute_all_metrics(all_pred, all_gt)

    return avg_metrics, batch_metrics, gif_frames


def save_gif(frames, output_path, fps=5):
    """Save list of PIL images as GIF."""
    if not frames:
        return
    duration = int(1000 / fps)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )


def save_mp4(frames, output_path, fps=5):
    """Save list of PIL images as MP4 using ffmpeg subprocess."""
    if not frames:
        return

    w, h = frames[0].size
    w_out = w if w % 2 == 0 else w + 1
    h_out = h if h % 2 == 0 else h + 1

    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{w_out}x{h_out}',
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', '-',
        '-an',
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        str(output_path),
    ]

    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    for frame in frames:
        if (w, h) != (w_out, h_out):
            frame = frame.resize((w_out, h_out), Image.LANCZOS)
        arr = np.array(frame.convert('RGB'), dtype=np.uint8)
        proc.stdin.write(arr.tobytes())
    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode()
        raise RuntimeError(f"ffmpeg failed: {stderr[-200:]}")


def print_and_save_metrics(avg_metrics, batch_metrics, episode_idx, language_cmd, output_dir):
    """Print metrics to console and save to JSON."""
    print("\n" + "=" * 60)
    print("Open-Loop Evaluation Results")
    print(f"Episode: {episode_idx}")
    print(f"Task: {language_cmd}")
    print("=" * 60)

    print("\nPer-Sample Averaged Metrics:")
    for name, value in sorted(avg_metrics.items()):
        print(f"  {name:25s}: {value:.6f}")

    print("\nBatch Metrics:")
    for name, value in sorted(batch_metrics.items()):
        print(f"  {name:25s}: {value:.6f}")
    print("=" * 60)

    results = {
        "episode_idx": episode_idx,
        "language_command": language_cmd,
        "avg_metrics": avg_metrics,
        "batch_metrics": batch_metrics,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Open-loop evaluation for generator"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--episode_idx", type=int, default=0,
        help="Episode index to evaluate",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Override data.data_dir (local directory or s3:// URI)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="eval_output",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (cuda, cpu, mps)",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="FPS for output GIF/video",
    )
    parser.add_argument(
        "--max_start_frames", type=int, default=0,
        help="If >0, evenly subsample this many start-frame eval windows "
             "(keeps in-training eval under the subprocess timeout on long "
             "episodes). 0 = all frames (full coverage).",
    )
    parser.add_argument(
        "--mjcf_path", type=str, default=DEFAULT_MJCF_PATH,
        help="Path to MuJoCo scene XML for skeleton rendering",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.data_dir is not None:
        config.data.data_dir = args.data_dir

    device = args.device or config.device
    if device == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            device = "mps"
            print("CUDA not available, using MPS (Apple Silicon)")
        else:
            device = "cpu"
            print("CUDA not available, using CPU")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading model...")
    model = load_model(args.checkpoint, config, device)

    print("Loading episode data...")
    root = episode_root(config.data.data_dir, config.data.cache_dir, args.episode_idx)
    data, video_path, language_cmd, episode_length = load_episode(
        root, args.episode_idx,
    )

    with open(root / "meta" / "info.json") as f:
        info = json.load(f)
    joint_names = info["features"]["action"]["names"]

    print("Loading MuJoCo model for skeleton rendering...")
    mj_model = mujoco.MjModel.from_xml_path(args.mjcf_path)
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=RENDER_SIZE, width=RENDER_SIZE)
    training_to_mj = build_training_to_mj_map(mj_model, joint_names)
    print(f"MuJoCo model loaded: {mj_model.nbody} bodies, {mj_model.njnt} joints")

    image_transform = get_vlm_image_transform(config.data.image_size)

    traj_len = config.model.trajectory_length
    num_valid = episode_length - traj_len + 1

    print(f"\nEpisode {args.episode_idx}: '{language_cmd}'")
    print(f"Episode length: {episode_length} frames")
    print(f"Evaluating {num_valid} start frames")

    avg_metrics, batch_metrics, gif_frames = evaluate_episode(
        model, data, video_path, language_cmd, episode_length,
        image_transform, device, joint_names,
        mj_model, mj_data, renderer, training_to_mj,
        trajectory_length=traj_len,
        max_start_frames=args.max_start_frames,
    )

    print_and_save_metrics(
        avg_metrics, batch_metrics,
        args.episode_idx, language_cmd, output_dir,
    )

    print(f"\nSaving {len(gif_frames)} frames...")

    gif_path = output_dir / f"episode_{args.episode_idx:06d}_eval.gif"
    save_gif(gif_frames, gif_path, fps=args.fps)
    print(f"GIF saved to: {gif_path}")

    mp4_path = output_dir / f"episode_{args.episode_idx:06d}_eval.mp4"
    try:
        save_mp4(gif_frames, mp4_path, fps=args.fps)
        print(f"MP4 saved to: {mp4_path}")
    except Exception as e:
        print(f"MP4 save failed ({e}), GIF is available.")

    renderer.close()

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
