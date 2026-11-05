#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Inference-side helpers shared between the run_sim_* and run_real_* scripts."""
from __future__ import annotations

import math

import numpy as np
import torch


def joint_activity(plan_qpos) -> float:
    """Mean frame-to-frame joint delta ‖Δjoints‖ (rad) over the plan.

    Logged per plan and used for the settle / hold thresholds. ``plan_qpos`` is a
    [N, 36] plan = pos(3), quat_xyzw(4), joints_deploy(29); the metric uses
    cols 7:36 (the 29 deploy-order joints). High for dynamic / posture-changing
    motion (e.g. sit, walk), ~0 for a static hold. A plan with <2 frames (or
    None) has no motion and returns 0.0.
    """
    if plan_qpos is None or len(plan_qpos) < 2:
        return 0.0
    joints = np.asarray(plan_qpos[:, 7:36], dtype=np.float64)
    dq = np.diff(joints, axis=0)
    return float(np.linalg.norm(dq, axis=1).mean())


# Plan/qpos columns of left_hip_pitch_joint / right_hip_pitch_joint ([0:3] pos, [3:7] quat XYZW,
# [7:36] joints in MJCF order) and their range in g1_29dof_rev_1_0.xml.
_HIP_PITCH_COLS = (7, 13)
_HIP_PITCH_RANGE = (-2.5307, 2.8798)
# Hip pitch axes (both on one line along pelvis y) pass through this point in the pelvis frame
# (left/right_hip_pitch_link pos = (0, +-0.064452, -0.1027)).
_HIP_PIVOT = np.array([0.0, 0.0, -0.1027])


def _rotate_xyzw(q, v):
    x, y, z, w = (float(c) for c in q)
    u = np.array([x, y, z])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def sit_lean_pose(seated, lean_deg: float) -> np.ndarray:
    """The seated reference frame with the upper body leaned forward by `lean_deg`.

    The pelvis pitches forward about the hip pitch axis (+y pitch = nose down) and both hip pitch
    joints move by the same angle the other way, so the thighs (and everything below) keep their world
    pose: the robot stays on the seat and only the trunk tips over the knees. Yaw and every other
    joint are unchanged; the pelvis position moves only as much as rotating about the hips needs.
    """
    P = np.array(seated, dtype=np.float64, copy=True)
    th = math.radians(float(lean_deg))
    q0 = P[3:7].copy()
    x1, y1, z1, w1 = (float(v) for v in q0)
    s, c = math.sin(th / 2.0), math.cos(th / 2.0)          # q0 * q_y(th), q_y = (0, s, 0, c) XYZW
    q = np.array([x1 * c - z1 * s, w1 * s + y1 * c, z1 * c + x1 * s, w1 * c - y1 * s])
    q /= max(float(np.linalg.norm(q)), 1e-9)
    P[3:7] = q
    P[0:3] += _rotate_xyzw(q0, _HIP_PIVOT) - _rotate_xyzw(q, _HIP_PIVOT)   # keep the hip axis fixed
    for col in _HIP_PITCH_COLS:
        P[col] = float(np.clip(P[col] - th, *_HIP_PITCH_RANGE))
    return P


def ramp_plan(ref0, knots, t_now: float, n_frames: int, fps: float) -> np.ndarray:
    """Hold plan that passes through timed poses.

    Frame 0 is `ref0` (the current reference, so an install never steps); frame k is at
    t_now + k/fps. `knots` = [(t_abs, pose36), ...] in time order; knots at or before t_now are
    dropped, so re-anchoring on every refresh closes the remaining gap. Piecewise linear, with a
    hemisphere-aligned nlerp on the quaternion. Past the last knot the plan holds the last pose.
    """
    ref0 = np.asarray(ref0, dtype=np.float64)
    pts = [(float(t_now), ref0)] + [(float(t), np.asarray(p, dtype=np.float64))
                                    for t, p in knots if float(t) > t_now]
    if len(pts) == 1:                       # every knot is in the past: hold the final pose
        last = np.asarray(knots[-1][1], dtype=np.float64) if knots else ref0
        return np.tile(last[None, :], (n_frames, 1))
    for i in range(1, len(pts)):            # q and -q are the same rotation: align each knot
        if float(np.dot(pts[i - 1][1][3:7], pts[i][1][3:7])) < 0.0:
            pts[i] = (pts[i][0], pts[i][1].copy())
            pts[i][1][3:7] *= -1.0
    ts = np.array([p[0] for p in pts])
    out = np.empty((n_frames, len(ref0)), dtype=np.float64)
    for k in range(n_frames):
        t = t_now + k / float(fps)
        j = int(np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(pts) - 2))
        a = float(np.clip((t - ts[j]) / max(ts[j + 1] - ts[j], 1e-9), 0.0, 1.0))
        out[k] = (1.0 - a) * pts[j][1] + a * pts[j + 1][1]
    out[:, 3:7] /= np.clip(np.linalg.norm(out[:, 3:7], axis=1, keepdims=True), 1e-9, None)
    return out


def motion_plan_to_qpos(abs_joints, abs_pos, abs_quat) -> np.ndarray:
    """Convert model output (joints, rel_pos, rel_quat_wxyz) to qpos format.

    qpos format: [pos(3), quat_xyzw(4), joints(29)] = 36 floats per row.
    Quaternions are normalized.
    """
    N = len(abs_joints)
    qpos_data = np.zeros((N, 36), dtype=np.float64)
    qpos_data[:, 0:3] = abs_pos
    quat_norm = abs_quat / (np.linalg.norm(abs_quat, axis=1, keepdims=True) + 1e-8)
    qpos_data[:, 3:7] = quat_norm[:, [1, 2, 3, 0]]  # wxyz -> xyzw
    qpos_data[:, 7:] = abs_joints
    return qpos_data


def generate_motion_plan(
    motion_model,
    rgb_tensor,
    prompt: str,
    device,
    *,
    prev_chunk=None,
    inference_delay: int = 4,
    beta: float = 1.0,
):
    """Single forward pass with optional RTC guidance.

    Args:
        prev_chunk: [T, 36] previous prediction (model format) for RTC inpainting.
                    None = standard prediction.
        inference_delay: d - frames already executed from prev_chunk
        beta: RTC guidance strength

    Returns:
        (joints [T,29], rel_pos [T,3], rel_quat [T,4]) in robot/body frame.
    """
    if prev_chunk is not None and motion_model.inpaint_prev_chunk:
        # Embedded-RTC generator (trained with inpaint_prev_chunk): hard-inpaint
        # the first prev_chunk_frames of the plan to the prefix instead of using
        # predict_rtc's guidance. inference_delay / beta do not apply to the
        # inpaint path. predict(prev_chunk=...) expects the
        # prefix in NORMALIZED model space, so slice the first P rows + normalize.
        P = int(motion_model.prev_chunk_frames)
        prefix = prev_chunk.float()[:P]
        if motion_model.normalizer is not None:
            prefix = motion_model.normalizer.normalize(prefix)
        with torch.no_grad():
            pred = motion_model.predict(rgb_tensor, prompt, prev_chunk=prefix).cpu()
    elif prev_chunk is not None:
        pred = motion_model.predict_rtc(
            rgb_tensor, prompt,
            prev_chunk=prev_chunk,
            inference_delay=inference_delay,
            beta=beta,
        ).cpu()
    else:
        with torch.no_grad():
            pred = motion_model.predict(rgb_tensor, prompt).cpu()

    if pred.dim() == 3:
        pred = pred.squeeze(0)
    joints = pred[:, :29].numpy()
    rel_pos = pred[:, 29:32].numpy()
    rel_quat = pred[:, 32:36].numpy()
    return joints, rel_pos, rel_quat


def tracker_qpos_to_model_chunk(tracker_plan: np.ndarray, horizon: int) -> torch.Tensor:
    """Convert tracker's remaining qpos plan into RTC conditioning input.

    Args:
        tracker_plan: [N, 36] qpos rows: pos(3), quat_xyzw(4), joints_deploy(29)
        horizon: T - target trajectory length. Pads (with last frame) or
                 truncates to T rows.

    Returns:
        [T, 36] float32 tensor in model format: joints(29), rel_pos(3),
        rel_quat_wxyz(4).
    """
    N = len(tracker_plan)
    model_fmt = np.zeros((N, 36), dtype=np.float64)
    model_fmt[:, :29] = tracker_plan[:, 7:36]    # joints
    model_fmt[:, 29:32] = tracker_plan[:, 0:3]   # rel_pos
    model_fmt[:, 32:36] = tracker_plan[:, [6, 3, 4, 5]]  # xyzw -> wxyz

    if N < horizon:
        pad = np.tile(model_fmt[-1:], (horizon - N, 1))
        model_fmt = np.concatenate([model_fmt, pad], axis=0)
    else:
        model_fmt = model_fmt[:horizon]

    return torch.tensor(model_fmt, dtype=torch.float32)
