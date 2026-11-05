#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
Trajectory extraction for the motion dataset.

Builds the [T, 36] motion target for a sample: the whole window is pulled from
the episode's columns with a single fancy-index and the quaternion math is
batched over the time axis.

36-D layout per frame:
    [0:29]  absolute joint positions
    [29:32] pelvis position relative to t0, de-yawed (YAW-ONLY gauge)
            = quat_rotate_vector(qz_inv, pos[t] - pos[t0]),  qz_inv = Rz(-yaw0)
    [32:36] pelvis quaternion with yaw removed, pitch/roll kept = qz_inv * q_t

The gauge is YAW-ONLY (not full SE(3)) to match deploy's write_gauged_pose, so
frame 0 keeps the motion's real pitch instead of collapsing to identity. A
normalizer fitted under a full-SE(3) gauge is therefore stale and must be re-fit.

Quaternions are [w, x, y, z] (Hamilton product); see data/quaternion.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .quaternion import quat_multiply, quat_rotate_vector, yaw_of, yaw_quat

JOINT_KEY = "observation.joint_pos"
BODY_POS_KEY = "observation.body_pos"
BODY_QUAT_KEY = "observation.body_quat"


# ── Column access ──────────────────────────────────────────────────────────


def _gather(column: Any, idx: np.ndarray) -> np.ndarray:
    """Rows ``idx`` of a dense column as float32."""
    return column[idx].astype(np.float32, copy=False)


# ── Extraction ─────────────────────────────────────────────────────────────


def extract_trajectory_vectorized(
    data: Any,
    start_frame: int,
    trajectory_length: int,
) -> torch.Tensor:
    """The [T, 36] motion target window starting at ``start_frame``.

    Args:
        data: dict of column -> float32 array (from ``MotionDataset._read_parquet``).
            Needs keys "observation.joint_pos" [L, 29],
            "observation.body_pos" [L, B>=1, 3], "observation.body_quat"
            [L, B>=1, 4]. Body index 0 is the pelvis.
        start_frame: t0, the frame the pelvis pose is made relative to.
        trajectory_length: T, window length. Frames past the episode end are
            padded by clamping the frame index to ep_len - 1.

    Returns:
        torch.FloatTensor [T, 36]
    """
    joint_col = data[JOINT_KEY]
    ep_len = len(joint_col)

    # Frame indices for the window, clamped to the last frame (padding).
    idx = np.arange(start_frame, start_frame + trajectory_length)
    np.clip(idx, None, ep_len - 1, out=idx)

    joint_pos = _gather(joint_col, idx)                      # [T, 29]
    pelvis_pos = _gather(data[BODY_POS_KEY], idx)[:, 0]      # [T, 3]
    pelvis_quat = _gather(data[BODY_QUAT_KEY], idx)[:, 0]    # [T, 4]

    # t0 reference frame.
    t0 = np.asarray([start_frame])
    pelvis_pos_t0 = _gather(data[BODY_POS_KEY], t0)[0, 0]    # [3]
    pelvis_quat_t0 = _gather(data[BODY_QUAT_KEY], t0)[0, 0]  # [4]
    # YAW-ONLY gauge (matches deploy write_gauged_pose): remove ONLY yaw about +z,
    # KEEP pitch/roll. So frame 0's quat carries the motion's real pitch (~0 for
    # walk, ~40deg for crawl) instead of collapsing to identity — the generator
    # learns to author the pitch, and yaw-only deploy install preserves it.
    qz_inv = yaw_quat(-yaw_of(pelvis_quat_t0))

    # [29:32] de-yawed displacement: Rz(-yaw0) @ (pos[t] - pos[t0]), batched over T.
    pelvis_pos_rel = quat_rotate_vector(
        qz_inv, pelvis_pos - pelvis_pos_t0
    )                                                        # [T, 3]

    # [32:36] q_rel = qz_inv * q_t (pitch/roll preserved), batched over T.
    pelvis_quat_rel = quat_multiply(qz_inv, pelvis_quat)      # [T, 4]

    trajectory = np.concatenate(
        [joint_pos, pelvis_pos_rel, pelvis_quat_rel], axis=-1
    )                                                        # [T, 36]
    return torch.from_numpy(np.ascontiguousarray(trajectory, dtype=np.float32))


def extract_sample_trajectory_vectorized(
    data: Any,
    start_frame: int,
    cold_k: int,
    trajectory_length: int,
    prev_chunk_frames: int,
) -> torch.Tensor:
    """The [T, 36] motion target for a sample, including cold-start samples.

    For normal samples (``cold_k < 0``) this is the window at ``start_frame``.
    For cold-start samples (``cold_k >= 0``, ``start_frame == 0``) the first
    ``prev_chunk_frames - cold_k`` frames hold frame 0 and the future plan
    follows, all relative to frame 0.

    Returns:
        torch.FloatTensor [T, 36]
    """
    if cold_k < 0:
        return extract_trajectory_vectorized(data, start_frame, trajectory_length)

    base = extract_trajectory_vectorized(data, 0, trajectory_length)  # [T, 36]
    shift = prev_chunk_frames - cold_k   # number of held frames
    idx = torch.clamp(
        torch.arange(trajectory_length, dtype=torch.long) - shift, min=0
    )
    return base[idx]
