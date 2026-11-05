#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Quaternion helpers ([w, x, y, z], Hamilton product), batched over leading axes.

Kept free of torch/torchvision imports so the dataloader workers that build the
motion targets (data/traj_extract.py) stay light.
"""

import numpy as np


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of unit quaternion(s). [..., 4] -> [..., 4], sign-flip x,y,z."""
    q = np.asarray(q, dtype=np.float32)
    conj = q.copy()
    conj[..., 1:] *= -1
    return conj


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 * q2 over broadcastable [..., 4] arrays."""
    q1 = np.asarray(q1, dtype=np.float32)
    q2 = np.asarray(q2, dtype=np.float32)
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) v by quaternion(s) q: v' = q * [0, v] * conj(q).

    Args:
        q: [..., 4] unit quaternion [w, x, y, z]
        v: [..., 3] vector (broadcastable against q)

    Returns:
        [..., 3] rotated vectors
    """
    q = np.asarray(q, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    shape = np.broadcast_shapes(q.shape[:-1], v.shape[:-1])
    v_quat = np.zeros(shape + (4,), dtype=np.float32)
    v_quat[..., 1:] = v
    rotated = quat_multiply(quat_multiply(q, v_quat), quat_conjugate(q))
    return rotated[..., 1:]


def yaw_of(q: np.ndarray) -> np.ndarray:
    """Yaw (rotation about +z) of a wxyz quaternion. Mirrors the C++
    make_frame_gauge yaw extraction in deploy motion_loader.cpp."""
    q = np.asarray(q, dtype=np.float32)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_quat(yaw: np.ndarray) -> np.ndarray:
    """Quaternion (wxyz) for a pure +z rotation by `yaw`. The yaw-only gauge:
    yaw_quat(-yaw_of(q0)) removes ONLY yaw (keeps pitch/roll), matching the
    yaw-only gauge deploy write_gauged_pose uses (qz_inv)."""
    yaw = np.asarray(yaw, dtype=np.float32)
    h = 0.5 * yaw
    zeros = np.zeros_like(h)
    return np.stack([np.cos(h), zeros, zeros, np.sin(h)], axis=-1).astype(np.float32)
