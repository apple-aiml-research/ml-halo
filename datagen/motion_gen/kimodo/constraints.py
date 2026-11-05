#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Kimodo constraint builders and canonical-frame helpers used by the sit segments (segments.py)."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R_scipy


def get_facing_direction(joints, use_g1=False):
    hip_vec = joints[1] - joints[8] if use_g1 else joints[22] - joints[26]
    facing = np.array([-hip_vec[2], 0.0, hip_vec[0]])
    n = np.linalg.norm(facing)
    return facing / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])


def compute_canon_angle(joints, use_g1=False):
    facing = get_facing_direction(joints, use_g1)
    return -np.degrees(np.arctan2(facing[0], facing[2]))


def canonicalize(j, r, angle):
    Ry = R_scipy.from_euler("y", np.radians(angle)).as_matrix().astype(np.float32)
    return (Ry @ j.T).T, Ry @ r


def build_constraint_object(frame_data_list, skeleton):
    """Build FullBodyConstraintSet from list of (frame_idx, joints, rotmats)."""
    from kimodo.constraints import FullBodyConstraintSet
    frames = [d[0] for d in frame_data_list]
    joints = np.stack([d[1] for d in frame_data_list])
    rotmats = np.stack([d[2] for d in frame_data_list])
    c = FullBodyConstraintSet(
        skeleton=skeleton,
        frame_indices=torch.tensor(frames, dtype=torch.long),
        global_joints_positions=torch.tensor(joints, dtype=torch.float32),
        global_joints_rots=torch.tensor(rotmats, dtype=torch.float32),
    )
    return c


def build_root2d_constraint(frame_indices, smooth_root_2d, global_root_heading, skeleton):
    """Build a Root2DConstraintSet: constrain the root ground-plane (x, z)
    trajectory + (optionally) the global heading (cos, sin) per frame, leaving
    the body pose free. Used for the heading-constrained in-place turn
    (``TURN_CONSTRAINT=heading``): holding root (x, z) fixed and sweeping the heading
    turns the robot in place instead of hopping to satisfy two full-body endpoint
    keyframes. All tensors are indexed by ``frame_indices``.

    ``global_root_heading=None`` constrains only root (x, z) and leaves the heading
    free on those frames. ``Root2DConstraintSet`` ties root2d and heading to the
    same ``frame_indices``, so a dense root hold + a sparse heading schedule needs
    two instances (see ``turn_mode="plateau"`` in ``motion_gen/kimodo/segments.py``).
    """
    from kimodo.constraints import Root2DConstraintSet
    return Root2DConstraintSet(
        skeleton=skeleton,
        frame_indices=torch.as_tensor(np.asarray(frame_indices), dtype=torch.long),
        smooth_root_2d=torch.as_tensor(np.asarray(smooth_root_2d), dtype=torch.float32),
        global_root_heading=(
            None if global_root_heading is None
            else torch.as_tensor(np.asarray(global_root_heading), dtype=torch.float32)
        ),
    )


def undo_canon_g1_qpos(qp, origin, canon_angle):
    """Undo canonicalization on G1 qpos [T, 36]."""
    qp = qp.copy()
    qp[:, 0] += origin[1]
    qp[:, 1] += origin[0]
    R_undo = R_scipy.from_euler("z", -canon_angle, degrees=True)
    for t in range(len(qp)):
        qp[t, :3] = R_undo.apply(qp[t, :3])
        q = R_undo * R_scipy.from_quat([qp[t, 4], qp[t, 5], qp[t, 6], qp[t, 3]])
        qx = q.as_quat()
        qp[t, 3:7] = [qx[3], qx[0], qx[1], qx[2]]
    return qp
