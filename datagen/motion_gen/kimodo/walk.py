#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Kimodo G1 walk segment of the sit motion."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np


def _g1_standing_keyframe(
    skel, model=None, device: str | None = None, cache_path: str = "~/.cache/halo/g1_standing_keyframe.npz"
) -> tuple[np.ndarray, np.ndarray]:
    """Return (joints_yup [J,3], rotmats_yup [J,3,3]) for the G1's canonical"""
    import torch

    cache = Path(os.path.expanduser(cache_path))
    if cache.exists():
        d = np.load(str(cache))
        return d["joints"], d["rotmats"]

    if model is None:
        from kimodo.model import load_model as kimodo_load_model

        device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[standing_kf] loading Kimodo (cache miss + no model passed)…")
        model = kimodo_load_model("Kimodo-G1-RP-v1", device=device)

    print(f"[standing_kf] cache miss at {cache}; generating Kimodo standing pose…")
    out = model(
        prompts=["a person stands still"],
        num_frames=30,
        constraint_lst=[],
        num_denoising_steps=200,
        num_samples=1,
        cfg_weight=[2.0, 2.0],
        return_numpy=False,
        post_processing=False,
    )
    joints = out["posed_joints"][0, 0].detach().cpu().numpy().astype(np.float32)
    rotmats = out["global_rot_mats"][0, 0].detach().cpu().numpy().astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cache), joints=joints, rotmats=rotmats)
    print(f"[standing_kf] saved {cache} (J={joints.shape[0]})")
    return joints, rotmats


_TRAJ_N_WAYPOINTS = 20
_TRAJ_CONSTRAIN_FRAC = 0.85  # constrain first 85%; last 15% settles to standing


def walk_qpos_from_traj(
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    spawn_yaw: float,
    g1_model,
    g1_skel,
    standing_kf,
    end_yaw: float | None = None,
    num_denoising_steps: int = 200,
    cfg_weight: float = 2.0,
    prompt: str = "a person walks toward and stops",
    device: str = "cuda:0",
    num_samples: int = 1,
) -> np.ndarray:
    """[T,36] scene-frame qpos following the unicycle root path (x,y,theta)."""
    import torch
    from kimodo.constraints import Root2DConstraintSet, FullBodyConstraintSet
    from kimodo.exports.mujoco import MujocoQposConverter
    from scipy.spatial.transform import Rotation as R_scipy

    T = len(x)
    num_frames = T
    spawn = np.array([x[0], y[0]], dtype=float)
    if end_yaw is None:
        end_yaw = float(theta[-1])

    fidx = np.linspace(0, int(_TRAJ_CONSTRAIN_FRAC * (T - 1)), _TRAJ_N_WAYPOINTS).astype(int)
    wp_world = np.stack([x[fidx], y[fidx]], axis=1)
    cos_y, sin_y = math.cos(-spawn_yaw), math.sin(-spawn_yaw)
    R_to_local = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    wp_local = (R_to_local @ (wp_world - spawn).T).T
    smooth = np.stack([wp_local[:, 1], wp_local[:, 0]], axis=1)  # (y,x) swap
    kyaw = theta[fidx] - spawn_yaw
    grh = np.stack([np.cos(kyaw), np.sin(kyaw)], axis=1)
    end_local = R_to_local @ (np.array([x[-1], y[-1]]) - spawn)

    fi = torch.tensor(fidx, dtype=torch.long)
    sr = torch.tensor(smooth, dtype=torch.float32)
    gh = torch.tensor(grh, dtype=torch.float32)
    if int(fi[0]) == 0:  # FullBody owns frame 0
        fi, sr, gh = fi[1:].clone(), sr[1:].clone(), gh[1:].clone()
    root2d = Root2DConstraintSet(skeleton=g1_skel, frame_indices=fi, smooth_root_2d=sr, global_root_heading=gh).to(
        device=device
    )

    j_kf, r_kf = standing_kf
    j0, r0 = j_kf.copy(), r_kf.copy()
    jT, rT = j_kf.copy(), r_kf.copy()
    yd = (end_yaw - spawn_yaw + math.pi) % (2 * math.pi) - math.pi
    cyd, syd = math.cos(yd), math.sin(yd)
    R_yup = np.array([[cyd, 0, syd], [0, 1, 0], [-syd, 0, cyd]], dtype=np.float32)
    jT = (R_yup @ jT.T).T
    rT = R_yup @ rT
    jT[:, 0] += float(end_local[1])
    jT[:, 2] += float(end_local[0])
    fullbody = FullBodyConstraintSet(
        skeleton=g1_skel,
        frame_indices=torch.tensor([0, num_frames - 1]),
        global_joints_positions=torch.tensor(np.stack([j0, jT]), dtype=torch.float32),
        global_joints_rots=torch.tensor(np.stack([r0, rT]), dtype=torch.float32),
    ).to(device=device)

    B = max(int(num_samples), 1)
    out = g1_model(
        prompts=[prompt] * B,
        num_frames=[num_frames] * B,
        constraint_lst=[root2d, fullbody],
        num_denoising_steps=num_denoising_steps,
        num_samples=B,
        cfg_weight=[cfg_weight, cfg_weight],
        return_numpy=False,
        post_processing=False,
    )

    conv = MujocoQposConverter(g1_skel)
    qpos = conv.dict_to_qpos(out, root_quat_w_first=True, numpy=True)
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim == 2:  # [T,36] -> [1,T,36]
        qpos = qpos[None]
    cw, sw = math.cos(spawn_yaw), math.sin(spawn_yaw)
    R_to_world = np.array([[cw, -sw], [sw, cw]], dtype=np.float32)
    Rz = R_scipy.from_euler("z", spawn_yaw)
    for b in range(len(qpos)):
        qpos[b, :, :2] = (R_to_world @ qpos[b, :, :2].T).T + spawn.astype(np.float32)
        for t in range(qpos.shape[1]):
            qw, qx, qy, qz = qpos[b, t, 3:7]
            nq = (Rz * R_scipy.from_quat([qx, qy, qz, qw])).as_quat()
            qpos[b, t, 3:7] = [nq[3], nq[0], nq[1], nq[2]]
    qpos = qpos.astype(np.float64)
    return qpos if int(num_samples) > 1 else qpos[0]
