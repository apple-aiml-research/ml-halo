#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Generate one Kimodo segment between two world-frame keyframes."""

import os

import numpy as np
import torch

from kimodo.tools import seed_everything
from motion_gen.kimodo.constraints import (
    build_constraint_object,
    build_root2d_constraint,
    compute_canon_angle,
    canonicalize,
    undo_canon_g1_qpos,
)

FPS = 30


def world_qpos_to_kf(qpos36, converter):
    """One world [36] qpos -> (joints[J,3], rotmats[J,3,3]) in WORLD kimodo frame."""
    seq = np.tile(qpos36, (10, 1))
    kd = converter.qpos_to_motion_dict(
        torch.tensor(seq).unsqueeze(0).float(), source_fps=float(FPS), root_quat_w_first=True
    )
    j = kd["posed_joints"][0].detach().cpu().numpy().astype(np.float32)
    r = kd["global_rot_mats"][0].detach().cpu().numpy().astype(np.float32)
    return j, r


def _canonical_endpoints(kf_start, kf_end, conv):
    """Both keyframes in kimodo's canonical frame of the START pose."""
    j0w, r0w = world_qpos_to_kf(kf_start, conv)
    j1w, r1w = world_qpos_to_kf(kf_end, conv)
    canon = compute_canon_angle(j0w, use_g1=True)
    j0c, r0c = canonicalize(j0w, r0w, canon)
    j1c, r1c = canonicalize(j1w, r1w, canon)
    origin = j0c[0][[0, 2]].copy()
    j0c[:, 0] -= origin[0]
    j0c[:, 2] -= origin[1]
    j1c[:, 0] -= origin[0]
    j1c[:, 2] -= origin[1]
    return j0c, r0c, j1c, r1c, canon, origin


def _sample_world(model, conv, prompt, num_frames, clst, seed, steps, cfg, num_samples, origin, canon):
    """Run kimodo on the constraint list and return world-frame qpos."""
    B = int(num_samples)
    micro = int(os.environ.get("KIMODO_MICRO_BATCH", "256"))
    qcs = []
    for s0 in range(0, B, micro):
        b = min(micro, B - s0)
        seed_everything(seed + s0)  # distinct noise per chunk
        out = model(
            prompts=[prompt] * b,
            num_frames=[num_frames] * b,
            constraint_lst=clst,
            num_denoising_steps=steps,
            cfg_weight=list(cfg),
            return_numpy=True,
            post_processing=False,
        )
        qc = np.asarray(conv.dict_to_qpos(out, device="cpu", root_quat_w_first=True, numpy=True))
        qcs.append(qc[None] if qc.ndim == 2 else qc)
    qc = np.concatenate(qcs, axis=0)  # [B,T,36]
    world = np.stack([undo_canon_g1_qpos(qc[i].astype(np.float64), origin, canon) for i in range(qc.shape[0])]).astype(
        np.float32
    )
    return world if B > 1 else world[0]


def gen_segment(
    model,
    skel,
    conv,
    prompt,
    num_frames,
    kf_start,
    kf_end,
    seed,
    dev,
    steps,
    cfg=(2.0, 2.0),
    num_samples=1,
    turn_mode="fullbody",
):
    """One segment between two WORLD keyframes ([36] qpos each)."""
    j0c, r0c, j1c, r1c, canon, origin = _canonical_endpoints(kf_start, kf_end, conv)
    if turn_mode in ("heading", "hybrid", "plateau"):
        r_hip, l_hip = skel.hip_joint_idx

        def _canon_heading(jc):
            d = jc[r_hip] - jc[l_hip]
            return float(np.arctan2(d[2], -d[0]))

        h0, h1 = _canon_heading(j0c), _canon_heading(j1c)
        dh = (h1 - h0 + np.pi) % (2 * np.pi) - np.pi  # shortest arc
        hs = h0 + np.linspace(0.0, 1.0, num_frames) * dh  # monotonic sweep
    if turn_mode == "plateau":
        end_xz = j1c[0][[0, 2]].astype(np.float32)  # canonical start->end ground offset
        if np.abs(end_xz).max() < 1e-6:
            end_xz[:] = 0.0  # snap float noise: the in-place pivot is an
        ramp = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)[:, None] * end_xz[None, :]
        root_only = build_root2d_constraint(np.arange(num_frames), ramp, None, skel).to(device=dev)
        span = num_frames - 5  # pre-plateau frames [0 .. N-6]
        n_sparse = max(int(round(num_frames / 8.0)), 2)
        sparse = np.unique(np.linspace(0, span - 1, n_sparse).round().astype(int))
        plat = np.arange(num_frames - 5, num_frames - 1)  # 4 frames; kf owns N-1
        hidx = np.concatenate([sparse, plat])
        hval = np.concatenate([hs[sparse], np.full(len(plat), h1)])
        heading = np.stack([np.cos(hval), np.sin(hval)], axis=-1).astype(np.float32)
        hc = build_root2d_constraint(hidx, ramp[hidx], heading, skel).to(device=dev)
        _kfs = [(num_frames - 1, j1c, r1c)]
        if os.environ.get("TURN_PIN_START") == "1":
            _kfs = [(0, j0c, r0c)] + _kfs
        fb = build_constraint_object(_kfs, skel).to(device=dev)
        clst = [root_only, hc, fb]
    elif turn_mode in ("heading", "hybrid"):
        nf_r2 = num_frames - 1 if turn_mode == "hybrid" else num_frames
        heading = np.stack([np.cos(hs[:nf_r2]), np.sin(hs[:nf_r2])], axis=-1).astype(np.float32)
        root2d = np.zeros((nf_r2, 2), dtype=np.float32)  # hold at canon start
        c = build_root2d_constraint(np.arange(nf_r2), root2d, heading, skel).to(device=dev)
        if turn_mode == "hybrid":
            fb = build_constraint_object([(num_frames - 1, j1c, r1c)], skel).to(device=dev)
            clst = [c, fb]
        else:
            clst = [c]
    else:
        kfs = [(0, j0c, r0c), (num_frames - 1, j1c, r1c)]
        c = build_constraint_object(kfs, skel).to(device=dev)
        clst = [c]
    return _sample_world(model, conv, prompt, num_frames, clst, seed, steps, cfg, num_samples, origin, canon)
