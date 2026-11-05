#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Per-episode camera/sensor FX ("cam_artifacts") for FPV/OAK domain randomization."""

from __future__ import annotations

import math

import numpy as np

_MULT_KEYS = ("brightness", "contrast", "saturation", "gamma")

_SAT_LEVEL = 250.0
_SAT_MAX_FRAC = 0.01


def _has_effect(art: dict) -> bool:
    if not art:
        return False
    for k, v in art.items():
        if k in _MULT_KEYS:
            if abs(float(v) - 1.0) > 1e-3:
                return True
        elif abs(float(v)) > 1e-3:  # signed (e.g. color_tone) or magnitude
            return True
    return False


def camera_yaw_pitch_from_qpos(qpos) -> tuple:
    """Per-frame ego/OAK camera (yaw, pitch) arrays from a G1 qpos sequence."""
    q = np.asarray(qpos, dtype=float)
    w, x, y, z = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
    root_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    wyaw = q[:, 19] if q.shape[1] > 19 else np.zeros(len(q))
    wpitch = q[:, 21] if q.shape[1] > 21 else np.zeros(len(q))
    return root_yaw + wyaw, wpitch


def egomotion_image_flow(cam_yaw, cam_pitch, fx: float = 566.5, fy: float = 566.8) -> list:
    """Per-frame image-plane motion ``(magnitude_px, angle_rad)`` from the camera yaw/pitch deltas."""
    yaw = np.asarray(cam_yaw, dtype=float)
    pit = np.asarray(cam_pitch, dtype=float)

    def _wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    out = [(0.0, 0.0)]
    for t in range(1, len(yaw)):
        vx = fx * _wrap(yaw[t] - yaw[t - 1])
        vy = fy * _wrap(pit[t] - pit[t - 1])
        out.append((float(math.hypot(vx, vy)), float(math.atan2(vy, vx))))
    return out


def draw_cam_artifact_randoms(
    art: dict, rng: np.random.Generator, H: int, W: int, motion: tuple | None = None
) -> dict | None:
    """Consume ``rng`` in the same order ``apply_cam_artifacts`` would; returns the draws (None if no effect)."""
    if not _has_effect(art):
        return None
    mb = None
    ego_gain = float(art.get("motion_blur_gain", 0.0))
    if motion is not None and ego_gain > 0.0:
        if rng.random() < float(art.get("motion_blur_rand_dir_prob", 0.15)):
            blur_len = float(rng.uniform(0.10, 0.30)) * 40.0
            blur_ang = float(rng.uniform(0.0, 2.0 * math.pi))
        else:
            _mag_px, _ang = motion
            blur_len = float(_mag_px) * float(rng.uniform(0.0, ego_gain))
            blur_ang = float(_ang)
        mb = (blur_len, blur_ang)
    ns = float(art.get("noise", 0.0))
    noise = rng.normal(0.0, ns * 255.0, (H, W, 3)).astype(np.float32) if ns > 1e-3 else None
    sp = float(art.get("salt_pepper", 0.0))
    sp_u = rng.random((H, W)) if sp > 1e-3 else None
    return {"mb": mb, "noise": noise, "sp_u": sp_u}


def apply_cam_artifacts(
    frame: np.ndarray,
    art: dict,
    rng: np.random.Generator | None = None,
    motion: tuple | None = None,
    draws: dict | None = None,
) -> np.ndarray:
    """Return `frame` (H×W×3 uint8 RGB) with the `art` FX applied."""
    if not _has_effect(art):
        return frame
    try:
        import cv2
    except Exception:
        cv2 = None

    f = frame.astype(np.float32)
    H, W = f.shape[:2]

    b = float(art.get("brightness", 1.0))
    if abs(b - 1.0) > 1e-3:
        f *= b
    c = float(art.get("contrast", 1.0))
    if abs(c - 1.0) > 1e-3:
        f = (f - 127.5) * c + 127.5
    g = float(art.get("gamma", 1.0))
    if abs(g - 1.0) > 1e-3:
        f = 255.0 * np.clip(f / 255.0, 0.0, 1.0) ** g

    wb = float(art.get("color_tone", 0.0))
    if abs(wb) > 1e-3:
        f[..., 0] *= 1.0 + 0.28 * wb  # R
        f[..., 2] *= 1.0 - 0.28 * wb  # B

    s = float(art.get("saturation", 1.0))
    if abs(s - 1.0) > 1e-3 and cv2 is not None:
        hsv = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * s, 0, 255)
        f = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

    lum = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    hi = float(np.quantile(lum, 1.0 - _SAT_MAX_FRAC))
    if hi > _SAT_LEVEL:
        f *= _SAT_LEVEL / hi

    chroma = float(art.get("chroma", 0.0))
    if chroma > 1e-3:
        sh = max(1, int(round(chroma * 4.0)))
        f[..., 0] = np.roll(f[..., 0], sh, axis=1)  # R
        f[..., 2] = np.roll(f[..., 2], -sh, axis=1)  # B

    blur_len = 0.0  # kernel length in pixels
    blur_ang = 0.0  # radians
    ego_gain = float(art.get("motion_blur_gain", 0.0))  # max gain (0..1 shutter fraction)
    if draws is not None:
        if draws.get("mb") is not None:
            blur_len, blur_ang = draws["mb"]
    elif motion is not None and ego_gain > 0.0:
        if rng.random() < float(art.get("motion_blur_rand_dir_prob", 0.15)):
            blur_len = float(rng.uniform(0.10, 0.30)) * 40.0
            blur_ang = float(rng.uniform(0.0, 2.0 * math.pi))
        else:
            _mag_px, _ang = motion
            blur_len = float(_mag_px) * float(rng.uniform(0.0, ego_gain))
            blur_ang = float(_ang)
    if blur_len > 1.0 and cv2 is not None:
        L = max(3, min(81, int(round(blur_len)) | 1))  # odd kernel, cap 81px
        kern = np.zeros((L, L), np.float32)
        cx = cy = (L - 1) / 2.0
        dx, dy = math.cos(blur_ang), math.sin(blur_ang)
        for t in range(L):
            off = t - (L - 1) / 2.0
            xi = int(round(cx + off * dx))
            yi = int(round(cy + off * dy))
            if 0 <= xi < L and 0 <= yi < L:
                kern[yi, xi] = 1.0
        kern /= max(kern.sum(), 1.0)
        f = cv2.filter2D(f, -1, kern)
    df = float(art.get("defocus", 0.0))
    if df > 1e-3 and cv2 is not None:
        sigma = df * 3.0
        f = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma, sigmaY=sigma)

    ns = float(art.get("noise", 0.0))
    if ns > 1e-3:
        _noise = draws["noise"] if draws is not None else rng.normal(0.0, ns * 255.0, f.shape).astype(np.float32)
        if _noise is not None:
            f = f + _noise

    sp = float(art.get("salt_pepper", 0.0))
    if sp > 1e-3:
        u = draws["sp_u"] if draws is not None else rng.random((H, W))
        if u is not None:
            f[u < sp / 2.0] = 0.0  # pepper
            f[u > 1.0 - sp / 2.0] = 255.0  # salt

    pz = float(art.get("posterize", 0.0))
    if pz > 1e-3:
        lv = max(2.0, 255.0 * (1.0 - pz) + 4.0 * pz)
        f = np.floor((f / 255.0) * lv) / lv * 255.0

    sl = float(art.get("scanline", 0.0))
    if sl > 1e-3:
        f[1::2, :, :] *= 1.0 - 0.6 * sl

    vg = float(art.get("vignette", 0.0))
    if vg > 1e-3:
        yy, xx = np.mgrid[0:H, 0:W]
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
        mask = 1.0 - vg * np.clip(r / np.sqrt(2.0), 0.0, 1.0) ** 2
        f *= mask[..., None].astype(np.float32)

    return np.clip(f, 0, 255).astype(np.uint8)
