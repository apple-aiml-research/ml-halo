#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""cv2.fisheye warp of the MuJoCo OAK pinhole render to the calibrated fisheye (Isaac renders fisheye natively)."""

from __future__ import annotations

import os
import subprocess

import numpy as np


def _build_fisheye_remap(intrinsics: dict) -> tuple[np.ndarray, np.ndarray]:
    """(map_x, map_y): for each fisheye pixel, its ray (cv2.fisheye.undistortPoints) projected into the pinhole render."""
    import cv2

    W_f, H_f = int(intrinsics["width"]), int(intrinsics["height"])
    K_fisheye = np.array(
        [[intrinsics["fx"], 0.0, intrinsics["cx"]], [0.0, intrinsics["fy"], intrinsics["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    D = np.array(intrinsics["distortion_coeffs"], dtype=np.float64).reshape(-1, 1)
    pr = intrinsics["pinhole_render"]
    grid_x, grid_y = np.meshgrid(np.arange(W_f, dtype=np.float32), np.arange(H_f, dtype=np.float32))
    fisheye_pixels = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 1, 2)
    normalized = cv2.fisheye.undistortPoints(fisheye_pixels, K_fisheye, D).reshape(H_f, W_f, 2)
    map_x = (normalized[..., 0] * pr["fx_px"] + pr["cx_px"]).astype(np.float32)
    map_y = (normalized[..., 1] * pr["fy_px"] + pr["cy_px"]).astype(np.float32)
    return map_x, map_y


def warp_pinhole_mp4_to_fisheye_mp4(src_mp4: str, output_path: str, intrinsics: dict, fps: int = 30) -> bool:
    """Warp every frame of the pinhole ``src_mp4`` to fisheye and encode ``output_path`` (libx264, crf 23)."""
    import cv2

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cap = cv2.VideoCapture(src_mp4)
    if not cap.isOpened():
        print(f"  [WARN] cannot open {src_mp4} for fisheye warp")
        return False
    map_x, map_y = _build_fisheye_remap(intrinsics)
    W_out, H_out = int(intrinsics["width"]), int(intrinsics["height"])
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W_out}x{H_out}", "-framerate", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        output_path,
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            warped = cv2.remap(
                img, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
            )
            proc.stdin.write(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).tobytes())
    finally:
        cap.release()
        proc.stdin.close()
        proc.wait()
    if proc.returncode != 0:
        err = proc.stderr.read().decode("utf-8", errors="ignore")[-300:]
        print(f"  [WARN] ffmpeg failed during fisheye warp: {err}")
        return False
    return True
