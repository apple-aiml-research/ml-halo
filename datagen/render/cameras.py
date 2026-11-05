#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""OAK helmet camera: mount and intrinsics (Kannala-Brandt fisheye -> Isaac / MuJoCo render configs)."""

from __future__ import annotations

import json
import math
import os

import numpy as np

# OV9782 sensor, 1280x800, 3.0 um pixels -> 3.84 x 2.40 mm.
OAK_CAM_HORIZONTAL_APERTURE = 3.84
# Mount in torso_link frame, composed from the THOR URDF chain torso -> head -> helmet -> oak1.
OAK_OFFSET_FROM_TORSO = (0.1040, 0.001, 0.478)
OAK_CAM_ROT = (0.5, -0.5, 0.5, -0.5)  # wxyz, ROS forward-facing
OAK_CAM_PITCH_DOWN_DEG = 25.0


def _derive_pinhole_render(data: dict) -> dict:
    """Wider (140 deg HFOV) pinhole render config for the MuJoCo pinhole + cv2.fisheye-warp path."""
    W_f, H_f = int(data["width"]), int(data["height"])
    h_aperture_mm = OAK_CAM_HORIZONTAL_APERTURE
    target_hfov_deg = 140.0
    render_w = int(os.environ.get("OAK_RENDER_WIDTH", "1600"))
    render_h = int(round(render_w * H_f / W_f))
    f_mm = (h_aperture_mm / 2.0) / math.tan(math.radians(target_hfov_deg / 2.0))
    px_size_mm = h_aperture_mm / render_w
    fx_render = f_mm / px_size_mm
    cx_off_mm = (data["cx"] - W_f / 2.0) * (h_aperture_mm / W_f)
    cy_off_mm = (data["cy"] - H_f / 2.0) * (h_aperture_mm / W_f)
    return {
        "width": render_w,
        "height": render_h,
        "fx_px": fx_render,
        "fy_px": fx_render,
        "cx_px": render_w / 2.0 + cx_off_mm / px_size_mm,
        "cy_px": render_h / 2.0 + cy_off_mm / px_size_mm,
    }


def _derive_fisheye_render(data: dict) -> dict:
    """Isaac fisheyePolynomial config: theta(r) = polyA + polyB r + ... + polyE r^4 fitted to the K-B model."""
    W = int(data["width"])
    H = int(data["height"])
    fx = float(data["fx"])
    fy = float(data["fy"])
    cx = float(data["cx"])
    cy = float(data["cy"])
    k1, k2, k3, k4 = [float(c) for c in data["distortion_coeffs"][:4]]

    # Sample both image diagonals (Isaac's set_matching_fisheye_polynomial_properties algorithm).
    X = np.concatenate([np.linspace(0, W, W), np.linspace(0, W, W)])
    Y = np.concatenate([np.linspace(0, H, W), np.linspace(H, 0, W)])
    pt_x_n = (X - cx) / fx
    pt_y_n = (Y - cy) / fy
    theta = np.arctan2(np.sqrt(pt_x_n * pt_x_n + pt_y_n * pt_y_n), 1.0)
    r_n = np.sqrt(pt_x_n * pt_x_n + pt_y_n * pt_y_n)
    theta_d = theta + k1 * theta**3 + k2 * theta**5 + k3 * theta**7 + k4 * theta**9
    safe_r = np.where(r_n > 1e-8, r_n, 1.0)
    cdist = theta_d * np.where(r_n > 1e-8, 1.0 / safe_r, 1.0)
    px = fx * (pt_x_n * cdist) + cx
    py = fy * (pt_y_n * cdist) + cy
    r_pixels = np.linalg.norm(np.array([px - cx, py - cy]), axis=0)
    coeffs = list(np.polyfit(r_pixels, theta, deg=4)[::-1])

    # Diagonal max FOV (full angle) + 5 deg margin.
    diag_r = float(np.linalg.norm([W - cx, H - cy]))
    thetas = np.linspace(0.01, np.pi / 2 - 0.01, 5000)
    rs_pix = (thetas + k1 * thetas**3 + k2 * thetas**5 + k3 * thetas**7 + k4 * thetas**9) * fx
    idx = int(np.searchsorted(rs_pix, diag_r))
    half_diag_fov = float(thetas[min(idx, len(thetas) - 1)])
    return {
        "width": W,
        "height": H,
        "cx_px": cx,
        "cy_px": cy,
        "max_fov_deg": float(np.degrees(2.0 * half_diag_fov)) + 5.0,
        "polyA": coeffs[0],
        "polyB": coeffs[1],
        "polyC": coeffs[2],
        "polyD": coeffs[3],
        "polyE": coeffs[4],
    }


def load_oak_intrinsics(path: str | None = None) -> dict:
    """OAK intrinsics from ``path`` (default render/oak_intrinsics.json) plus derived pinhole_render / fisheye_render."""
    path = path or os.path.join(os.path.dirname(__file__), "oak_intrinsics.json")
    with open(path) as f:
        data = json.load(f)
    data["pinhole_render"] = _derive_pinhole_render(data)
    data["fisheye_render"] = _derive_fisheye_render(data)
    return data
