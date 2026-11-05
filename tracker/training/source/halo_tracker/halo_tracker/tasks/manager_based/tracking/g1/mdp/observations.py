#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Builds on BeyondMimic whole_body_tracking (Copyright (c) 2024, The Isaac Lab Project Developers, MIT License).

"""Observation terms: motion-reference / anchor-frame terms, the foot-relative height map, and the
depth-image pipeline with its sim-to-real noise model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

    from .commands import MotionCommand


def robot_body_pos_b(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """Robot tracked-body positions in the robot anchor frame, flattened (B, num_bodies*3)."""
    command: "MotionCommand" = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """Robot tracked-body orientations in the robot anchor frame as 6D rotation, flattened."""
    command: "MotionCommand" = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """Reference anchor position expressed in the robot anchor frame."""
    command: "MotionCommand" = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """Reference anchor orientation in the robot anchor frame as 6D rotation."""
    command: "MotionCommand" = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def tracking_command(
    env: "ManagerBasedRLEnv",
    command_name: str,
    command_type: str,
    flatten_horizon: bool,
    horizon: int | None = None,
) -> torch.Tensor:
    """Future-trajectory observation produced by ``MotionCommand.ref_motion``.

    ``command_type``:
        - ``"full"``: include keypoint world-frame position errors (privileged critic).
        - ``"partial"``: trajectory only (actor).
    """
    command: "MotionCommand" = env.command_manager.get_term(command_name)
    if command_type == "full":
        return command.ref_motion(include_keypoints=True, flatten_horizon=flatten_horizon, horizon=horizon)
    if command_type == "partial":
        return command.ref_motion(include_keypoints=False, flatten_horizon=flatten_horizon, horizon=horizon)
    raise ValueError(f"tracking_command: unknown command_type {command_type!r} (expected 'full' or 'partial')")


# ── Scene-task terms (height map / depth) ──


def _foot_height_offset(env, asset_cfg: SceneEntityCfg, foot_body_names: tuple[str, ...]) -> torch.Tensor:
    """Mean foot-link world z, ``(B, 1)``, the reference height of the foot-relative height map."""
    asset = env.scene[asset_cfg.name]
    body_idx, _ = asset.find_bodies(list(foot_body_names))
    return asset.data.body_pos_w[:, body_idx, 2].mean(dim=-1, keepdim=True)


def height_scan_2d(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    grid_size: tuple[int, int],
    asset_cfg: SceneEntityCfg,
    clip: float = 1.0,
    foot_body_names: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link"),
) -> torch.Tensor:
    """Foot-relative height scan ``h = foot_z - ray_hit_z``, clipped to ``[-clip, clip]``, as ``(B, 1, H, W)``.

    Relative to the feet (as in RPL), the map keeps the shape of the ground and objects when the torso drops
    (sitting, crouching). ``grid_size = (H, W)`` must match ``GridPatternCfg.size / resolution``, with the pattern's
    rays flattened H-major: ``ordering="yx"`` (x outer, y inner) for ``(H, W) = (n_x, n_y)``.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    flat = _foot_height_offset(env, asset_cfg, foot_body_names) - sensor.data.ray_hits_w[..., 2]
    h, w = grid_size
    return flat.view(-1, 1, h, w).clamp(-clip, clip)


# -- Depth camera (student perceptive input; DR parameters in CnnStudentDepthCfg) --


def depth_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    out_hw: tuple[int, int] = (48, 84),
    clip: tuple[float, float] = (0.2, 2.5),
    log_scale: bool = True,
    delay_range_steps: tuple[int, int] = (1, 3),
    noise_std_m: float = 0.0,
    stereo_noise_coeff: float = 0.0,
    dropout_prob: float = 0.0,
    scale_range: tuple[float, float] = (1.0, 1.0),
    bias_range_m: tuple[float, float] = (0.0, 0.0),
    flicker_std_m: float = 0.0,
    clutter_cfg: dict | None = None,
    rsm_mask_prob: float = 0.0,
    rsm_max_cols: int = 0,
    obs_decim: int = 1,
) -> torch.Tensor:
    """Depth image from a depth camera, noise-injected + downsampled + delayed → ``(B, 1, H_out, W_out)``.

    Matches the distillation student's perceptive head. Each stage is a no-op at its disabling default,
    applied in this order:

      1. Fetch ``distance_to_image_plane`` (Z-depth, what the D435 reports), ``(B, H, W, 1)``.
      2. Replace non-finite (missed-ray) values with ``far``.
      3. Synthetic clutter (``clutter_cfg``): per-env rectangles at random depths applied via
         ``torch.minimum`` (occlude-only), resampled at reset. Observation-space only — no physics.
      4. Range-dependent stereo noise (``stereo_noise_coeff``): σ(d) = coeff·d² (D435 stereo model).
      5. Temporal flicker (``flicker_std_m``): per-pixel per-step jitter; then constant-σ Gaussian noise
         (``noise_std_m``).
      6. Global scale + bias (``scale_range`` + ``bias_range_m``): per-env calibration residual,
         persistent within an episode.
      7. Pixel dropouts (``dropout_prob``): fraction of pixels set to ``far`` (D435 stereo-match fails).
      8. Clip to ``[z_near, z_far]`` and optional ``log1p``.
      9. ``(B, H, W, 1)`` → ``(B, 1, H, W)``, then area-downsample to ``out_hw``.
     10. Observation latency (``delay_range_steps``): return the frame from ``delay`` steps ago (per-env,
         resampled at reset).
     11. Random Side Masking (``rsm_mask_prob`` / ``rsm_max_cols``): occlude 1..``rsm_max_cols`` peripheral
         columns on a random edge, filled with far. Applied last, on the (possibly delayed) frame.

    ``obs_decim`` N > 1 processes a new frame only every N-th control step and returns the held frame in between
    (depth rate = control rate / N, the ``obs_decim`` of the deploy configs' depth group). The held steps never
    touch ``sensor.data``, so a lazy camera does not render on them.

    ``out_hw`` is ``(height, width)`` (PyTorch ``interpolate(size=...)`` convention). Persistent state
    (scale, bias, clutter rects, latency ring buffer) is stashed on ``env`` under sensor-namespaced keys.
    """
    sensor = env.scene.sensors[sensor_cfg.name]  # RPLDepthCamera
    hold_key = f"_depth_obs_hold__{sensor_cfg.name}"
    if obs_decim > 1 and int(env.common_step_counter) % obs_decim != 0:
        held = getattr(env, hold_key, None)
        if held is not None:
            # Remember resets that land on a held step so the next fresh frame still resamples the DR state.
            _depth_reset_mask(env, sensor_cfg.name, consume=False)
            return held
    reset_mask = _depth_reset_mask(env, sensor_cfg.name, consume=True)
    # Z-depth (perpendicular to the image plane), what the D435 reports.
    depth = sensor.data.output["distance_to_image_plane"]  # (B, H, W, 1)

    # 2. inf → far (rays that missed every mesh).
    far = float(clip[1])
    depth = torch.where(torch.isfinite(depth), depth, torch.full_like(depth, far))

    # Per-env DR state (scale + bias + clutter). Only allocated when one of those is active, otherwise
    # ``env`` never grows a state attribute.
    scale_active = scale_range != (1.0, 1.0)
    bias_active = bias_range_m != (0.0, 0.0)
    clutter_active = clutter_cfg is not None
    if scale_active or bias_active or clutter_active:
        dr_state = _depth_dr_state(
            env, sensor_cfg.name, depth.shape, depth.device, reset_mask,
            scale_range=scale_range, bias_range_m=bias_range_m,
            clutter_cfg=clutter_cfg,
        )
    else:
        dr_state = None

    # 3. Synthetic clutter (occluders modeled in image space).
    if clutter_active and dr_state is not None and dr_state["n_max"] > 0:
        depth = _inject_depth_clutter(depth, dr_state)

    # 4. Range-dependent stereo noise.
    if stereo_noise_coeff > 0.0:
        sigma = stereo_noise_coeff * depth.pow(2)
        depth = depth + sigma * torch.randn_like(depth)

    # 5. Temporal flicker.
    if flicker_std_m > 0.0:
        depth = depth + flicker_std_m * torch.randn_like(depth)
    if noise_std_m > 0.0:
        depth = depth + noise_std_m * torch.randn_like(depth)

    # 6. Per-env scale + bias (calibration residual).
    if dr_state is not None and (scale_active or bias_active):
        scale = dr_state["scale"].view(-1, 1, 1, 1)
        bias = dr_state["bias"].view(-1, 1, 1, 1)
        depth = depth * scale + bias

    # 7. Pixel dropouts — set to far (our "no data" convention, matching step 2) not 0, so the CNN
    # doesn't see spurious close pixels.
    if dropout_prob > 0.0:
        drop_mask = torch.rand_like(depth) < dropout_prob
        depth = torch.where(drop_mask, torch.full_like(depth, far), depth)

    # 8. Clip + optional log normalization.
    depth = depth.clamp(min=float(clip[0]), max=far)
    if log_scale:
        depth = torch.log1p(depth)

    # 9. (B, H, W, 1) → (B, 1, H, W) for Conv2d, then downsample.
    depth = depth.permute(0, 3, 1, 2)
    h_out, w_out = out_hw
    if depth.shape[-2:] != (h_out, w_out):
        depth = torch.nn.functional.interpolate(
            depth, size=(h_out, w_out), mode="area"
        )

    # 10. Observation latency via per-env ring buffer.
    if delay_range_steps is not None and delay_range_steps[1] > 0:
        depth = _apply_depth_delay(env, sensor_cfg.name, depth, delay_range_steps, reset_mask)

    # 11. Random Side Masking — applied last, on the (possibly delayed) frame the policy consumes, so it
    # models real-time lateral occlusion rather than a stale-buffer artifact.
    if rsm_mask_prob > 0.0 and rsm_max_cols > 0:
        depth = _apply_depth_rsm(depth, rsm_mask_prob, rsm_max_cols, far, log_scale)

    if obs_decim > 1:
        setattr(env, hold_key, depth)
    return depth


def _depth_reset_mask(env: ManagerBasedEnv, sensor_name: str, consume: bool) -> torch.Tensor:
    """Per-env "episode reset since the last processed depth frame" mask.

    Accumulates ``episode_length_buf == 0`` over every call, including the held steps of ``obs_decim`` > 1,
    and is cleared when a fresh frame consumes it. With ``obs_decim`` = 1 it equals ``episode_length_buf == 0``.
    """
    key = f"_depth_pending_reset__{sensor_name}"
    pending = getattr(env, key, None)
    if pending is None or pending.shape[0] != env.num_envs:
        pending = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    if hasattr(env, "episode_length_buf"):
        pending = pending | (env.episode_length_buf == 0)
    setattr(env, key, torch.zeros_like(pending) if consume else pending)
    return pending


def _depth_dr_state(
    env: ManagerBasedEnv,
    sensor_name: str,
    shape: tuple[int, int, int, int],
    device: torch.device,
    reset_mask: torch.Tensor,
    scale_range: tuple[float, float],
    bias_range_m: tuple[float, float],
    clutter_cfg: dict | None,
) -> dict:
    """Per-env depth DR state: scale, bias, and up to ``n_max`` clutter rects.

    Resampled on episode reset (``reset_mask``, see :func:`_depth_reset_mask`), same lifecycle as
    :func:`_apply_depth_delay`; non-reset envs keep their calibration/clutter across steps (matching real
    hardware, whose per-session residual is roughly constant).
    """
    B, H, W, _ = shape
    key = f"_depth_dr__{sensor_name}"
    state = getattr(env, key, None)
    n_max = int(clutter_cfg.get("n_range", (0, 0))[1]) if clutter_cfg else 0

    # Lazy-alloc or reshape if num_envs / clutter capacity changed.
    if (
        state is None
        or state["scale"].shape[0] != B
        or state.get("n_max", 0) != n_max
        or state.get("H") != H
        or state.get("W") != W
    ):
        state = {
            "scale": torch.ones(B, device=device),
            "bias": torch.zeros(B, device=device),
            "n_max": n_max,
            "H": H,
            "W": W,
        }
        if n_max > 0:
            state["clutter_rects"] = torch.zeros(B, n_max, 4, dtype=torch.long, device=device)
            state["clutter_depths"] = torch.zeros(B, n_max, device=device)
            state["clutter_valid"] = torch.zeros(B, n_max, dtype=torch.bool, device=device)
        _resample_depth_dr(
            state, torch.ones(B, dtype=torch.bool, device=device),
            scale_range=scale_range, bias_range_m=bias_range_m,
            clutter_cfg=clutter_cfg,
        )
        setattr(env, key, state)

    # Per-env resample on episode reset. Pass the bool reset mask straight to the resampler (writes only
    # masked envs via torch.where) to avoid a per-step GPU->CPU sync.
    _resample_depth_dr(
        state, reset_mask,
        scale_range=scale_range, bias_range_m=bias_range_m,
        clutter_cfg=clutter_cfg,
    )
    return state


def _resample_depth_dr(
    state: dict,
    mask: torch.Tensor,
    scale_range: tuple[float, float],
    bias_range_m: tuple[float, float],
    clutter_cfg: dict | None,
) -> None:
    """Resample ``state`` entries where ``mask`` (bool, ``(B,)``) is True, in-place.

    Sync-free: candidate values are drawn for ALL envs and kept only where ``mask`` is True via
    ``torch.where``, so there is no GPU->CPU sync on the per-step reset path (matters under DDP, where
    any rank stalling stalls all).
    """
    dev = state["scale"].device
    B = state["scale"].shape[0]
    H, W = state["H"], state["W"]

    if scale_range != (1.0, 1.0):
        new = torch.empty(B, device=dev).uniform_(*scale_range)
        state["scale"] = torch.where(mask, new, state["scale"])
    if bias_range_m != (0.0, 0.0):
        new = torch.empty(B, device=dev).uniform_(*bias_range_m)
        state["bias"] = torch.where(mask, new, state["bias"])

    n_max = state.get("n_max", 0)
    if clutter_cfg is None or n_max == 0:
        return

    n_range = tuple(clutter_cfg.get("n_range", (0, n_max)))
    d_range = tuple(clutter_cfg.get("depth_range_m", (1.0, 5.0)))
    s_range = tuple(clutter_cfg.get("size_frac_range", (0.08, 0.35)))
    mask_b1 = mask.view(B, 1)  # broadcast the per-env mask over the n_max rect slots

    # First ``n_per_env[i]`` rect slots are valid for env i; the rest are masked out.
    n_per_env = torch.randint(n_range[0], n_range[1] + 1, (B,), device=dev)
    arange_nmax = torch.arange(n_max, device=dev).unsqueeze(0)
    new_valid = arange_nmax < n_per_env.unsqueeze(1)
    state["clutter_valid"] = torch.where(mask_b1, new_valid, state["clutter_valid"])

    new_depths = torch.empty(B, n_max, device=dev).uniform_(*d_range)
    state["clutter_depths"] = torch.where(mask_b1, new_depths, state["clutter_depths"])

    # Pixel-integer sizes, then random top-left within bounds.
    h_pix = (torch.empty(B, n_max, device=dev).uniform_(*s_range) * H).long().clamp(min=1, max=H - 1)
    w_pix = (torch.empty(B, n_max, device=dev).uniform_(*s_range) * W).long().clamp(min=1, max=W - 1)
    max_top = (H - h_pix).clamp(min=1).float()
    max_left = (W - w_pix).clamp(min=1).float()
    top = (torch.rand(B, n_max, device=dev) * max_top).long()
    left = (torch.rand(B, n_max, device=dev) * max_left).long()
    rects = state["clutter_rects"]
    rects[:, :, 0] = torch.where(mask_b1, top, rects[:, :, 0])
    rects[:, :, 1] = torch.where(mask_b1, top + h_pix, rects[:, :, 1])
    rects[:, :, 2] = torch.where(mask_b1, left, rects[:, :, 2])
    rects[:, :, 3] = torch.where(mask_b1, left + w_pix, rects[:, :, 3])


def _inject_depth_clutter(depth: torch.Tensor, state: dict) -> torch.Tensor:
    """Occlude ``depth`` with per-env synthetic rectangles.

    Each rect (from ``state["clutter_rects"]`` / ``clutter_depths`` / ``clutter_valid``, all ``(B, n_max,
    ...)``) is overlaid via ``torch.minimum`` so it can only SHORTEN the measured depth, never lengthen it
    (a wall can't de-occlude something already closer). Rects beyond the final clip range clamp to far and
    vanish. Loops over ``n_max`` (typically ≤3), vectorized across B,H,W within each iteration.
    """
    B, H, W, _ = depth.shape
    dev = depth.device
    n_max = state["n_max"]
    rects = state["clutter_rects"]       # (B, n_max, 4) long — top, bot, left, right
    depths = state["clutter_depths"]     # (B, n_max)
    valid = state["clutter_valid"]       # (B, n_max) bool

    y = torch.arange(H, device=dev).view(1, H, 1)
    x = torch.arange(W, device=dev).view(1, 1, W)

    # inf = "no rectangle covers this pixel" → torch.minimum leaves existing geometry unchanged there.
    clutter_field = torch.full(
        (B, H, W), float("inf"), device=dev, dtype=depth.dtype
    )
    for n in range(n_max):
        top = rects[:, n, 0].view(B, 1, 1)
        bot = rects[:, n, 1].view(B, 1, 1)
        left = rects[:, n, 2].view(B, 1, 1)
        right = rects[:, n, 3].view(B, 1, 1)
        in_rect = (y >= top) & (y < bot) & (x >= left) & (x < right)
        apply = in_rect & valid[:, n].view(B, 1, 1)
        depth_n = depths[:, n].view(B, 1, 1).expand(B, H, W)
        clutter_field = torch.where(
            apply, torch.minimum(clutter_field, depth_n), clutter_field
        )

    return torch.minimum(depth, clutter_field.unsqueeze(-1))


def _apply_depth_delay(
    env: ManagerBasedEnv,
    sensor_name: str,
    frame: torch.Tensor,
    delay_range_steps: tuple[int, int],
    reset_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-env observation-latency ring buffer for :func:`depth_image`.

    Samples an integer per-env delay ∈ ``delay_range_steps`` at each episode reset (``reset_mask``) and
    returns the processed frame from ``delay`` frames ago (one frame per ``obs_decim`` control steps). The
    effective delay is clamped to the frames processed this episode, so a returned frame is from within the
    current episode (a reset that lands on a held step of obs_decim still returns the held frame for one step).
    The buffer holds ``dmax + 1`` frames at the output resolution.
    """
    B = frame.shape[0]
    dev = frame.device
    dmin, dmax = int(delay_range_steps[0]), int(delay_range_steps[1])
    buf_size = dmax + 1

    key = f"_depth_delay__{sensor_name}"
    state = getattr(env, key, None)
    if state is None or state["buf"].shape[1:] != frame.shape:
        state = {
            "buf": torch.zeros((buf_size,) + tuple(frame.shape), device=dev, dtype=frame.dtype),
            "delay": torch.randint(dmin, dmax + 1, (B,), device=dev),
            "frames": torch.zeros(B, dtype=torch.long, device=dev),  # frames written this episode
            "step": 0,  # Python int — avoids per-call GPU→CPU sync
        }
        setattr(env, key, state)

    # Resample per-env delay on reset. Sync-free: draw for every env, keep only where reset (torch.where).
    new_delay = torch.randint(dmin, dmax + 1, (B,), device=dev)
    state["delay"] = torch.where(reset_mask, new_delay, state["delay"])
    state["frames"] = torch.where(reset_mask, torch.zeros_like(state["frames"]), state["frames"])

    # Write current frame into the ring.
    write_slot = state["step"] % buf_size
    state["buf"][write_slot] = frame

    # Effective delay clamped to the frames written this episode (the current one is frame 0).
    eff_delay = torch.minimum(state["delay"], state["frames"])
    state["frames"] = state["frames"] + 1
    delayed_slot = (state["step"] - eff_delay) % buf_size  # (B,)

    idx = torch.arange(B, device=dev)
    delayed = state["buf"][delayed_slot, idx]  # (B, C, H, W)

    state["step"] += 1
    return delayed


def _apply_depth_rsm(
    frame: torch.Tensor,
    mask_prob: float,
    max_cols: int,
    far: float,
    log_scale: bool,
) -> torch.Tensor:
    """Random Side Masking for depth (per-env, per-step).

    With probability ``mask_prob`` each env masks 1..``max_cols`` columns on a random left/right edge,
    filled with the far-plane value (post-``log1p`` when ``log_scale``), forcing reliance on the central
    depth region. ``frame`` is ``(B, 1, H, W)``; resampled every step (per-frame augmentation).
    """
    B, _, H, W = frame.shape
    dev = frame.device
    n_mask = torch.where(
        torch.rand(B, device=dev) < mask_prob,
        torch.randint(1, max_cols + 1, (B,), device=dev),
        torch.zeros(B, dtype=torch.long, device=dev),
    )
    side_left = torch.rand(B, device=dev) < 0.5
    col = torch.arange(W, device=dev).view(1, 1, 1, W)
    n_e = n_mask.view(B, 1, 1, 1)
    left = (col < n_e) & side_left.view(B, 1, 1, 1)
    right = (col >= (W - n_e)) & (~side_left).view(B, 1, 1, 1) & (n_e > 0)
    col_mask = (left | right).expand(B, 1, H, W)
    # Far-plane fill in the SAME representation as ``frame`` (post-log if applied). ``frame.new_tensor``
    # keeps device/dtype.
    fill_val = float(torch.log1p(frame.new_tensor(float(far)))) if log_scale else float(far)
    return torch.where(col_mask, torch.full_like(frame, fill_val), frame)


