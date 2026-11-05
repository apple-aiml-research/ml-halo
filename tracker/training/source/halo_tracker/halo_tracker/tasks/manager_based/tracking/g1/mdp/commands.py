#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Builds on BeyondMimic whole_body_tracking (Copyright (c) 2024, The Isaac Lab Project Developers, MIT License).

"""Motion command generators for motion tracking.

* ``MotionCommand``: owns the reference motion (NPZ clips), per-env frame state, adaptive
  failure-weighted start-frame sampling and the reset-to-reference logic.
* ``MotionCommandObjectTerrain``: scene-aware variant for multi-asset scenes. Each env is bound to
  one spawned terrain / object variant; it only plays clips compatible with that variant and parks
  unused objects in a below-floor "graveyard".

Motion NPZs are read from local folders (layout: tracker/training/README.md#motion-data).
"""

from __future__ import annotations

import math
import os
import glob
import numpy as np
import torch
from collections import Counter
from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from .. import ASSETS_DIR

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import RigidObject


# ── Asset catalog: motion-filename → asset mapping (single source of truth) ──
# To add a family: drop its URDF under <repo>/assets/objects/, add the keyword/prefix
# here, and add the scene-entity field in motion_tracking_scene_env_cfg.py.

# Filename substring → canonical object_key (must match the scene-entity attr the
# scene cfg registers). ``MotionCommandObjectTerrain`` iterates this at init to find
# graveyard-dispatch objects. Checked via ``substring in filename``.
#   "box" → push_v2_*.npz (push-box clips) → objects/pushbox/pushbox.urdf
OBJECT_KEYS_FROM_FILENAME: dict[str, list[str]] = {
    "box": ["push_v2"],
}

# Filename prefix → terrain category; a category's URDF is ``<repo>/assets/objects/<category>/<category>.urdf``.
#   "labsofa" → labsofa_*.npz (sit-on-sofa clips) → objects/labsofa/labsofa.urdf
TERRAIN_PREFIX_CATEGORIES: dict[str, str] = {
    "labsofa_": "labsofa",
}


def get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    """Resolve body_names to indices within command.cfg.body_names (the tracked body list).

    Both motion-side and robot-side body arrays are already filtered to cfg.body_names order
    (MotionLoader slices the NPZ bodies at load; MotionCommand.body_indexes the robot's).
    So indices here are into the tracked-body dimension (0..len(cfg.body_names)-1).
    """
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def _require_clips(motion_files: str | list[str] | None) -> list[str]:
    """The clip list as a list; fails clearly when no motion data was configured."""
    files = [motion_files] if isinstance(motion_files, str) else list(motion_files or [])
    if not files or any(f is None for f in files):
        raise ValueError("No motion clips: pass --motion_data_folder DIR [DIR ...] (or --motion_data FILE [FILE ...]).")
    return files


def _require_motion_folders(motion_folder: str | list[str] | None) -> None:
    """Fail early, with a pointer to the data docs, when a configured motion folder has no NPZ clips."""
    if not motion_folder:
        return
    folders = [motion_folder] if isinstance(motion_folder, str) else list(motion_folder)
    for f in folders:
        if not (Path(f).is_dir() and any(Path(f).rglob("*.npz"))):
            raise FileNotFoundError(
                f"No motion NPZ files under {f!r}. Convert your clips first (tracker/data/README.md) and pass the "
                "folder(s) with --motion_data_folder."
            )


def _select_motion_files(
    folder: str | list[str],
    *,
    recursive: bool,
    max_per_subfolder: int | None = None,
) -> list[str]:
    """Glob ``*.npz`` from ``folder``, cap per-subfolder, sort ``_original`` first.

    Accepts a single path or a list of paths (each root globbed independently,
    results concatenated). Files are grouped by immediate parent dir and each
    LEAF subfolder capped independently; ``_original`` (retargeted-from-mocap)
    variants sort first. The final list is re-sorted lexicographically for
    deterministic, reproducible indexing.

    Args:
        folder: Root directory or list of root directories to glob from.
        recursive: If ``True``, walk every subtree (``**/*.npz``); else
            only immediate children.
        max_per_subfolder: Upper bound on NPZs kept per leaf subfolder.
            ``None`` (default) means unlimited.
    """
    folders = [folder] if isinstance(folder, str) else list(folder)
    paths: list[str] = []
    for f in folders:
        if not f:
            continue
        pattern = os.path.join(f, "**", "*.npz") if recursive else os.path.join(f, "*.npz")
        paths.extend(glob.glob(pattern, recursive=recursive))
    if not paths:
        return []

    def _rank(p: str) -> tuple[int, str]:
        base = os.path.basename(p)
        return (0 if "_original" in base else 1, base)

    groups: dict[str, list[str]] = {}
    for p in paths:
        groups.setdefault(os.path.dirname(p), []).append(p)

    selected: list[str] = []
    for subdir in sorted(groups):
        files = sorted(groups[subdir], key=_rank)
        if max_per_subfolder is not None and len(files) > max_per_subfolder:
            keep = files[:max_per_subfolder]
            n_orig_kept = sum(1 for p in keep if "_original" in os.path.basename(p))
            print(
                f"[MotionCommand] {subdir}: cap {len(files)} → {len(keep)} "
                f"({n_orig_kept} _original prioritized)"
            )
            files = keep
        selected.extend(files)

    return sorted(selected)


def _resolve_motion_files(cfg: MotionCommandCfg) -> str | list[str] | None:
    """The clips of ``cfg.motion_folder`` (globbed recursively, capped per leaf folder), else ``cfg.motion_file``."""
    if not cfg.motion_folder:
        return cfg.motion_file
    _require_motion_folders(cfg.motion_folder)
    motion_files = _select_motion_files(
        cfg.motion_folder, recursive=True, max_per_subfolder=cfg.max_motions_per_subfolder
    )
    assert motion_files, f"No .npz files under {cfg.motion_folder}."
    print(f"[MotionCommand] Loaded {len(motion_files)} NPZ files from {cfg.motion_folder}")
    return motion_files


class MotionLoader:
    """Loads processed NPZ motion data with boundary tracking.

    Resolves body indices from each NPZ's ``body_names`` independently of the simulated robot, whose body
    count and order may differ from the NPZ's.
    """

    def __init__(self, motion_files: str | list[str], tracked_body_names: list[str],
                 device: str = "cpu"):

        motion_files = _require_clips(motion_files)
        self.num_motions = len(motion_files)

        # Raise the soft open-file limit (often 1024) up to the hard limit, for headroom with thousands of clips and
        # the USD / PhysX descriptors.
        try:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            target = max(soft, min(hard, 65536))
            if target > soft:
                resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        except (ImportError, ValueError, OSError):
            pass

        # np.load returns a lazy NpzFile with an open file handle: materialize each clip into a dict and close it.
        all_data: list[dict[str, np.ndarray]] = []
        for f in motion_files:
            assert os.path.isfile(f), f"Motion file not found: {f}"
            with np.load(f, allow_pickle=True) as nz:
                all_data.append({k: nz[k] for k in nz.files})

        # The env advances one motion frame per control step, so every clip must share one frame rate.
        fps_values = {float(np.asarray(d["fps"]).reshape(-1)[0]) for d in all_data}
        if len(fps_values) != 1:
            raise RuntimeError(f"motion clips have mixed frame rates {sorted(fps_values)}; re-export them at one fps.")
        self.fps = all_data[0]["fps"]

        # Resolve the body indices per clip: datasets need not share a body_names list (e.g. a ``world`` body in
        # some), so each clip is sliced to the tracked bodies before the concat.
        per_motion_body_idx: list[list[int]] = []
        idx_cache: dict[tuple[str, ...], list[int]] = {}
        for mi, d in enumerate(all_data):
            body_names = [str(n) for n in d["body_names"]]
            key = tuple(body_names)
            if key not in idx_cache:
                try:
                    idx_cache[key] = [body_names.index(n) for n in tracked_body_names]
                except ValueError as e:
                    missing = [n for n in tracked_body_names if n not in body_names]
                    raise RuntimeError(
                        f"motion file {motion_files[mi]} is missing tracked "
                        f"body(s) {missing}. Tracked set (from cfg.body_names) "
                        f"must be a subset of every NPZ's body_names."
                    ) from e
            per_motion_body_idx.append(idx_cache[key])

        def _cat(key):
            """Scalar-per-motion concat along time dim. Shape: (sum(T_i), …)."""
            return torch.cat([
                torch.tensor(d[key], dtype=torch.float32, device=device)
                for d in all_data
            ], dim=0)

        def _cat_body(key):
            """Per-motion slice-to-tracked + concat. Shape: (sum(T_i), N_track, …)."""
            return torch.cat([
                torch.tensor(d[key], dtype=torch.float32, device=device)[:, idx]
                for d, idx in zip(all_data, per_motion_body_idx)
            ], dim=0)

        # Joint state (already in Isaac Lab order from data processing)
        self.joint_pos = _cat("joint_pos")
        self.joint_vel = _cat("joint_vel")

        # Body state — tracked-body subset, concatenated across motions.
        self._body_pos_w = _cat_body("body_pos_w")
        self._body_quat_w = _cat_body("body_quat_w")
        self._body_lin_vel_w = _cat_body("body_lin_vel_w")
        self._body_ang_vel_w = _cat_body("body_ang_vel_w")

        # Root state
        self.root_pos = _cat("root_pos")
        self.root_quat = _cat("root_quat")         # wxyz
        self.root_gravity = _cat("root_gravity")
        self.root_lin_vel = _cat("root_lin_vel_w")
        self.root_ang_vel = _cat("root_ang_vel_w")

        # Root velocity in body frame
        self.root_lin_vel_b = _cat("root_lin_vel_b")
        self.root_ang_vel_b = _cat("root_ang_vel_b")

        # Object state (optional — present when motion includes object interaction)
        # Check ALL files, not just the first — mixed folders may have both types.
        self.has_object = any("object_pos_w" in d for d in all_data)
        if self.has_object:
            obj_tensors = {"object_pos_w": [], "object_quat_w": [],
                           "object_lin_vel_w": [], "object_ang_vel_w": []}
            for d in all_data:
                n = d["joint_pos"].shape[0]
                for key in obj_tensors:
                    if key in d:
                        obj_tensors[key].append(torch.tensor(d[key], dtype=torch.float32, device=device))
                    else:
                        dim = 4 if "quat" in key else 3
                        obj_tensors[key].append(torch.zeros(n, dim, dtype=torch.float32, device=device))
            self.object_pos_w = torch.cat(obj_tensors["object_pos_w"], dim=0)
            self.object_quat_w = torch.cat(obj_tensors["object_quat_w"], dim=0)
            self.object_lin_vel_w = torch.cat(obj_tensors["object_lin_vel_w"], dim=0)
            self.object_ang_vel_w = torch.cat(obj_tensors["object_ang_vel_w"], dim=0)
            # Default quat for files without object: identity (1,0,0,0)
            zero_mask = self.object_quat_w.norm(dim=-1) < 1e-6
            self.object_quat_w[zero_mask, 0] = 1.0
        else:
            self.object_pos_w = None
            self.object_quat_w = None
            self.object_lin_vel_w = None
            self.object_ang_vel_w = None

        self.time_step_total = self.joint_pos.shape[0]

        # Motion boundaries
        self.motion_boundaries = []
        cumulative_frames = 0
        for data in all_data:
            num_frames = data["joint_pos"].shape[0]
            self.motion_boundaries.append((cumulative_frames, cumulative_frames + num_frames))
            cumulative_frames += num_frames

        # Pre-compute boundary tensors for fast vectorized operations
        self.motion_start_indices = torch.tensor(
            [start for start, _ in self.motion_boundaries],
            dtype=torch.long,
            device=device
        )  # (num_motions,)

        self.motion_end_indices = torch.tensor(
            [end - 1 for _, end in self.motion_boundaries],
            dtype=torch.long,
            device=device
        )  # (num_motions,)

    def is_at_motion_boundary(self, time_steps: torch.Tensor) -> torch.Tensor:
        return (time_steps.unsqueeze(1) == self.motion_end_indices.unsqueeze(0)).any(dim=1)

    def get_motion_id(self, time_steps: torch.Tensor) -> torch.Tensor:
        motion_ids = torch.searchsorted(self.motion_end_indices, time_steps, right=False)
        motion_ids = torch.clamp(motion_ids, 0, self.num_motions - 1)
        return motion_ids

    # Tracked bodies, already sliced to ``tracked_body_names`` at load (returned without a copy)
    @property
    def body_pos_w(self): return self._body_pos_w
    @property
    def body_quat_w(self): return self._body_quat_w
    @property
    def body_lin_vel_w(self): return self._body_lin_vel_w
    @property
    def body_ang_vel_w(self): return self._body_ang_vel_w


class MotionCommand(CommandTerm):
    """Motion command with horizon, adaptive sampling, and config-driven keypoints."""

    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
        self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)

        # Robot-side body indices (Isaac Lab order)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(cfg.body_names, preserve_order=True)[0],
            dtype=torch.long, device=self.device
        )

        # A subclass (MotionCommandObjectTerrain) installs its scene-aware loader before calling this __init__.
        # Clips bound to an object or terrain need that scene-aware command.
        if getattr(self, "motion", None) is None:
            self.motion = MotionLoader(
                _resolve_motion_files(cfg), tracked_body_names=cfg.body_names, device=self.device
            )
        control_hz = 1.0 / (env.cfg.decimation * env.cfg.sim.dt)
        motion_hz = float(np.asarray(self.motion.fps).reshape(-1)[0])
        if abs(motion_hz - control_hz) > 1e-3 * control_hz:
            raise RuntimeError(
                f"motion clips are {motion_hz:g} fps but the env steps at {control_hz:g} Hz (one motion frame per "
                f"control step); re-export them with data/scripts/extract_sequences.py --output_fps {control_hz:g}."
            )
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._frozen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # keypoints
        keypoint_body_names = ["left_ankle_roll_link", "right_ankle_roll_link",
                               "left_wrist_yaw_link", "right_wrist_yaw_link"]
        self.keypoint_body_indexes = [cfg.body_names.index(n) for n in keypoint_body_names if n in cfg.body_names]
        self.keypoint_dim = 3 * len(self.keypoint_body_indexes)

        self.joint_dim = self.motion.joint_pos.shape[-1]

        # Adaptive sampling
        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [cfg.adaptive_lambda**i for i in range(cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        for k in ["error_anchor_pos", "error_anchor_rot", "error_anchor_lin_vel", "error_anchor_ang_vel",
                   "error_body_pos", "error_body_rot", "error_joint_pos", "error_joint_vel",
                   "sampling_entropy", "sampling_top1_prob", "sampling_top1_bin"]:
            self.metrics[k] = torch.zeros(self.num_envs, device=self.device)


    # -- Command interface --

    @property
    def command(self) -> torch.Tensor:
        return self.ref_motion(include_keypoints=False)

    def ref_motion(self, flatten_horizon=None, horizon=None, include_keypoints=False):
        """Future trajectory: [t+H-1, ..., t+1, t] (reversed), clamped to the current clip.

        Per frame: joint_pos, joint_vel, root gravity, root lin/ang velocity (body frame). With
        ``include_keypoints=True`` it appends the world-frame keypoint position error
        ``motion_keypoint_pos_w - robot_keypoint_pos_w`` (ankles and wrists).
        """
        horizon = self.cfg.horizon if horizon is None else horizon
        flatten = self.cfg.flatten_horizon if flatten_horizon is None else flatten_horizon

        offsets = torch.arange(horizon - 1, -1, -1, device=self.device).unsqueeze(0)
        future_t = self.time_steps.unsqueeze(1) + offsets

        motion_ids = self.motion.get_motion_id(self.time_steps)
        lo = self.motion.motion_start_indices[motion_ids].unsqueeze(1)
        hi = self.motion.motion_end_indices[motion_ids].unsqueeze(1)
        future_t = torch.clamp(future_t, lo, hi)
        flat_t = future_t.reshape(-1)

        features = [self.motion.joint_pos[flat_t], self.motion.joint_vel[flat_t],
                    self.motion.root_gravity[flat_t],
                    self.motion.root_lin_vel_b[flat_t], self.motion.root_ang_vel_b[flat_t]]
        dims = [self.joint_dim, self.joint_dim, 3, 3, 3]

        if include_keypoints and self.keypoint_body_indexes:
            kp_idx = self.keypoint_body_indexes
            num_kp = len(kp_idx)
            # Motion keypoint world pos (E*H, num_kp, 3)
            motion_kp_w = self.motion.body_pos_w[flat_t][:, kp_idx]
            env_origins = self._env.scene.env_origins.unsqueeze(1).expand(-1, horizon, -1).reshape(-1, 1, 3)
            motion_kp_w = motion_kp_w + env_origins
            # Robot keypoint world pos, repeated for horizon (E*H, num_kp, 3)
            robot_kp_w = self.robot.data.body_pos_w[:, self.body_indexes[kp_idx]]
            robot_kp_w = robot_kp_w.unsqueeze(1).expand(-1, horizon, -1, -1).reshape(-1, num_kp, 3)
            # World-frame pos error, flattened
            kp_error = (motion_kp_w - robot_kp_w).reshape(flat_t.shape[0], -1)
            features.append(kp_error)
            dims.append(self.keypoint_dim)

        if flatten:
            return torch.cat([f.reshape(self.num_envs, -1) for f in features], dim=-1)
        else:
            return torch.cat([f.reshape(self.num_envs, horizon, d) for f, d in zip(features, dims)], dim=-1)

    # -- Properties (reference) --

    @property
    def joint_pos(self): return self.motion.joint_pos[self.time_steps]
    @property
    def joint_vel(self): return self.motion.joint_vel[self.time_steps]
    @property
    def body_pos_w(self): return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]
    @property
    def body_quat_w(self): return self.motion.body_quat_w[self.time_steps]
    @property
    def body_lin_vel_w(self): return self.motion.body_lin_vel_w[self.time_steps]
    @property
    def body_ang_vel_w(self): return self.motion.body_ang_vel_w[self.time_steps]
    @property
    def anchor_pos_w(self): return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins
    @property
    def anchor_quat_w(self): return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]
    @property
    def anchor_lin_vel_w(self): return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]
    @property
    def anchor_ang_vel_w(self): return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    # -- Properties (robot) --

    @property
    def robot_joint_pos(self): return self.robot.data.joint_pos
    @property
    def robot_joint_vel(self): return self.robot.data.joint_vel
    @property
    def robot_body_pos_w(self): return self.robot.data.body_pos_w[:, self.body_indexes]
    @property
    def robot_body_quat_w(self): return self.robot.data.body_quat_w[:, self.body_indexes]
    @property
    def robot_body_lin_vel_w(self): return self.robot.data.body_lin_vel_w[:, self.body_indexes]
    @property
    def robot_body_ang_vel_w(self): return self.robot.data.body_ang_vel_w[:, self.body_indexes]
    @property
    def robot_anchor_pos_w(self): return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]
    @property
    def robot_anchor_quat_w(self): return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]
    @property
    def robot_anchor_lin_vel_w(self): return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]
    @property
    def robot_anchor_ang_vel_w(self): return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    # -- Metrics --

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)
        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(dim=-1)
        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)


    # -- Adaptive sampling --

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1)
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        probs = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        probs = torch.nn.functional.pad(probs[None, None], (0, self.cfg.adaptive_kernel_size - 1), mode="replicate")
        probs = torch.nn.functional.conv1d(probs, self.kernel.view(1, 1, -1)).view(-1)
        probs = probs / probs.sum()

        sampled_bins = torch.multinomial(probs, len(env_ids), replacement=True)
        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count * (self.motion.time_step_total - 1)
        ).long()

        H = -(probs * (probs + 1e-12).log()).sum()
        self.metrics["sampling_entropy"][:] = H / math.log(self.bin_count)
        pmax, imax = probs.max(dim=0)
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    # -- CommandTerm lifecycle --

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        if self.cfg.use_adaptive_sampling:
            self._adaptive_sampling(env_ids)
        else:
            rand_motion = torch.randint(0, self.motion.num_motions, (len(env_ids),), device=self.device)
            self.time_steps[env_ids] = self.motion.motion_start_indices[rand_motion]

        # Freeze at t=0: with some probability, keep time_steps at motion start and freeze advancement
        self._frozen[env_ids] = False
        if self.cfg.freeze_at_t0_prob > 0.0:
            freeze_mask = torch.rand(len(env_ids), device=self.device) < self.cfg.freeze_at_t0_prob
            motion_ids = self.motion.get_motion_id(self.time_steps[env_ids])
            starts = self.motion.motion_start_indices[motion_ids]
            at_start = self.time_steps[env_ids] == starts
            self._frozen[env_ids] = freeze_mask & at_start

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])

        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )

        joint_vel += sample_uniform(*self.cfg.joint_velocity_range, joint_vel.shape, joint_vel.device)

        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )
        # Terminations are computed before the next ``_update_command``; without this, the first step after the
        # initial reset compares the robot against the zero-initialized relative poses and ends every episode.
        self._update_relative_body_poses()

    def _update_relative_body_poses(self):
        """Reference body poses re-anchored to the robot's anchor (robot XY and heading, reference height)."""
        anchor_pos = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos = robot_anchor_pos.clone()
        delta_pos[..., 2] = anchor_pos[..., 2]
        delta_ori = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(anchor_quat)))

        self.body_quat_relative_w = quat_mul(delta_ori, self.body_quat_w)
        self.body_pos_relative_w = delta_pos + quat_apply(delta_ori, self.body_pos_w - anchor_pos)

    def _update_command(self):
        # Only advance non-frozen environments
        advance_mask = ~self._frozen
        self.time_steps[advance_mask] += 1

        at_boundary = self.motion.is_at_motion_boundary(self.time_steps)
        overflow = self.time_steps >= self.motion.time_step_total
        resample_ids = torch.cat([torch.where(at_boundary)[0], torch.where(overflow)[0]]).unique()

        if len(resample_ids) > 0:
            self._resample_command(resample_ids)

        self._update_relative_body_poses()

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed
            + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor"))
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor"))
                self.current_body_visualizers = [
                    VisualizationMarkers(self.cfg.body_visualizer_cfg.replace(prim_path=f"/Visuals/Command/current/{n}"))
                    for n in self.cfg.body_names]
                self.goal_body_visualizers = [
                    VisualizationMarkers(self.cfg.body_visualizer_cfg.replace(prim_path=f"/Visuals/Command/goal/{n}"))
                    for n in self.cfg.body_names]
            for v in [self.current_anchor_visualizer, self.goal_anchor_visualizer] + self.current_body_visualizers + self.goal_body_visualizers:
                v.set_visibility(True)
        else:
            if hasattr(self, "current_anchor_visualizer"):
                for v in [self.current_anchor_visualizer, self.goal_anchor_visualizer] + self.current_body_visualizers + self.goal_body_visualizers:
                    v.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)
        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    class_type: type = MotionCommand

    asset_name: str = MISSING
    motion_file: str | list[str] | None = None
    """Explicit NPZ clips, used when ``motion_folder`` is not set (``None`` rather than ``MISSING``, so a task may set
    only ``motion_folder``)."""
    motion_folder: str | list[str] | None = None
    """Root folder(s) globbed recursively for NPZ clips; overrides ``motion_file``."""
    max_motions_per_subfolder: int | None = 1000
    """Cap on the clips kept per leaf folder of ``motion_folder`` (``_original`` clips first; ``None`` = no cap)."""
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    horizon: int = 1
    flatten_horizon: bool = True

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    joint_position_range: tuple[float, float] = (-0.1, 0.1)
    joint_velocity_range: tuple[float, float] = (-0., 0.)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    use_adaptive_sampling: bool = False
    freeze_at_t0_prob: float = 0.0
    """Probability that a resample landing on a clip's first frame holds the reference there (no advance)."""

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)


# ============================================================================
# Scene-aware motion command: MotionCommandObjectTerrain(Cfg) — per-env
# object/terrain asset routing + compatible-motion selection for the
# multi-asset scene tasks.
# ============================================================================

HIDDEN_Z = -100.0
"""Graveyard height for objects an env's current motion does not use."""

_CALIB_SKIP_FRAMES = 5
"""Leading frames skipped when sampling a start frame (the mocap calibration pose)."""


def resolve_object_type(npz_path: str) -> str | None:
    """Object key (``OBJECT_KEYS_FROM_FILENAME``) of the first entry with a substring in the file name, or None."""
    fname = Path(npz_path).stem
    for object_key, substrings in OBJECT_KEYS_FROM_FILENAME.items():
        if any(sub in fname for sub in substrings):
            return object_key
    return None


def _resolve_motion_terrain_variant(npz_path: str, object_dir: Path) -> str | None:
    """Absolute path of the terrain URDF the file name binds the clip to (``TERRAIN_PREFIX_CATEGORIES``), or None
    when the clip is not terrain-bound or the URDF is not installed under ``object_dir``."""
    fname = Path(npz_path).stem
    for prefix, category in TERRAIN_PREFIX_CATEGORIES.items():
        if fname.startswith(prefix):
            urdf = object_dir / category / f"{category}.urdf"
            return str(urdf) if urdf.exists() else None
    return None


def _drop_clips_without_terrain(motion_files: list[str], object_dir: Path) -> list[str]:
    """Drop clips whose filename binds a terrain (``TERRAIN_PREFIX_CATEGORIES``) that is not installed.

    Such a clip would otherwise resolve to "no terrain constraint" and be played on flat ground, where its
    reference sits on geometry that does not exist.
    """
    kept, dropped = [], Counter()
    for f in motion_files:
        stem = Path(f).stem
        prefix = next((p for p in TERRAIN_PREFIX_CATEGORIES if stem.startswith(p)), None)
        if prefix is not None and _resolve_motion_terrain_variant(f, object_dir) is None:
            dropped[prefix] += 1
        else:
            kept.append(f)
    if dropped:
        print(f"[SceneMotionLoader] WARNING: skipped {sum(dropped.values())} clip(s) whose terrain asset is not "
              f"installed under {object_dir} (by filename prefix: {dict(dropped)}).")
    if not kept:
        raise RuntimeError(f"none of the {len(motion_files)} motion clips has its terrain asset installed under "
                           f"{object_dir}; see assets/objects/README.md.")
    return kept


class SceneMotionLoader(MotionLoader):
    """MotionLoader + per-motion (object_type, terrain_variant)."""

    def __init__(self, motion_files, tracked_body_names, device="cpu"):
        motion_files = _drop_clips_without_terrain(_require_clips(motion_files), ASSETS_DIR / "objects")
        super().__init__(motion_files, tracked_body_names, device)
        # The per-env dispatch and the teacher routing classify each clip by its file name.
        self.motion_files: list[str] = list(motion_files)

        # Object type bucketing.
        obj_types = [resolve_object_type(f) for f in motion_files]
        unique = sorted({t for t in obj_types if t is not None})
        self.object_type_list = unique
        type_to_id = {t: i for i, t in enumerate(unique)}
        self.motion_object_type_ids = torch.tensor(
            [type_to_id.get(t, -1) for t in obj_types],
            dtype=torch.long, device=device,
        )

        # Per-motion terrain URDF, matched against each env's spawned terrain in MotionCommandObjectTerrain.
        object_dir = ASSETS_DIR / "objects"
        assert object_dir.is_dir(), f"object asset root not found: {object_dir}"
        self.motion_terrain_variant: list[str | None] = [
            _resolve_motion_terrain_variant(f, object_dir) for f in motion_files
        ]

        n_terrain = sum(v is not None for v in self.motion_terrain_variant)
        print(f"[SceneMotionLoader] {self.num_motions} motions, "
              f"object types: {unique or ['none']}, terrain-bound: {n_terrain}")

        # Clips under a "robot-terrain" path that all lack a terrain prefix are most likely misnamed: they would be
        # played on flat ground as robot-only clips.
        terrain_dirs = [f for f in motion_files if "robot-terrain" in f]
        if terrain_dirs and n_terrain == 0:
            raise RuntimeError(
                f"motion folder appears to reference terrains ({terrain_dirs[0]}) but "
                f"no NPZ resolved to a terrain URDF under {object_dir}. "
                "Check the file-name prefixes in TERRAIN_PREFIX_CATEGORIES (mdp/commands.py)."
            )


class MotionCommandObjectTerrain(MotionCommand):
    """Motion command with per-env multi-asset spawning + Graveyard Dispatcher.

    Exposes:
      * ``has_object_mask`` — ``(num_envs,)`` bool: True where env has active object.
      * ``robot_object_pos_w`` etc. — sim object state, **zeroed** for inactive envs.
      * ``object_pos_w`` etc. — reference object state from motion data, also zeroed.
    """

    cfg: "MotionCommandObjectTerrainCfg"

    def __init__(self, cfg: "MotionCommandObjectTerrainCfg", env: ManagerBasedRLEnv):
        # Install the scene-aware loader before the parent __init__, which then skips its own loader.
        self.motion = SceneMotionLoader(_resolve_motion_files(cfg), cfg.body_names, env.device)
        super().__init__(cfg, env)

        # The parent's _resample_command calls _adaptive_sampling (a no-op here) only with use_adaptive_sampling;
        # otherwise it would overwrite the per-env compatible start frames with a pick over all clips. "Adaptive" is
        # a misnomer for this subclass: _resample_command picks a compatible clip per env and a uniform frame in it.
        assert cfg.use_adaptive_sampling, (
            "MotionCommandObjectTerrain requires cfg.use_adaptive_sampling=True; "
            "the parent's else-branch would silently overwrite per-env compatible-"
            "motion sampling with a uniform random pick, defeating the Graveyard "
            "Dispatcher and producing URDF/motion mismatches."
        )

        # Every scene rigid object named after an ``OBJECT_KEYS_FROM_FILENAME`` key is graveyard-managed.
        self._scene_objects: dict[str, "RigidObject"] = {}
        for key in OBJECT_KEYS_FROM_FILENAME:
            if key in env.scene.rigid_objects:
                self._scene_objects[key] = env.scene[key]

        self._obj_type_to_id = {t: i for i, t in enumerate(self.motion.object_type_list)}

        # Per-env: which object type is currently active (-1 = none).
        self._env_active_obj_type = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device)

        # Graveyard: a per-env grid below the floor, so the hidden objects do not overlap (overlapping AABBs of
        # every env would overflow the PhysX broadphase pair buffers). The spacing must exceed the largest hidden
        # object; 2.0 m covers the push box.
        GRAVEYARD_SPACING = 2.0
        side = int(math.ceil(math.sqrt(self.num_envs)))
        ix = torch.arange(self.num_envs, device=self.device) % side
        iy = torch.arange(self.num_envs, device=self.device) // side
        self._graveyard_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._graveyard_state[:, 0] = (ix.float() - side * 0.5) * GRAVEYARD_SPACING
        self._graveyard_state[:, 1] = (iy.float() - side * 0.5) * GRAVEYARD_SPACING
        self._graveyard_state[:, 2] = HIDDEN_Z
        self._graveyard_state[:, 3] = 1.0  # identity quat w

        # Cached identity quaternion for object_quat_w and _get_active_object_state.
        self._identity_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._identity_quat[:, 0] = 1.0

        # Per-env spawned terrain URDF: MultiAssetSpawnerCfg(random_choice=False) gives env_i assets[i % N].
        self._env_terrain_variant = self._compute_env_variant_strings(env, "terrain_object")

        # Per env, the motion ids whose (terrain, object type) match its spawned scene; _resample_command samples them.
        self._env_compatible_motions = self._build_compatible_motion_buckets()

        # Padded (num_envs, max_bucket) bucket tensor (-1 padding) and a validity mask, so one torch.multinomial
        # samples every resetting env.
        bucket_sizes = [int(b.numel()) for b in self._env_compatible_motions]
        max_bucket = max(bucket_sizes)
        padded = torch.full(
            (self.num_envs, max_bucket), -1, dtype=torch.long, device=self.device,
        )
        for i, b in enumerate(self._env_compatible_motions):
            padded[i, : b.numel()] = b
        self._env_buckets_padded = padded
        col_idx = torch.arange(max_bucket, device=self.device).unsqueeze(0)
        bucket_len = torch.tensor(bucket_sizes, device=self.device).unsqueeze(1)
        # 1.0 where bucket[i, j] is valid, 0.0 for padded positions (multinomial weights).
        self._env_bucket_valid = (col_idx < bucket_len).float()
        if cfg.object_sampling_share is not None:
            # Object clips get `object_sampling_share` of the probability in mixed buckets, each group uniform.
            valid = self._env_bucket_valid
            uses_object = (self.motion.motion_object_type_ids[padded.clamp(min=0)] >= 0).float() * valid
            others = valid - uses_object
            n_obj, n_oth = uses_object.sum(1, keepdim=True), others.sum(1, keepdim=True)
            mixed = (n_obj > 0) & (n_oth > 0)
            share = float(cfg.object_sampling_share)
            weighted = uses_object * share / n_obj.clamp(min=1) + others * (1.0 - share) / n_oth.clamp(min=1)
            self._env_bucket_valid = torch.where(mixed, weighted, valid)

        print(f"[MotionCommandObjectTerrain] scene objects: {list(self._scene_objects.keys())}; "
              f"terrain variants in scene: "
              f"{len(set(self._env_terrain_variant)) if self._env_terrain_variant else 0}")

        # Move every object to its graveyard slot before the first physics step: the scene cfg spawns all of them
        # at the same init_state.pos.
        if self._scene_objects:
            all_env_ids = torch.arange(self.num_envs, device=self.device)
            for obj in self._scene_objects.values():
                obj.write_root_state_to_sim(
                    self._graveyard_state[all_env_ids], env_ids=all_env_ids
                )

    # ── Per-env variant resolution ────────────────────────────────────

    def _compute_env_variant_strings(self, env: ManagerBasedRLEnv, entity_name: str) -> list[str | None] | None:
        """Mirror ``MultiAssetSpawnerCfg(random_choice=False)`` → env_i ↦ assets[i % N].

        Returns a per-env list of asset_path strings (one per env), or ``None`` if
        the scene does not have this entity. For single-URDF entities the same
        path is repeated for every env.
        """
        # Read from the scene cfg: ``terrain_object`` is an AssetBaseCfg, not a rigid object.
        entity_cfg = getattr(env.scene.cfg, entity_name, None)
        if entity_cfg is None:  # absent, or not spawned (assets missing)
            return None
        spawn_cfg = entity_cfg.spawn
        if spawn_cfg is None:
            return None
        # MultiAssetSpawnerCfg has assets_cfg list; UrdfFileCfg has asset_path.
        assets_cfg = getattr(spawn_cfg, "assets_cfg", None)
        if assets_cfg is not None and len(assets_cfg) > 0:
            n = len(assets_cfg)
            return [getattr(assets_cfg[i % n], "asset_path", None) for i in range(self.num_envs)]
        single_path = getattr(spawn_cfg, "asset_path", None)
        return [single_path] * self.num_envs

    def _build_compatible_motion_buckets(self) -> list[torch.Tensor]:
        """For each env, list motion_ids whose variant matches the spawned URDF.

        Two-tier compatibility match:

          **Tier 1 — exact match (preferred).** The motion's terrain and object type match the env:
            * terrain_variant set → the env's spawned terrain URDF must match exactly.
            * terrain_variant None (robot-only / push-box clips, on flat ground) → the env has no
              terrain asset (no terrain, or the SuperScene's generated flat plate).
            * the object type must be spawned in the scene, or the motion is robot-only.

          **Tier 2 — robot-only fallback** (only if Tier 1 is empty). Play any robot-only motion on
          the spawned terrain. Noisy signal (robot motion ↔ wrong terrain) but lets training proceed,
          e.g. when the sofa clips are not part of the motion set.

        Logs the fallback count. Raises only if an env has no compatible motion at all.
        """
        motion_terrain = self.motion.motion_terrain_variant
        motion_obj_type = self.motion.motion_object_type_ids.tolist()

        def _obj_ok(m: int) -> bool:
            """True iff motion ``m``'s object type is spawnable in this scene."""
            oid = motion_obj_type[m]
            return (
                oid < 0
                or self.motion.object_type_list[oid] in self._scene_objects
            )

        def _terrain_ok(m: int, env_terrain: str | None, env_has_real_terrain: bool) -> bool:
            """True iff motion ``m`` is terrain-compatible with this env."""
            if motion_terrain[m] is not None:
                return (
                    env_terrain is not None
                    and os.path.normpath(motion_terrain[m]) == os.path.normpath(env_terrain)
                )
            # Motion has no terrain constraint — reject if env has a real
            # terrain asset. Flat-plate and no-terrain envs both accept
            # robot-only / push-box motions.
            return not env_has_real_terrain

        buckets: list[list[int]] = [[] for _ in range(self.num_envs)]
        fallback_envs: list[int] = []
        env_descs: dict[int, str] = {}

        for i in range(self.num_envs):
            env_terrain = self._env_terrain_variant[i] if self._env_terrain_variant else None
            env_has_real_terrain = env_terrain is not None
            env_descs[i] = f"terrain={env_terrain}"

            # Tier 1: exact (terrain, object) match.
            for m in range(self.motion.num_motions):
                if _terrain_ok(m, env_terrain, env_has_real_terrain) and _obj_ok(m):
                    buckets[i].append(m)
            if buckets[i]:
                continue

            # Tier 2: robot-only fallback.
            for m in range(self.motion.num_motions):
                if motion_terrain[m] is None and motion_obj_type[m] < 0:
                    buckets[i].append(m)
            if buckets[i]:
                fallback_envs.append(i)

        if fallback_envs:
            uniq = sorted({env_descs[i] for i in fallback_envs})
            print(
                f"[MotionCommandObjectTerrain] WARN: {len(fallback_envs)} envs had no "
                f"motion matching their spawned terrain; fell back to robot-only clips "
                f"(degraded signal: robot motion ↔ wrong terrain). Include the matching "
                f"clips in --motion_data_folder. Uncovered: {uniq}"
            )

        # Final hard check — only fires if the motion set doesn't even
        # contain a single robot-only motion (truly broken data).
        for i in range(self.num_envs):
            assert buckets[i], (
                f"env {i} ({env_descs[i]}) has no compatible motion even after the "
                f"robot-only fallback. The motion set lacks any robot-only / flat "
                f"motion — check --motion_data_folder."
            )

        return [torch.tensor(b, dtype=torch.long, device=self.device) for b in buckets]


    # ── Per-env mask: does this env have an active object? ────────────

    @property
    def has_object_mask(self) -> torch.Tensor:
        """(num_envs,) bool — True where the env's current motion uses an object."""
        return self._env_active_obj_type >= 0

    # ── Reference object state (from NPZ) — zeroed for inactive envs ──

    @property
    def object_pos_w(self):
        if self.motion.object_pos_w is None:
            return torch.zeros(self.num_envs, 3, device=self.device)
        pos = self.motion.object_pos_w[self.time_steps] + self._env.scene.env_origins
        return pos * self.has_object_mask.float().unsqueeze(-1)

    @property
    def object_quat_w(self):
        if self.motion.object_quat_w is None:
            return self._identity_quat.clone()
        q = self.motion.object_quat_w[self.time_steps].clone()
        inactive = ~self.has_object_mask
        q[inactive] = self._identity_quat[inactive]
        return q

    # ── Sim object state — returns ACTIVE object's state per env ──────

    @property
    def robot_object_pos_w(self):
        return self._get_active_object_state("root_pos_w")

    @property
    def robot_object_quat_w(self):
        return self._get_active_object_state("root_quat_w")

    def _get_active_object_state(self, attr: str) -> torch.Tensor | None:
        """Gather per-env object state from whichever object entity is active.

        For envs with no active object, returns zeros (pos/vel) or identity quat.
        """
        if not self._scene_objects:
            return None

        sample = getattr(next(iter(self._scene_objects.values())).data, attr)
        if "quat" in attr:
            result = self._identity_quat.clone()
        else:
            result = torch.zeros_like(sample)

        for key, obj in self._scene_objects.items():
            # An object no loaded clip uses is never active.
            if key not in self._obj_type_to_id:
                continue
            obj_id = self._obj_type_to_id[key]
            mask = self._env_active_obj_type == obj_id
            if mask.any():
                result[mask] = getattr(obj.data, attr)[mask]

        return result

    # ── Per-env compatible motion sampling ────────────────────────────

    def _sample_compatible_motions(self, env_ids_t: torch.Tensor) -> torch.Tensor:
        """Pick one motion id per env, uniformly from that env's precomputed compatible bucket.

        One batched ``torch.multinomial`` over the padded bucket tensor built in ``__init__``; padded
        positions carry weight 0, so a motion outside an env's bucket is unreachable. Every bucket has
        at least one valid entry (asserted in ``_build_compatible_motion_buckets``).
        """
        rows = self._env_buckets_padded[env_ids_t]      # (N, max_bucket)
        weights = self._env_bucket_valid[env_ids_t]     # (N, max_bucket)
        sampled_col = torch.multinomial(weights, 1).squeeze(1)           # (N,)
        return rows.gather(1, sampled_col.unsqueeze(1)).squeeze(1)       # (N,)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        """No-op: ``_resample_command`` has already set the start frames."""
        return

    # ── Graveyard Dispatcher (reset logic) ────────────────────────────

    def _resample_command(self, env_ids: Sequence[int]):
        # The terrain is fixed per env and the sampled clips always match it, so only objects are dispatched.
        if len(env_ids) == 0:
            return

        env_ids_t = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(
            env_ids, device=self.device)

        motion_ids = self._sample_compatible_motions(env_ids_t)

        # Uniform start frame within the chosen clip, skipping its first _CALIB_SKIP_FRAMES on longer clips.
        starts = self.motion.motion_start_indices[motion_ids]
        ends = self.motion.motion_end_indices[motion_ids]
        lengths = ends - starts
        effective_skip = torch.where(
            lengths > 2 * _CALIB_SKIP_FRAMES,
            torch.full_like(lengths, _CALIB_SKIP_FRAMES),
            torch.zeros_like(lengths),
        )
        window = (lengths - effective_skip).float().clamp(min=1.0)
        offsets = (torch.rand(len(motion_ids), device=self.device) * window).long()
        self.time_steps[env_ids_t] = starts + effective_skip + offsets

        # The parent resets the robot to the reference at these start frames.
        super()._resample_command(env_ids)

        # Determine which object type each env now needs.
        needed = self.motion.motion_object_type_ids[motion_ids]
        self._env_active_obj_type[env_ids_t] = needed

        # For each scene object: activate or send to graveyard.
        for key, obj in self._scene_objects.items():
            # An object no loaded clip uses always stays in the graveyard.
            if key not in self._obj_type_to_id:
                obj.write_root_state_to_sim(
                    self._graveyard_state[env_ids_t], env_ids=env_ids_t
                )
                continue
            obj_id = self._obj_type_to_id[key]
            active_mask = needed == obj_id
            active_ids = env_ids_t[active_mask]
            inactive_ids = env_ids_t[~active_mask]

            if len(inactive_ids) > 0:
                obj.write_root_state_to_sim(
                    self._graveyard_state[inactive_ids], env_ids=inactive_ids
                )

            if len(active_ids) > 0 and self.motion.has_object:
                ts = self.time_steps[active_ids]
                pos = self.motion.object_pos_w[ts] + self._env.scene.env_origins[active_ids]
                quat = self.motion.object_quat_w[ts]
                lin_vel = self.motion.object_lin_vel_w[ts]
                ang_vel = self.motion.object_ang_vel_w[ts]
                obj.write_root_state_to_sim(
                    torch.cat([pos, quat, lin_vel, ang_vel], dim=-1),
                    env_ids=active_ids,
                )


@configclass
class MotionCommandObjectTerrainCfg(MotionCommandCfg):
    """Config for the per-env multi-asset Graveyard Dispatcher motion command."""

    class_type: type = MotionCommandObjectTerrain
    object_sampling_share: float | None = None
    """Sampling probability of object clips in envs that also have other clips; None = uniform over clips."""
