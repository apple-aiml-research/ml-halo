#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
Reference motion loading for G1 motion tracking.

- MotionLoader: CSV loading, FPS resampling, velocities, padding and reference rebuilds.
- transition_steps: blend length for a pose gap (start-of-plan transition and end-of-clip ramp).
- get_time_indices: reference frame indices for the observation history window.
"""

from typing import Optional
import numpy as np

from ..math import quat_slerp, so3_derivative, quat_difference, axis_angle_from_quat

# Blend-length rule shared by the start-of-plan transition (send_ref_motion) and the end-of-clip
# ramp (pad_motion_if_needed); see transition_steps.
TRANSITION_MIN_STEPS = 10
TRANSITION_MAX_STEPS = 100
TRANSITION_GAP_RAD = 0.3

# The per-frame motion arrays. A reference swap replaces all of them together (MotionLoader.install).
MOTION_ARRAYS = ("motion_dof_poss", "motion_dof_vels", "motion_base_poss", "motion_base_rots",
                 "motion_base_lin_vels", "motion_base_ang_vels")


def transition_steps(error: float, min_steps: int = TRANSITION_MIN_STEPS,
                     max_steps: int = TRANSITION_MAX_STEPS, gap_threshold: float = TRANSITION_GAP_RAD) -> int:
    """Blend length, in frames, for a pose gap of ``error`` radians.

    ``min_steps`` below ``gap_threshold``, else ``trunc(error / gap_threshold * min_steps)`` clamped to
    ``[min_steps, max_steps]``. Evaluated in float32 in that operation order, as the C++
    ``compute_transition_steps`` does, so both deployments give the same count for the same inputs.
    A non-finite gap gets ``max_steps``.
    """
    e = np.float32(error)
    if not np.isfinite(e):
        return int(max_steps)
    g = np.float32(gap_threshold)
    if e < g:
        return int(min_steps)
    steps = int(e / g * np.float32(min_steps))
    return int(min(max(steps, int(min_steps)), int(max_steps)))


def _gradient(arr: np.ndarray, dt: float) -> np.ndarray:
    """``np.gradient`` along time; zeros for a single frame (np.gradient needs two)."""
    if arr.shape[0] < 2:
        return np.zeros_like(arr)
    return np.gradient(arr, dt, axis=0)


def _resample_indices(n_in: int, input_fps: float, output_fps: float):
    """Source-frame indices and blend weights for resampling ``n_in`` frames from ``input_fps`` to ``output_fps``.

    Computed in float64 with a rounding-tolerant floor, so an output frame that lands on an input frame copies it
    exactly (a 50 fps clip played at 50 Hz is passed through unchanged, as the training clips are). The output count
    is ``floor(duration * output_fps) + 1``: both ends of the clip are kept when they land on an output frame. Used for
    CSV clips; the C++ CSV loader (``deploy_cpp/src/policy/motion_loader.cpp``) uses the same rule.
    """
    ratio = float(input_fps) / float(output_fps)
    n_out = int(np.floor((n_in - 1) / ratio + 1e-9)) + 1
    f = np.arange(n_out, dtype=np.float64) * ratio
    i0 = np.minimum(np.floor(f + 1e-9 * np.maximum(1.0, f)).astype(np.int64), n_in - 1)
    i1 = np.minimum(i0 + 1, n_in - 1)
    blend = np.clip(f - i0, 0.0, 1.0).astype(np.float32)
    return i0, i1, blend


class MotionLoader:
    """Reference motion for G1 motion tracking.

    Loads a CSV clip (rows ``[x, y, z, qx, qy, qz, qw, 29 joints in deployment
    order]``) or a static standing pose, resamples it from ``input_fps`` to
    ``output_fps`` (the control rate) and computes velocities. Joint arrays are
    stored in training order (``joint_remap``); base quaternions are wxyz.
    """

    MAX_PAD_S = 10.0  # pad_motion_if_needed: longest standing-pose tail, seconds

    def __init__(
        self,
        motion_file: Optional[str],
        *,
        standing_joint_pos: np.ndarray,
        standing_base_pos: np.ndarray,
        standing_base_quat: np.ndarray,
        input_fps: int = 30,
        output_fps: int = 50,
        joint_remap: Optional[np.ndarray] = None,
    ):
        """Initialize MotionLoader.

        Args:
            motion_file: Path to CSV motion file (None = the standing pose)
            standing_joint_pos: Standing joint positions, deployment order (config ``standing_joint_pos``)
            standing_base_pos: Standing base position (config ``standing_base_pos``)
            standing_base_quat: Standing base quaternion wxyz (config ``standing_base_quat_wxyz``)
            input_fps: Input FPS of the CSV file
            output_fps: Output FPS for interpolation (the control rate)
            joint_remap: Index map from deployment to training joint order
        """
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.joint_remap = joint_remap

        self.standing_joint_pos = np.asarray(standing_joint_pos, dtype=np.float32)
        self.standing_base_pos = np.asarray(standing_base_pos, dtype=np.float32)
        self.standing_base_quat = np.asarray(standing_base_quat, dtype=np.float32)

        # Load and process motion
        if motion_file is None:
            self._load_standing_pose()
        else:
            self._load_motion()
            self._interpolate_motion()
            self._compute_velocities()

        # Remap joints to training order
        if self.joint_remap is not None:
            self.motion_dof_poss = self.motion_dof_poss[:, self.joint_remap]
            self.motion_dof_vels = self.motion_dof_vels[:, self.joint_remap]

    def _load_standing_pose(self):
        """Load standing pose as static motion."""
        num_frames = int(10.0 * self.output_fps)

        self.motion_dof_poss = np.tile(self.standing_joint_pos, (num_frames, 1)).astype(np.float32)
        self.motion_base_poss = np.tile(self.standing_base_pos, (num_frames, 1)).astype(np.float32)
        self.motion_base_rots = np.tile(self.standing_base_quat, (num_frames, 1)).astype(np.float32)

        self.motion_dof_vels = np.zeros((num_frames, self.standing_joint_pos.shape[0]), dtype=np.float32)
        self.motion_base_lin_vels = np.zeros((num_frames, 3), dtype=np.float32)
        self.motion_base_ang_vels = np.zeros((num_frames, 3), dtype=np.float32)

        self.output_frames = num_frames
        self.duration = (num_frames - 1) * self.output_dt

        print(f"[MotionLoader] Loaded standing pose: {self.output_frames} frames @ {self.output_fps} FPS")

    def _load_motion(self):
        """Load motion from CSV file."""
        motion = np.loadtxt(self.motion_file, delimiter=",", ndmin=2).astype(np.float32)

        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]
        self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]  # xyzw -> wxyz
        self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"[MotionLoader] Loaded {self.motion_file}: {self.input_frames} frames @ {self.input_fps} FPS")

    def _interpolate_motion(self):
        """Interpolate motion to output FPS."""
        index_0, index_1, blend = _resample_indices(self.input_frames, self.input_fps, self.output_fps)
        self.output_frames = index_0.shape[0]

        # LERP for positions
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend[:, np.newaxis]
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend[:, np.newaxis]
        )

        # SLERP for quaternions
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend
        )

        print(f"  Interpolated to {self.output_frames} frames @ {self.output_fps} FPS")

    def _lerp(self, a: np.ndarray, b: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """Linear interpolation."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: np.ndarray, b: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """Spherical linear interpolation for quaternions."""
        result = np.zeros_like(a)
        for i in range(a.shape[0]):
            result[i] = quat_slerp(a[i], b[i], blend[i])
        return result

    def interpolate_qpos_data(self, qpos_data, input_fps):
        """Interpolate raw qpos_data from input_fps to self.output_fps.

        Args:
            qpos_data: Raw motion data (N, 36) with [xyz(3), quat_xyzw(4), joints(29)]
            input_fps: FPS of the input data

        Returns:
            Interpolated qpos_data at self.output_fps. No-op if fps match.
        """
        if input_fps == self.output_fps or len(qpos_data) < 2:
            return qpos_data

        # Streamed plans keep the float32 time grid of the C++ plan resampler
        # (deploy_cpp/src/policy/motion_processing.cpp), whose frame counts the plan-switch logic relies on.
        N = qpos_data.shape[0]
        duration = (N - 1) / input_fps
        times = np.arange(0, duration, 1.0 / self.output_fps, dtype=np.float32)
        num_output = times.shape[0]
        phase = times / duration
        index_0 = np.floor(phase * (N - 1)).astype(np.int32)
        index_1 = np.minimum(index_0 + 1, N - 1)
        blend = phase * (N - 1) - index_0

        # Extract components
        base_pos = qpos_data[:, :3].astype(np.float32)
        quat_xyzw = qpos_data[:, 3:7].astype(np.float32)
        joints = qpos_data[:, 7:].astype(np.float32)

        # LERP positions and joints
        out_base_pos = self._lerp(base_pos[index_0], base_pos[index_1], blend[:, np.newaxis])
        out_joints = self._lerp(joints[index_0], joints[index_1], blend[:, np.newaxis])

        # SLERP quaternions (xyzw -> wxyz for slerp, then back)
        quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
        out_quat_wxyz = self._slerp(quat_wxyz[index_0], quat_wxyz[index_1], blend)
        out_quat_xyzw = out_quat_wxyz[:, [1, 2, 3, 0]]

        # Reassemble
        out_qpos = np.concatenate([out_base_pos, out_quat_xyzw, out_joints], axis=1)

        print(f"[MotionLoader] Interpolated qpos_data: {N} frames @ {input_fps} FPS -> {num_output} frames @ {self.output_fps} FPS")
        return out_qpos

    def _compute_velocities(self):
        """Compute velocities from positions."""
        self.motion_base_lin_vels = _gradient(self.motion_base_poss, self.output_dt)
        self.motion_dof_vels = _gradient(self.motion_dof_poss, self.output_dt)
        self.motion_base_ang_vels = so3_derivative(self.motion_base_rots, self.output_dt)

    def arrays(self) -> dict:
        """The current per-frame motion arrays, by attribute name (``MOTION_ARRAYS``)."""
        return {k: getattr(self, k) for k in MOTION_ARRAYS}

    def install(self, arrays: dict) -> None:
        """Replace every motion array (and the frame count / duration) with ``arrays``.

        The arrays are never modified in place afterwards, so a reader that took references to them
        under the controller's swap lock keeps a consistent clip. Callers that share the loader with
        the control loop hold that lock around this call (G1Controller does).
        """
        for k in MOTION_ARRAYS:
            setattr(self, k, arrays[k])
        self.output_frames = len(arrays["motion_dof_poss"])
        self.duration = max(0.0, (self.output_frames - 1) * self.output_dt)

    def _standing_train(self) -> np.ndarray:
        """The standing joint pose in the order the motion arrays use (training order when remapped)."""
        if self.joint_remap is not None:
            return self.standing_joint_pos[self.joint_remap]
        return self.standing_joint_pos.copy()

    def padded(self, arrays: dict, required_frames: int) -> dict:
        """``arrays`` extended to ``required_frames`` (at most MAX_PAD_S seconds more), or unchanged.

        After the last clip frame the joints ramp linearly to the standing pose (``end``) over K frames
        and then hold it; the root position and orientation hold the last frame. K is ``transition_steps``
        of the largest joint gap between the last frame and ``end`` (the start-of-plan rule). With
        ``last``, ``end`` and ``delta = end - last`` in float32, padded frame ``i = 1, 2, ...`` is:

            i <  K:  joints = last + delta * (float32(i) / float32(K)),
                     joint velocity = delta * (float32(output_fps) / float32(K))
            i >= K:  joints = end, joint velocity = 0

        Root linear and angular velocities are 0. The C++ loader pads with the same rule. A pad
        shorter than K ends inside the ramp.

        Args:
            arrays: Motion arrays to extend (``MOTION_ARRAYS`` keys)
            required_frames: Required number of frames
        """
        current_frames = len(arrays["motion_dof_poss"])
        if current_frames >= required_frames or current_frames == 0:
            return arrays
        # Playback holds the last frame past the end, so padding beyond a few seconds changes nothing
        # the robot sees. Padding to a full run (recording_duration can be hours) would cost hundreds
        # of ms per send on the control path and bloat every trim.
        required_frames = min(required_frames, current_frames + int(self.MAX_PAD_S * self.output_fps))
        pad = required_frames - current_frames

        end = self._standing_train().astype(np.float32)
        last = np.asarray(arrays["motion_dof_poss"][-1], dtype=np.float32)
        quat = np.asarray(arrays["motion_base_rots"][-1]).copy()
        base_pos = np.asarray(arrays["motion_base_poss"][-1]).copy()

        delta = (end - last).astype(np.float32)
        k = transition_steps(float(np.max(np.abs(delta))))
        i = np.arange(1, pad + 1)
        in_ramp = (i < k)[:, None]
        alpha = (i.astype(np.float32) / np.float32(k)).astype(np.float32)[:, None]
        joints = np.where(in_ramp, last + delta * alpha, end).astype(np.float32)
        ramp_vel = (delta * (np.float32(self.output_fps) / np.float32(k))).astype(np.float32)
        joint_vels = np.where(in_ramp, ramp_vel, np.float32(0.0)).astype(np.float32)

        print(f"[MotionLoader] Padding {pad} frames: {min(pad, k - 1)}-frame ramp to the standing pose, then hold")
        return {
            "motion_dof_poss": np.vstack([arrays["motion_dof_poss"], joints]),
            "motion_dof_vels": np.vstack([arrays["motion_dof_vels"], joint_vels]),
            "motion_base_poss": np.vstack([arrays["motion_base_poss"], np.tile(base_pos, (pad, 1))]),
            "motion_base_rots": np.vstack([arrays["motion_base_rots"], np.tile(quat, (pad, 1))]),
            "motion_base_lin_vels": np.vstack([arrays["motion_base_lin_vels"], np.zeros((pad, 3), np.float32)]),
            "motion_base_ang_vels": np.vstack([arrays["motion_base_ang_vels"], np.zeros((pad, 3), np.float32)]),
        }

    def pad_motion_if_needed(self, required_frames: int):
        """Pad the loaded motion in place to ``required_frames``; see ``padded`` for the padding rule."""
        out = self.padded(self.arrays(), required_frames)
        if out["motion_dof_poss"] is not self.motion_dof_poss:
            self.install(out)
            print(f"  New duration: {self.duration:.2f}s ({self.output_frames} frames)")

    def compute_adaptive_transition_steps(
        self,
        current_pos: np.ndarray,
        new_qpos_data: np.ndarray,
        current_quat: Optional[np.ndarray] = None,
        min_steps: int = TRANSITION_MIN_STEPS,
        max_steps: int = TRANSITION_MAX_STEPS,
        gap_threshold: float = TRANSITION_GAP_RAD
    ) -> int:
        """Compute adaptive transition steps based on position/orientation difference.

        Args:
            current_pos: Current joint positions in training order
            new_qpos_data: New motion data (N, 36)
            current_quat: Current base quaternion (wxyz)
            min_steps: Minimum transition steps
            max_steps: Maximum transition steps
            gap_threshold: Threshold in radians for scaling

        Returns:
            Number of transition frames (``transition_steps`` of the larger of the two gaps)
        """
        new_joints_first = np.asarray(new_qpos_data[0, 7:], dtype=np.float32)

        if self.joint_remap is not None:
            new_joints_first = new_joints_first[self.joint_remap]

        max_joint_diff = np.float32(np.max(np.abs(np.asarray(current_pos, dtype=np.float32) - new_joints_first)))

        if current_quat is not None:
            new_quat_xyzw = new_qpos_data[0, 3:7]
            new_quat_wxyz = new_quat_xyzw[[3, 0, 1, 2]]
            quat_rel = quat_difference(current_quat, new_quat_wxyz)
            quat_error = np.float32(np.linalg.norm(axis_angle_from_quat(quat_rel)))
            combined_error = max(max_joint_diff, quat_error)
        else:
            combined_error = max_joint_diff

        return transition_steps(combined_error, min_steps, max_steps, gap_threshold)

    def build_reference_motion(
        self,
        current_pos: np.ndarray,
        new_qpos_data: np.ndarray,
        current_quat: Optional[np.ndarray] = None,
        start_steps: int = 20,
        enable_smooth_transition: bool = True
    ):
        """New reference-motion arrays (``MOTION_ARRAYS``), optionally with blend ramps at both ends.

        Pure: the loader is not modified (``install`` the result). With
        ``enable_smooth_transition``, prepends a ``start_steps`` ramp from the current
        joints and attitude to the clip's first frame (the root position holds the first
        frame's) and appends one from its last frame to the standing joint pose, over
        ``transition_steps`` of that end gap (the root holds the last frame).

        Args:
            current_pos: Current joint positions in training order
            new_qpos_data: New motion data (N, 36), joints in deployment order
            current_quat: Current base quaternion (wxyz; default: standing quat)
            start_steps: Number of frames of the start ramp
            enable_smooth_transition: Whether to add the ramps
        """
        end_pos = self._standing_train()
        if current_quat is None:
            current_quat = self.standing_base_quat.copy()

        new_base_pos = new_qpos_data[:, :3]
        new_base_quat_xyzw = new_qpos_data[:, 3:7]
        new_joint_pos = new_qpos_data[:, 7:]
        current_base_pos = new_base_pos[0].copy()

        new_base_quat_wxyz = new_base_quat_xyzw[:, [3, 0, 1, 2]]

        if self.joint_remap is not None:
            new_joint_pos = new_joint_pos[:, self.joint_remap]

        if enable_smooth_transition and start_steps > 0:
            # Pre-transition
            pre_joint = np.linspace(current_pos, new_joint_pos[0], start_steps, dtype=np.float32)
            pre_quat = np.array([quat_slerp(current_quat, new_base_quat_wxyz[0], i / max(1, start_steps - 1))
                                 for i in range(start_steps)], dtype=np.float32)
            pre_pos = np.linspace(current_base_pos, new_base_pos[0], start_steps, dtype=np.float32)

            # Post-transition: joints ramp to the standing pose over the end-gap length (the clip-end rule of
            # ``padded``); the root holds the last frame.
            last = new_joint_pos[-1].astype(np.float32)
            k_post = transition_steps(float(np.max(np.abs(end_pos - last))))
            alpha = (np.arange(1, k_post + 1, dtype=np.float32) / np.float32(k_post))[:, None]
            post_joint = (last + (end_pos - last).astype(np.float32) * alpha).astype(np.float32)
            post_quat = np.tile(new_base_quat_wxyz[-1], (k_post, 1)).astype(np.float32)
            post_pos = np.tile(new_base_pos[-1], (k_post, 1)).astype(np.float32)

            # Concatenate
            dof_poss = np.vstack([pre_joint, new_joint_pos, post_joint])
            base_rots = np.vstack([pre_quat, new_base_quat_wxyz, post_quat])
            base_poss = np.vstack([pre_pos, new_base_pos, post_pos])
        else:
            dof_poss = new_joint_pos
            base_rots = new_base_quat_wxyz
            base_poss = new_base_pos

        return {
            "motion_dof_poss": dof_poss,
            "motion_dof_vels": _gradient(dof_poss, self.output_dt),
            "motion_base_poss": base_poss,
            "motion_base_rots": base_rots,
            "motion_base_lin_vels": _gradient(base_poss, self.output_dt),
            "motion_base_ang_vels": so3_derivative(base_rots, self.output_dt),
        }


def get_time_indices(counter: int, control_dt: float, motion_fps: int, horizon: int, max_idx: int) -> np.ndarray:
    """Reference frame indices for the ``horizon`` policy steps starting at policy step ``counter``.

    Returns the frames at steps ``counter + horizon - 1`` down to ``counter``
    (descending; the last entry is the current frame), as int32 clipped to [0, max_idx].
    """
    # float64 + floor with a small relative tolerance: with float32 arithmetic or a float32 dt,
    # counter * dt * fps lands just below the integer and truncation shows frame c-1 instead of c.
    # The tolerance only absorbs that rounding; fractional frame times still floor.
    offsets = np.arange(horizon - 1, -1, -1, dtype=np.float64)
    passed_times = (counter + offsets) * float(control_dt)
    frames = passed_times * motion_fps
    time_indices = np.floor(frames + 1e-6 * np.maximum(1.0, frames)).astype(np.int32)
    return np.clip(time_indices, 0, max_idx)