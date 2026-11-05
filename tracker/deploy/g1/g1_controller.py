#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
G1 motion-tracking controller.

Config-driven: the policy inputs are assembled from the config's ``observation:``
block by the ObservationBuilder. Backend-agnostic: works with SimulatorBackend
(deploy_mujoco.py) or UnitreeBackend (deploy_real.py, via G1RealController).
Also implements live reference swaps (send_ref_motion), e.g. for the web interface.
"""

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    BaseController,
    ControllerConfig,
    MotionLoader,
    HistoryBuffer,
    build_index_map,
    get_gravity_orientation,
    get_time_indices,
)
from utils.math import quat_to_rotation_matrix, quat_difference, quat_to_euler, quat_mul
from utils.core.motion_loader import TRANSITION_MIN_STEPS, TRANSITION_MAX_STEPS, TRANSITION_GAP_RAD
from utils.core.observation_builder import (
    ObservationSpec, ObservationBuilder, STATE_TERM_DIMS, depth_hold_steps,
)


class G1Controller(BaseController):
    """G1 motion-tracking controller (29 DOF humanoid).

    Features:
    - Observation layout from the config (one group per ONNX input), checked
      against the loaded policy's inputs at start-up
    - Optional depth input (``student_depth``) from a depth source
    - Reference alignment to the robot pose and live reference swaps
    - Torque-aware action clipping via BaseController
    """

    def __init__(self, config: ControllerConfig, policy_path: str, backend,
                 motion_loader: MotionLoader = None, depth_source=None):
        """Config-driven G1 controller.

        The observation layout comes from the config's ``observation:`` block, so one controller
        serves the proprio student (g1_v2.yaml, 2 inputs) and the depth student (g1_v2_depth.yaml,
        3 inputs).

        ``depth_source`` supplies the depth student's ``student_depth`` input: any object with
        ``get_depth() -> (H, W) float32 meters`` (sim: a MuJoCo render; real: a RealSense) and
        ``close()``. The builder does the clip+log1p encode. Configs without an image group ignore it.
        """
        obs_cfg = getattr(config, "observation", None)
        if not obs_cfg:
            raise ValueError("config has no 'observation:' block — required by the "
                             "config-driven G1Controller (see configs/g1_v2.yaml).")
        self.obs_spec = ObservationSpec.from_dict(obs_cfg)
        hlen = self.obs_spec.history_length

        super().__init__(config, policy_path, backend)
        self.motion_loader = motion_loader

        # Initial joint positions from motion data or config
        if motion_loader is not None:
            init_joint_pos_train = motion_loader.motion_dof_poss[0, :].copy()
            self.init_joint_pos = init_joint_pos_train[self.train_to_deploy_map]
        else:
            self.init_joint_pos = np.array(config.default_angles, dtype=np.float32)
        self.motion_data = self._update_motion_data() if motion_loader else None

        # Robot anchor attitude of the last observation: the seed of send_ref_motion's transition ramp.
        self.base_quat_buffer = HistoryBuffer(hlen, 4)
        # Reference swaps vs the control tick. The tick holds _motion_swap_lock from the motion read
        # to the counter increment; send_ref_motion builds the new reference outside it and only
        # publishes (arrays + counter) under it, so the tick never sees half a swap and never waits
        # for a build. _motion_send_lock serializes whole send calls (e.g. threaded web requests);
        # the control loop never takes it.
        self._motion_swap_lock = threading.Lock()
        self._motion_send_lock = threading.Lock()

        # Depth (`student_depth`); inert for configs without an image group.
        is_sim = bool(getattr(backend, "IS_SIMULATION", False))
        hold = depth_hold_steps(self.obs_spec, is_sim)
        self.obs_builder = ObservationBuilder(self.obs_spec, depth_hold_steps=hold)
        self.depth_source = depth_source
        self._needs_depth = self.obs_builder.has_image_group()
        if self._needs_depth:
            ctl_hz = 1.0 / config.control_dt
            print(f"[G1Controller] depth fed at {ctl_hz / hold:.1f} Hz: each frame held {hold} control "
                  f"step(s) ({'sim' if is_sim else 'hardware'}, control {ctl_hz:.1f} Hz)")
            if self.depth_source is None:
                print("[G1Controller] WARNING: config declares an image (depth) group but no "
                      "depth_source was given — the builder emits the far-plane fallback every "
                      "tick and the policy runs BLIND while looking perfectly healthy.")
        self._onnx_inputs = {i.name: tuple(i.shape[1:]) for i in self.policy.session.get_inputs()}
        if set(self.obs_spec.groups) != set(self._onnx_inputs):
            raise RuntimeError(
                f"obs groups {sorted(self.obs_spec.groups)} != ONNX inputs "
                f"{sorted(self._onnx_inputs)} — fix the config's observation.groups.")
        # Build every group once from placeholder inputs and check the sizes against the ONNX inputs,
        # so a spec/policy mismatch fails here and not on the first CONTROL tick.
        self._validate_dims(self._dry_build(config.num_joints))
        self._default_angles_f32 = np.array(config.default_angles, dtype=np.float32)

        print(f"[G1Controller] config-driven: anchor={self.obs_spec.anchor_mode} "
              f"groups={list(self.obs_spec.groups)} history={hlen} "
              f"onnx_inputs={self._onnx_inputs}  motion={motion_loader is not None}")

    @staticmethod
    def create_motion_loader(
        config: ControllerConfig,
        motion_name: str,
        input_fps: int = 30,
        duration: float = None,
    ) -> MotionLoader:
        """Create a MotionLoader with standing positions from config.

        Args:
            config: Controller configuration.
            motion_name: Path to a CSV clip, or a bare name resolved to tracker/data/<name>.csv.
                None = static standing pose.
            input_fps: Input FPS of the CSV file.
            duration: If provided, pad motion to cover this many seconds.
        """
        # Resolve motion path
        if motion_name is None:
            motion_file = None
        elif motion_name.endswith('.csv'):
            motion_file = motion_name
        else:
            root_dir = Path(__file__).resolve().parent.parent.parent
            motion_file = str(root_dir / 'data' / f'{motion_name}.csv')

        print(f"[INFO] Loading motion: {motion_file}")

        output_fps = int(1.0 / config.control_dt)
        loader = MotionLoader(
            motion_file,
            standing_joint_pos=config.standing_joint_pos,
            standing_base_pos=config.standing_base_pos,
            standing_base_quat=config.standing_base_quat_wxyz,
            input_fps=input_fps,
            output_fps=output_fps,
            joint_remap=build_index_map(config.deployment_joint_order, config.training_joint_order),
        )

        if duration is not None:
            loader.pad_motion_if_needed(int(duration * output_fps))

        return loader

    def _update_motion_data(self):
        """Update motion_data tuple from motion_loader."""
        if self.motion_loader is None:
            return None
        return (
            self.motion_loader.motion_dof_poss,
            self.motion_loader.motion_dof_vels,
            self.motion_loader.motion_base_rots,
            self.motion_loader.motion_base_lin_vels,
            self.motion_loader.motion_base_ang_vels,
            self.motion_loader.motion_base_poss,
        )

    @staticmethod
    def _orientation_quat(state) -> np.ndarray:
        """Robot anchor attitude (wxyz): the odometry attitude, for the yaw alignment and motion_anchor_ori.

        Both must use the same source as odom_position (the anchor the reference is translated to), or the
        IMU-to-odom yaw offset (~110° on hardware) shows up as an orientation error and the robot spins.
        Raises without odometry rather than falling back to the IMU attitude. Gravity and angular velocity
        stay on the IMU: neither depends on the yaw reference.
        """
        if getattr(state, "odom_quaternion", None) is None:
            raise RuntimeError(
                "odom_quaternion unavailable — orientation terms require odom "
                "(shared frame with odom_position). Check rt/odommodestate."
            )
        return np.asarray(state.odom_quaternion, dtype=np.float32)

    def _dry_build(self, num_joints: int) -> dict:
        """The policy feed built by a fresh builder from zero state and a one-frame motion."""
        motion = SimpleNamespace(
            motion_dof_poss=np.zeros((1, num_joints), np.float32),
            motion_dof_vels=np.zeros((1, num_joints), np.float32),
            motion_base_rots=np.array([[1.0, 0.0, 0.0, 0.0]], np.float32),
            motion_base_lin_vels=np.zeros((1, 3), np.float32),
            motion_base_ang_vels=np.zeros((1, 3), np.float32),
        )
        frame = {name: np.zeros(dim, np.float32) for name, dim in STATE_TERM_DIMS.items()}
        indices = np.zeros(self.obs_spec.history_length, np.int32)
        return ObservationBuilder(self.obs_spec).update_and_build(frame, indices, motion)

    def _validate_dims(self, feed):
        if set(feed) != set(self._onnx_inputs):
            raise RuntimeError(f"obs groups {sorted(feed)} != ONNX inputs "
                               f"{sorted(self._onnx_inputs)}.")
        for name, arr in feed.items():
            expected = self._onnx_inputs[name]
            built = arr.shape[1:]
            # Symbolic (dynamic) ONNX dims accept any size.
            if len(built) != len(expected) or any(isinstance(e, int) and e != b
                                                  for b, e in zip(built, expected)):
                raise RuntimeError(f"group '{name}' builds {built} but ONNX "
                                   f"expects {expected} — fix the config's observation block.")

    def cleanup(self):
        """Damp the robot (backend cleanup), then release the depth source."""
        super().cleanup()
        if self.depth_source is not None:
            self.depth_source.close()

    def _policy_step(self) -> np.ndarray:
        """One policy step under the swap lock: a reference swap lands before or after it, never inside."""
        with self._motion_swap_lock:
            return super()._policy_step()

    def restart_motion(self) -> None:
        """Play the current reference from its first frame again."""
        with self._motion_swap_lock:
            self.counter = 0

    def get_observation(self):
        """Config-driven observation. Returns a dict {group_name: ndarray} that
        maps 1:1 to the ONNX inputs (e.g. ``obs`` + ``policy_temporal`` [+
        ``student_depth``] for the student policies).

        Reads the reference motion and ``counter``; the control loop calls it inside
        ``_policy_step``, under the swap lock."""
        if self.motion_data is None:
            raise RuntimeError("Motion data not loaded")

        max_idx = self.motion_loader.motion_dof_poss.shape[0] - 1
        state = self.backend.read_state()
        self._cached_joint_pos = state.joint_positions.copy()
        self._cached_joint_vel = state.joint_velocities.copy()

        qj = state.joint_positions.copy()
        dqj = state.joint_velocities.copy()
        imu_quat = state.imu_quaternion.copy()
        omega = state.imu_angular_velocity.copy()

        gravity = get_gravity_orientation(imu_quat)
        qj_train = (qj - self._default_angles_f32)[self.deploy_to_train_map]
        dqj_train = dqj[self.deploy_to_train_map]

        time_indices = get_time_indices(
            self.counter, self.config.control_dt,
            self.motion_loader.output_fps, self.obs_spec.history_length, max_idx,
        )
        t_now = int(time_indices[-1])

        # Robot anchor attitude (odometry; shares the frame with the odom-aligned reference).
        anchor_quat = self._orientation_quat(state)
        ref_seq = self.motion_loader.motion_base_rots
        ref_anchor_quat = ref_seq[min(t_now, ref_seq.shape[0] - 1)]
        quat_rel = quat_difference(anchor_quat, ref_anchor_quat)
        motion_anchor_ori = quat_to_rotation_matrix(quat_rel)[:, :2].flatten().astype(np.float32)

        frame_values = {
            "base_ang_vel":        omega.astype(np.float32),
            "proj_gravity":        gravity.astype(np.float32),
            "joint_pos_rel":       qj_train.astype(np.float32),
            "joint_vel_rel":       dqj_train.astype(np.float32),
            "last_action":         self.last_actions.astype(np.float32),
            "motion_anchor_ori_b": motion_anchor_ori,
        }
        self.base_quat_buffer.add(anchor_quat.astype(np.float32))

        # The source is only read (rendered, in sim) when the builder keeps the frame.
        if self._needs_depth and self.depth_source is not None:
            self.obs_builder.set_depth_from(self.depth_source.get_depth)

        return self.obs_builder.update_and_build(frame_values, time_indices, self.motion_loader)

    def infer_action(self, observation):
        """Feed the policy runner (the tracking command is part of the observation)."""
        return self.policy.infer(observation).astype(np.float32)

    # apply_action is inherited from BaseController: every step it stores the
    # torque-clipped action (training order) as last_actions, which is the
    # `last_action` observation term of the next policy step.

    def align_reference_motion(self, qpos_data, anchor_pos=None, anchor_quat_wxyz=None):
        """SE(2)+Z align a clip so its frame 0 lands at an anchor pose.

        Rotates the clip about world-Z by (anchor_yaw - motion_yaw) and
        translates so frame 0's XY and Z sit at the anchor. The anchor selects
        what the clip is rebased ONTO:

          * Default (anchor_pos / anchor_quat_wxyz = None) -> the robot's
            current odom pose. Rebase onto the robot. Use for a fresh / cold
            start (standing from spawn, first motion load) so the clip begins
            where the robot actually is and the initial tracking error is ~0.
          * Explicit anchor -> rebase onto that pose instead, e.g. the previous
            reference motion's pose at the splice point. Use for a
            continuation / replan while already tracking, so the reference
            trajectory stays continuous and the robot's accumulated drift is
            NOT forgiven (the tracking error is preserved for the policy to
            correct).
        """
        if anchor_pos is None or anchor_quat_wxyz is None:
            # Rebase onto the robot's current pose (default), in the odometry frame of
            # _orientation_quat and odom_position.
            state = self.backend.read_state()
            anchor_quat_wxyz = np.asarray(self._orientation_quat(state), dtype=np.float64)
            anchor_pos = np.asarray(state.odom_position[0:3], dtype=np.float64)
        else:
            # Rebase onto an explicit pose (e.g. the reference splice).
            anchor_quat_wxyz = np.asarray(anchor_quat_wxyz, dtype=np.float64)
            anchor_pos = np.asarray(anchor_pos[0:3], dtype=np.float64)

        anchor_yaw = quat_to_euler(anchor_quat_wxyz)[2]

        motion_quat_xyzw = qpos_data[0, 3:7]
        motion_quat_wxyz = motion_quat_xyzw[[3, 0, 1, 2]]
        motion_yaw = quat_to_euler(motion_quat_wxyz)[2]

        yaw_offset = anchor_yaw - motion_yaw
        anchor_xyz = anchor_pos

        xyz_delta = anchor_xyz - qpos_data[0, 0:3].astype(np.float64)
        if abs(yaw_offset) < 1e-3 and np.linalg.norm(xyz_delta) < 1e-4:
            return qpos_data

        qpos_data = qpos_data.copy()

        if abs(yaw_offset) >= 1e-3:
            yaw_rot_wxyz = np.array(
                [np.cos(yaw_offset / 2), 0.0, 0.0, np.sin(yaw_offset / 2)],
                dtype=np.float64,
            )
            quats_xyzw = qpos_data[:, 3:7]
            quats_wxyz = quats_xyzw[:, [3, 0, 1, 2]]
            yaw_rot_batch = np.tile(yaw_rot_wxyz, (len(quats_wxyz), 1))
            rotated_wxyz = quat_mul(yaw_rot_batch, quats_wxyz)
            # Renormalize: quat_mul drifts off unit norm in float32, and this
            # runs on every re-anchor, so the drift would otherwise compound.
            rotated_wxyz /= np.linalg.norm(rotated_wxyz, axis=1, keepdims=True)
            qpos_data[:, 3:7] = rotated_wxyz[:, [1, 2, 3, 0]]

        motion_xy = qpos_data[:, 0:2] - qpos_data[0, 0:2]
        c, s = np.cos(yaw_offset), np.sin(yaw_offset)
        rotated_x = motion_xy[:, 0] * c - motion_xy[:, 1] * s
        rotated_y = motion_xy[:, 0] * s + motion_xy[:, 1] * c
        qpos_data[:, 0] = anchor_xyz[0] + rotated_x
        qpos_data[:, 1] = anchor_xyz[1] + rotated_y

        # Z: pure translation (yaw rotation about world z leaves z invariant).
        # Frame 0 lands at the anchor z; all other frames keep their authored
        # Z deltas from frame 0.
        qpos_data[:, 2] = anchor_xyz[2] + (qpos_data[:, 2] - qpos_data[0, 2])

        return qpos_data

    def send_ref_motion(self, qpos_data, input_fps=None, min_transition_steps=TRANSITION_MIN_STEPS,
                        max_transition_steps=TRANSITION_MAX_STEPS, enable_smooth_transition=True,
                        gap_threshold=TRANSITION_GAP_RAD, counter=0, pad_motion=True, align_anchor="robot"):
        """Replace the loader's reference motion with a new clip.

        Pipeline:
          1. (optional) FPS-interpolate qpos_data to loader.output_fps.
          2. Drop the first `counter` frames of qpos_data.
          3. Rebase the clip's frame 0 (SE(2)+Z) per ``align_anchor``.
          4. (optional) Prepend/append blend ramps from current robot state and
             to the standing rest pose.
          5. Recompute velocities; (optional) pad to recording_duration.
          6. Publish the new reference and reset self.counter to 0 in one step
             (under the swap lock), so playback starts at the new clip.

        Args:
            qpos_data: New motion (N, 36) in [xyz(3), quat_xyzw(4), joints(29)],
                joints in deployment order.
            input_fps: FPS of qpos_data. If provided and != loader.output_fps,
                qpos_data is interpolated first. Pass None or loader.output_fps
                to skip resampling.
            min_transition_steps: Lower bound for adaptive transition length.
            max_transition_steps: Upper bound for adaptive transition length.
            enable_smooth_transition: If True, prepends a pre-transition ramp
                (current robot state -> qpos_data[0]) and appends a post-
                transition ramp (qpos_data[-1] -> standing pose). If False,
                qpos_data is stored verbatim with no ramps.
            gap_threshold: Joint/quat error (radians) above which transition
                length scales up; passed to compute_adaptive_transition_steps.
            counter: Number of leading frames to drop from qpos_data after
                FPS interpolation (frames that already elapsed).
            pad_motion: If True, pad toward recording_duration (capped at
                MotionLoader.MAX_PAD_S; playback holds the last frame anyway) with a
                ramp from the last frame to the standing pose, then the held pose
                (MotionLoader.padded). Set False when the supplied clip already
                covers the desired duration.
            align_anchor: What to rebase the incoming clip's frame 0 onto:
                "robot"     - the robot's current odom pose. Use for a fresh /
                              cold start (standing from spawn, first motion
                              load) so the clip begins where the robot is.
                "reference" - the PREVIOUS reference's splice pose (frame 0 of
                              the loader's current reference, read before it is
                              overwritten). Use for a continuation / replan
                              while already tracking so the reference trajectory
                              stays continuous and the robot's drift is not
                              forgiven. Falls back to "robot" if no reference
                              exists yet.
                None/"none" - no rebase; install the clip verbatim.

        Returns:
            bool: True on success, False if no motion_loader is attached or the
                plan arrived too late (nothing left to play, or a single frame
                without a transition ramp; nothing is changed).
        """

        if self.motion_loader is None:
            print("[ERROR] No motion loader available")
            return False

        with self._motion_send_lock:
            return self._send_ref_motion_locked(
                qpos_data, input_fps=input_fps,
                min_transition_steps=min_transition_steps,
                max_transition_steps=max_transition_steps,
                enable_smooth_transition=enable_smooth_transition,
                gap_threshold=gap_threshold, counter=counter,
                pad_motion=pad_motion, align_anchor=align_anchor,
            )

    def _send_ref_motion_locked(self, qpos_data, input_fps=None,
                                min_transition_steps=TRANSITION_MIN_STEPS,
                                max_transition_steps=TRANSITION_MAX_STEPS,
                                enable_smooth_transition=True, gap_threshold=TRANSITION_GAP_RAD,
                                counter=0, pad_motion=True, align_anchor="robot"):
        """Inner body of send_ref_motion, called under _motion_send_lock."""
        print(f"[INFO] Received motion data: {qpos_data.shape}")

        # A NaN / inf in the plan would reach the observation and hold the policy off for the whole window it
        # covers: refuse it and keep the current reference (the C++ send_ref_motion does the same).
        if not np.isfinite(np.asarray(qpos_data, dtype=np.float64)).all():
            print("[WARN] Plan rejected: non-finite values; keeping the current reference")
            return False

        # Interpolate from input_fps to output_fps if needed
        if input_fps is not None and input_fps != self.motion_loader.output_fps:
            qpos_data = self.motion_loader.interpolate_qpos_data(qpos_data, input_fps)

        # A plan that arrives after `counter` frames of it have already elapsed has
        # nothing left to play: reject it before touching any state, so the current
        # reference keeps playing. Without a transition ramp the velocities need at
        # least 2 frames.
        min_left = 1 if enable_smooth_transition else 2
        if len(qpos_data) - counter < min_left:
            print(f"[WARN] Plan rejected: counter={counter}, {len(qpos_data)} frames "
                  f"(arrived too late); keeping the current reference")
            return False

        qpos_data = qpos_data[counter:]

        # Rebase the incoming clip before installing it. `align_anchor` chooses
        # what frame 0 is pinned to (see the docstring). "reference" reads the
        # loader's CURRENT reference splice pose now, before
        # the publish below replaces it.
        if align_anchor == "reference":
            ml = self.motion_loader
            with self._motion_swap_lock:
                ref_counter = int(self.counter)
                ref_poss, ref_rots = ml.motion_base_poss, ml.motion_base_rots
            if len(ref_poss) > 0:
                # Pin to the reference's current playback frame (self.counter), not
                # frame 0: the incoming clip has just had `counter` frames dropped, so
                # its frame 0 is "now". Anchoring to the old reference's frame 0 would
                # start `counter` frames behind, a backward jump at every swap.
                k = int(min(ref_counter, len(ref_poss) - 1))
                ref_anchor_pos = np.asarray(ref_poss[k], dtype=np.float64)
                ref_anchor_quat = np.asarray(ref_rots[k], dtype=np.float64)
                print(f"[align] REFERENCE anchor: counter={ref_counter} "
                      f"k={k}/{len(ref_poss) - 1} "
                      f"ref_anchor_xy=({ref_anchor_pos[0]:+.3f},{ref_anchor_pos[1]:+.3f})")
                qpos_data = self.align_reference_motion(
                    qpos_data, anchor_pos=ref_anchor_pos, anchor_quat_wxyz=ref_anchor_quat)
            else:
                # No prior reference to be continuous with -> rebase onto robot.
                print("[align] REFERENCE requested but no prior reference "
                      "(motion_base_poss empty) -> falling back to ROBOT anchor")
                qpos_data = self.align_reference_motion(qpos_data)
        elif align_anchor == "robot":
            print(f"[align] ROBOT anchor (align_anchor='robot'); self.counter={self.counter}")
            qpos_data = self.align_reference_motion(qpos_data)
        # align_anchor None / "none": install verbatim (already world-frame).

        # Seed the ramp from the robot's live joint angles (absolute, training order).
        cur_joint_pos = np.asarray(self.backend.read_state().joint_positions,
                                   dtype=np.float32)[self.deploy_to_train_map]
        cur_quat = self.base_quat_buffer.get_latest()

        # Compute adaptive transition steps
        transition_steps = 0
        if enable_smooth_transition:
            transition_steps = self.motion_loader.compute_adaptive_transition_steps(
                current_pos=cur_joint_pos,
                current_quat=cur_quat,
                new_qpos_data=qpos_data,
                min_steps=min_transition_steps,
                max_steps=max_transition_steps,
                gap_threshold=gap_threshold
            )

        # Build the new reference (smooth transitions, padding) without touching the loader.
        ml = self.motion_loader
        arrays = ml.build_reference_motion(
            current_pos=cur_joint_pos,
            new_qpos_data=qpos_data,
            current_quat=cur_quat,
            start_steps=transition_steps,
            enable_smooth_transition=enable_smooth_transition
        )
        if pad_motion:
            required_frames = int(self.recording_duration * ml.output_fps)
            arrays = ml.padded(arrays, required_frames)

        # Publish arrays and counter together.
        with self._motion_swap_lock:
            ml.install(arrays)
            self.motion_data = self._update_motion_data()
            self.counter = 0

        print(f"[SUCCESS] Reference motion updated with {len(arrays['motion_dof_poss'])} frames!")
        return True
