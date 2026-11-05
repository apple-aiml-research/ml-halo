#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
MuJoCo Simulator Backend.

MuJoCo physics simulation of the G1 for sim-to-sim evaluation.
"""

import os
import time

import numpy as np
import mujoco
import mujoco.viewer

from ..math import RobotState, MotorCommand, ControllerConfig, pd_control


class SimulatorBackend:
    """MuJoCo simulation backend.

    Provides:
    - Model loading from XML
    - PD control computation
    - Passive viewer management
    - State reading and command sending
    """

    # Selects the sim depth frame hold (observation_builder.depth_hold_steps). A backend without
    # this flag is treated as hardware.
    IS_SIMULATION = True

    #: Viewer sync rate (Hz); override with HALO_VIEWER_FPS. 0 = sync every physics step.
    #: See send_command for why per-physics-step syncing makes the view sub-realtime.
    VIEWER_FPS_DEFAULT = 60.0

    def __init__(self, config: ControllerConfig):
        """Initialize MuJoCo backend.

        Args:
            config: Controller configuration with xml_path
        """
        self.config = config
        self.num_joints = config.num_joints
        self._initialized = False

        self.xml_path = config.xml_path
        self.simulation_dt = config.simulation_dt
        self.decimation = config.decimation

        # Pre-compute arrays
        self.joint_kps = config.joint_kps.copy()
        self.joint_kds = config.joint_kds.copy()
        self.default_angles = config.default_angles.copy()
        self.max_torque = config.max_torque.copy()

        # MuJoCo objects (initialized in init())
        self.m = None
        self.d = None
        self.viewer = None
        _fps = float(os.environ.get("HALO_VIEWER_FPS", self.VIEWER_FPS_DEFAULT))
        self._viewer_period = (1.0 / _fps) if _fps > 0 else 0.0
        self._last_sync = 0.0

    def init(self, headless: bool = False, **kwargs) -> bool:
        """Initialize MuJoCo model and viewer.

        Args:
            headless: If True, don't create viewer

        Returns:
            True if successful
        """
        try:
            self.m = mujoco.MjModel.from_xml_path(self.xml_path)
            self.d = mujoco.MjData(self.m)
            self.m.opt.timestep = self.simulation_dt

            self._launch_viewer(headless)

            self._initialized = True
            print(f"[SimulatorBackend] Loaded model: {self.xml_path}")
            print(f"  Joints: {self.num_joints}, dt: {self.simulation_dt}s, decimation: {self.decimation}")
            return True

        except Exception as e:
            print(f"[SimulatorBackend] Failed to load model: {e}")
            return False

    def init_from_model(self, mj_model, mj_data,
                        headless: bool = False,
                        apply_default_timestep: bool = True) -> bool:
        """Initialize the backend from a caller-supplied ``(model, data)`` pair.

        Bypasses XML loading. Used for models composed at runtime, e.g. the robot
        with the training depth camera injected
        (``variant_support.compile_scene_with_depth_camera``). After this call
        ``self.m`` / ``self.d`` / ``self.viewer`` behave the same as after :meth:`init`.

        Args:
            mj_model: pre-loaded ``mujoco.MjModel``.
            mj_data: matching ``mujoco.MjData``.
            headless: skip viewer launch.
            apply_default_timestep: overwrite ``mj_model.opt.timestep`` with
                ``self.simulation_dt`` to keep the physics rate consistent with
                the controller config.

        Returns:
            True if successful.
        """
        try:
            self.m = mj_model
            self.d = mj_data
            if apply_default_timestep:
                self.m.opt.timestep = self.simulation_dt

            self._launch_viewer(headless)

            self._initialized = True
            print("[SimulatorBackend] Initialized from in-memory model")
            print(f"  Joints: {self.num_joints}, dt: {self.simulation_dt}s, decimation: {self.decimation}")
            return True

        except Exception as e:
            print(f"[SimulatorBackend] init_from_model failed: {e}")
            return False

    def _launch_viewer(self, headless: bool) -> None:
        if headless:
            return
        self.viewer = mujoco.viewer.launch_passive(self.m, self.d)

    def set_initial_pose(self, base_pos=None, base_quat=None, joint_angles=None):
        """Set initial simulation pose and propagate kinematics.

        Args:
            base_pos: [x, y, z] base position (None = keep XML default)
            base_quat: [w, x, y, z] base quaternion (None = keep XML default)
            joint_angles: Joint angles in deployment order, which is the MJCF
                qpos[7:] order (None = keep XML default)
        """
        if base_pos is not None:
            self.d.qpos[0:3] = np.array(base_pos, dtype=np.float64)
        if base_quat is not None:
            self.d.qpos[3:7] = np.array(base_quat, dtype=np.float64)
        if joint_angles is not None:
            self.d.qpos[7:7 + len(joint_angles)] = joint_angles
        mujoco.mj_forward(self.m, self.d)

    def read_state(self) -> RobotState:
        """Read state from MuJoCo simulation.

        Returns:
            RobotState with joint positions, velocities, IMU data
        """
        # Joint state (skip floating base: 7 qpos, 6 qvel)
        joint_pos = self.d.qpos[7:7 + self.num_joints].copy().astype(np.float32)
        joint_vel = self.d.qvel[6:6 + self.num_joints].copy().astype(np.float32)

        # Floating-base (pelvis) state — used for the IMU fields, since the real G1
        # IMU is mounted on the pelvis.
        base_quat    = self.d.qpos[3:7].copy().astype(np.float32)        # wxyz
        base_ang_vel = self.d.qvel[3:6].copy().astype(np.float32)        # body-frame angular vel

        # odom_* world-frame pose = the floating base (pelvis), i.e. qpos[0:7].
        # The reference motion's root is the pelvis (motion qpos[0:7]), and
        # align_reference_motion / motion_anchor_ori compare the robot against
        # that motion root, so the robot side must be the pelvis too (a torso
        # pose would add the waist joints to the yaw alignment).
        return RobotState(
            joint_positions=joint_pos,
            joint_velocities=joint_vel,
            imu_quaternion=base_quat,
            imu_angular_velocity=base_ang_vel,
            odom_position=self.d.qpos[0:3].copy().astype(np.float32),
            odom_quaternion=base_quat,
        )

    def send_command(self, cmd: MotorCommand):
        """Apply PD control and step simulation once.

        Performs a single PD control + mj_step. The caller is responsible
        for calling this at the physics rate (e.g., 200 Hz) with decimation
        handled in the control loop.

        Args:
            cmd: Motor command with positions, velocities, gains
        """
        q = self.d.qpos[7:7 + self.num_joints]
        dq = self.d.qvel[6:6 + self.num_joints]

        # PD control, with the plant's identified drive when the config gives one (sim_kp_scale / sim_kd_scale)
        kp = cmd.kp if self.config.sim_kp_scale is None else cmd.kp * self.config.sim_kp_scale
        kd = cmd.kd if self.config.sim_kd_scale is None else cmd.kd * self.config.sim_kd_scale
        tau = pd_control(
            target_q=cmd.positions,
            q=q,
            kp=kp,
            target_dq=cmd.velocities,
            dq=dq,
            kd=kd,
        )

        # Add feedforward torque
        tau += cmd.torques

        # Clip torques to safety limits
        tau = np.clip(tau, -self.max_torque, self.max_torque)

        # Apply to simulation
        self.d.ctrl[:self.num_joints] = tau

        mujoco.mj_step(self.m, self.d)

        # Sync the viewer at display rate, not physics rate. send_command runs once per
        # physics step (200 Hz at dt=0.005), and viewer.sync() copies the whole scene to the
        # render thread; on large scenes that costs far more than the physics step and drops
        # the run below real time. Visual only: physics and policy are unaffected, and
        # headless runs never reach this branch. HALO_VIEWER_FPS=0 syncs every step.
        if self.viewer and self.viewer.is_running():
            _now = time.time()
            if self._viewer_period <= 0.0 or (_now - self._last_sync) >= self._viewer_period:
                self._last_sync = _now
                self.viewer.sync()

    def wait_for_state(self, timeout: float = 5.0) -> bool:
        """Wait for valid state (always true for simulation)."""
        return self.d is not None

    def cleanup(self):
        """Cleanup viewer."""
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def is_viewer_running(self) -> bool:
        """Check if viewer is still running."""
        return self.viewer is not None and self.viewer.is_running()