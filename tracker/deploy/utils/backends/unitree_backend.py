#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
Unitree Robot Backend.

Provides communication with the Unitree G1 via unitree_sdk2py (HG message types).

Every received message is stamped with ``time.monotonic()``. Odometry older than
``ODOM_STALE_S`` is reported as absent (``odom_* = None``), the same as never
received; the controller decides what a stale low state means.
"""

import time
import threading
import numpy as np

try:
    from unitree_sdk2py.core.channel import (
        ChannelPublisher,
        ChannelSubscriber,
        ChannelFactoryInitialize,
    )
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    from unitree_sdk2py.utils.crc import CRC
    UNITREE_SDK_AVAILABLE = True
except ImportError as e:
    UNITREE_SDK_AVAILABLE = False
    print(f"[WARNING] unitree_sdk2py not fully available: {e}")

from ..math import RobotState, MotorCommand, ControllerConfig
from ..core.remote_controller import RemoteController, MotorMode, init_cmd_hg, create_damping_cmd

# Odometry (rt/odommodestate) older than this is invalid. Same threshold as the C++ deployment.
ODOM_STALE_S = 0.1


class UnitreeBackend:
    """Backend for the Unitree G1.

    Handles DDS communication with Unitree robot hardware (HG message types).
    """

    def __init__(self, config: ControllerConfig):
        """Initialize Unitree backend.

        Args:
            config: Controller configuration (robot_name must name the G1)
        """
        self.config = config
        self.num_joints = config.num_joints
        self._initialized = False

        if not UNITREE_SDK_AVAILABLE:
            raise RuntimeError("unitree_sdk2py is not installed")

        robot_name = config.robot_name.lower()
        if 'g1' not in robot_name and 'humanoid' not in robot_name:
            raise ValueError(f"UnitreeBackend only supports the G1, got robot_name={config.robot_name!r}")

        # Communication state
        self.low_state = None
        self.low_cmd = None
        self.odom_state = None
        self.remote_controller = RemoteController()

        # Thread safety
        self._state_lock = threading.Lock()
        self._odom_lock = threading.Lock()
        self._last_state_time = 0.0   # time.monotonic() of the last rt/lowstate message (0 = none yet)
        self._last_odom_time = 0.0    # time.monotonic() of the last rt/odommodestate message (0 = none yet)

        # CRC calculator for command integrity
        self.crc = CRC()

    def init(self, network_interface: str = "eth0", **kwargs) -> bool:
        """Initialize DDS communication.

        Args:
            network_interface: Network interface for DDS communication

        Returns:
            True if successful
        """
        try:
            # Initialize DDS factory
            ChannelFactoryInitialize(0, network_interface)

            self._init_hg_communication()

            self._initialized = True
            print(f"[UnitreeBackend] Initialized on {network_interface}")
            print("  Robot type: HG (G1)")
            print(f"  Joints: {self.num_joints}")
            return True

        except Exception as e:
            print(f"[UnitreeBackend] Init failed: {e}")
            return False

    def _init_hg_communication(self):
        """Initialize HG (humanoid/G1) DDS communication."""
        # Create publisher and subscriber
        self.low_cmd_publisher = ChannelPublisher("rt/lowcmd", LowCmdHG)
        self.low_cmd_publisher.Init()

        self.low_state_subscriber = ChannelSubscriber("rt/lowstate", LowStateHG)
        self.low_state_subscriber.Init(self._low_state_handler, 10)

        # Odometry from the onboard state estimator (base position, velocity, attitude)
        self.odom_state_subscriber = ChannelSubscriber("rt/odommodestate", SportModeState_)
        self.odom_state_subscriber.Init(self._odom_state_handler, 10)

        # Initialize low command
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        init_cmd_hg(self.low_cmd, mode_machine=0, mode_pr=MotorMode.PR)

    def _low_state_handler(self, msg):
        """Handler for low state messages."""
        with self._state_lock:
            self.low_state = msg
            self._last_state_time = time.monotonic()
            self.remote_controller.set(msg.wireless_remote)
            if self.low_cmd is not None:
                self.low_cmd.mode_machine = msg.mode_machine

    def _odom_state_handler(self, msg):
        """Handler for odometry (SportModeState) messages. A non-finite pose counts as no message (the
        stream then goes stale and the odometry gate / stale-odom latch handle it)."""
        if not (np.isfinite(msg.position).all() and np.isfinite(msg.imu_state.quaternion).all()):
            return
        with self._odom_lock:
            self.odom_state = msg
            self._last_odom_time = time.monotonic()

    def read_state(self) -> RobotState:
        """Read current robot state.

        Returns:
            RobotState with joint positions, velocities, IMU data (zeros before the first
            low state); the odom fields are None when odometry is absent or older than
            ODOM_STALE_S. The age of the low state is ``state_age_s()``.
        """
        with self._state_lock:
            if self.low_state is None:
                return RobotState(
                    joint_positions=np.zeros(self.num_joints, dtype=np.float32),
                    joint_velocities=np.zeros(self.num_joints, dtype=np.float32),
                )

            # Extract joint state
            joint_pos = np.array(
                [self.low_state.motor_state[i].q for i in range(self.num_joints)],
                dtype=np.float32
            )
            joint_vel = np.array(
                [self.low_state.motor_state[i].dq for i in range(self.num_joints)],
                dtype=np.float32
            )

            # Extract IMU data
            imu = self.low_state.imu_state
            imu_quat = np.array(imu.quaternion, dtype=np.float32)  # wxyz
            imu_ang_vel = np.array(imu.gyroscope, dtype=np.float32)

        # Odometry, if received and fresh
        odom_position = None
        odom_quaternion = None
        with self._odom_lock:
            if self.odom_state is not None and self._odom_age_locked() <= ODOM_STALE_S:
                odom_position = np.array(self.odom_state.position, dtype=np.float32)
                odom_quaternion = np.array(self.odom_state.imu_state.quaternion, dtype=np.float32)

        return RobotState(
            joint_positions=joint_pos,
            joint_velocities=joint_vel,
            imu_quaternion=imu_quat,
            imu_angular_velocity=imu_ang_vel,
            odom_position=odom_position,
            odom_quaternion=odom_quaternion,
        )

    def _odom_age_locked(self) -> float:
        return (time.monotonic() - self._last_odom_time) if self._last_odom_time > 0.0 else float("inf")

    def state_age_s(self) -> float:
        """Seconds since the last rt/lowstate message (inf before the first one)."""
        with self._state_lock:
            t = self._last_state_time
        return (time.monotonic() - t) if t > 0.0 else float("inf")

    def odom_age_s(self) -> float:
        """Seconds since the last rt/odommodestate message (inf before the first one)."""
        with self._odom_lock:
            return self._odom_age_locked()

    def odom_live(self) -> bool:
        """True if odometry arrived within ODOM_STALE_S."""
        return self.odom_age_s() <= ODOM_STALE_S

    def wait_for_odom(self, timeout: float = 2.0) -> bool:
        """Wait until odometry is live. False on timeout."""
        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout:
            if self.odom_live():
                return True
            time.sleep(0.01)
        return False

    def send_command(self, cmd: MotorCommand):
        """Send motor command to robot.

        Args:
            cmd: Motor command with positions, velocities, gains, torques
        """
        if not self._initialized or self.low_cmd is None:
            return

        # Fill motor commands
        for i in range(self.num_joints):
            self.low_cmd.motor_cmd[i].q = float(cmd.positions[i])
            self.low_cmd.motor_cmd[i].dq = float(cmd.velocities[i])
            self.low_cmd.motor_cmd[i].kp = float(cmd.kp[i])
            self.low_cmd.motor_cmd[i].kd = float(cmd.kd[i])
            self.low_cmd.motor_cmd[i].tau = float(cmd.torques[i])

        # Calculate CRC and send
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.low_cmd_publisher.Write(self.low_cmd)

    def wait_for_state(self, timeout: float = 5.0) -> bool:
        """Wait for valid robot state.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if state received, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self._state_lock:
                if self.low_state is not None:
                    return True
            time.sleep(0.01)
        return False

    def send_damping(self):
        """Send one damping command (kp = 0, kd = 8 on every motor)."""
        if self.low_cmd is not None:
            create_damping_cmd(self.low_cmd)
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.low_cmd_publisher.Write(self.low_cmd)

    def cleanup(self):
        """Set robot to safe damping state."""
        self.send_damping()

    def get_remote_controller(self) -> RemoteController:
        """Get remote controller state.

        Returns:
            RemoteController with button state
        """
        return self.remote_controller
