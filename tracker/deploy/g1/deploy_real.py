#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
Real-robot deployment of the G1 tracker.

Uses G1RealController (extends G1Controller) with UnitreeBackend and encapsulates
the real-robot startup sequence (wait -> move to default pose -> stand up / track).

A depth policy (config with an image group, e.g. configs/g1_v2_depth.yaml) reads the RealSense D435i
in-process (pyrealsense2), so run it on the computer the camera is attached to.

Usage (from tracker/deploy/g1; --config is relative to this directory):
    python deploy_real.py --config configs/g1_v2.yaml --policy_path POLICY.onnx --motion MOTION.csv --net eth0
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    UnitreeBackend, ControllerConfig,
    KeyMap, ControlState, MotorCommand, quat_to_euler,
)
from utils.backends.unitree_backend import ODOM_STALE_S
from utils.core.base_controller import all_finite
from g1_controller import G1Controller
import variant_support

PROJ_DIR = Path(__file__).resolve().parent


# Robot state older than this many control periods is stale (the C++ SafetyMonitor uses the same limit).
STATE_STALE_PERIODS = 3


class G1RealController(G1Controller):
    """G1 controller for real robot deployment.

    Extends G1Controller with a remote-controller state machine:
    - INIT: wait for Start, then move (blocking 2 s cosine ramp) to the init pose
      (frame 0 of the loaded motion)
    - DEFAULT_POS: hold the init pose; B = stand up, then CONTROL; A = CONTROL on
      the loaded motion (aligned to the robot, with a smooth transition)
    - CONTROL: run the policy via run_real_step(); B = stand up, A = replay motion
    Select is an emergency stop in every state, including the move to the init pose.

    Safety (same rules and limits as the C++ deployment), checked every loop iteration
    by ``safety_check``: robot state older than STATE_STALE_PERIODS control periods,
    a non-finite robot state, or (in CONTROL) odometry older than ODOM_STALE_S latch
    damping and end the run. CONTROL is only entered with live odometry and a fresh
    depth frame (depth policy); a refused press must be pressed again. In CONTROL, depth going stale (the source then feeds the far plane)
    and recovering are logged.
    """

    def __init__(self, config, policy_path: str, backend,
                 motion_loader=None, depth_source=None):
        super().__init__(config, policy_path, backend, motion_loader,
                         depth_source=depth_source)

        self.state = ControlState.INIT
        self.move_duration = 2.0
        self._state_stale_s = STATE_STALE_PERIODS * float(config.control_dt)
        self._depth_live = True       # depth freshness last reported in CONTROL

        # Pre-allocate zero arrays for MotorCommand fields
        self._zeros_f32 = np.zeros(config.num_joints, dtype=np.float32)
        self._zeros_i32 = np.zeros(config.num_joints, dtype=np.int32)

        self._original_qpos_data = None  # Raw CSV data for replay
        self._original_input_fps = None  # Set by store_original_motion()

        # Button state for rising-edge detection
        self._prev_b_button = 0
        self._prev_a_button = 0

    # ── safety ────────────────────────────────────────────────────────────────

    def _safety_fault(self, in_control: bool):
        """The first failing safety check as a message, or None."""
        age = self.backend.state_age_s()
        if age > self._state_stale_s:
            return (f"robot state stale ({age * 1e3:.0f} ms old, limit "
                    f"{self._state_stale_s * 1e3:.0f} ms)")
        state = self.backend.read_state()
        if not all_finite(state.joint_positions, state.joint_velocities,
                          state.imu_quaternion, state.imu_angular_velocity):
            return "non-finite robot state"
        if in_control and not self.backend.odom_live():
            return (f"odometry stale ({self.backend.odom_age_s() * 1e3:.0f} ms old, limit "
                    f"{ODOM_STALE_S * 1e3:.0f} ms)")
        return None

    def safety_check(self, in_control: bool) -> bool:
        """Run the safety checks once. False = a fault is latched (damping sent); stop the loop."""
        if self.fault is not None:
            return False
        fault = self._safety_fault(in_control)
        if fault is not None:
            self.latch_damping(fault)
            return False
        return True

    def control_entry_blocker(self):
        """Why CONTROL cannot be entered now (a message), or None."""
        if not self.backend.odom_live():
            return f"no live odometry (anchor_mode=odom; last message {self.backend.odom_age_s():.2f} s ago)"
        if self._needs_depth and self.depth_source is not None and not self.depth_source.is_fresh():
            return "no fresh depth frame (the policy would see the far plane, an empty world)"
        return None

    def try_enter_control(self, install) -> bool:
        """Install a reference with ``install()`` and enter CONTROL, unless a blocker refuses it.

        Called on a rising edge, so a refused press does nothing until the button is pressed again.
        """
        reason = self.control_entry_blocker()
        if reason is not None:
            print(f"[INFO] Not entering CONTROL: {reason}. Press again once it is live.")
            return False
        if not install():
            print("[WARN] Reference install failed; staying in DEFAULT_POS")
            return False
        self._depth_live = True
        self.state = ControlState.CONTROL
        return True

    def _watch_depth(self):
        """Log depth going stale / recovering while in CONTROL (once per transition)."""
        if not self._needs_depth or self.depth_source is None:
            return
        live = self.depth_source.is_fresh()
        if live != self._depth_live:
            self._depth_live = live
            if live:
                print("[INFO] depth recovered")
            else:
                print("[WARN] depth STALE: feeding the far plane (empty world) until the camera recovers")

    # ── motions ───────────────────────────────────────────────────────────────

    def move_to_default_pos(self) -> bool:
        """Blocking cosine interpolation from current position to init_joint_pos.

        Uses reduced gains (kps * 0.5) during the transition for safety.
        Duration is self.move_duration seconds. Select, a web stop or a safety
        fault aborts it; returns False then.
        """
        state = self.backend.read_state()
        start_pos = state.joint_positions.copy()
        target_pos = self.init_joint_pos.copy()
        move_steps = int(self.move_duration / self.config.control_dt)

        for step in range(move_steps):
            if self.backend.get_remote_controller().button[KeyMap.select] or not self.robot_running:
                print("[INFO] Emergency stop during the move to the start pose")
                self.robot_running = False
                return False
            if not self.safety_check(in_control=False):
                return False
            alpha = step / move_steps
            smooth_alpha = 0.5 * (1 - np.cos(alpha * np.pi))
            interp_pos = start_pos + smooth_alpha * (target_pos - start_pos)

            cmd = MotorCommand(
                positions=interp_pos,
                velocities=self._zeros_f32,
                kp=self.joint_kps * 0.5,
                kd=self.joint_kds,
                torques=self._zeros_f32,
                modes=self._zeros_i32,
            )
            self.backend.send_command(cmd)
            time.sleep(self.config.control_dt)
        return True

    def hold_default_pos(self):
        """Send one MotorCommand holding init_joint_pos with full gains."""
        cmd = MotorCommand(
            positions=self.init_joint_pos,
            velocities=self._zeros_f32,
            kp=self.joint_kps,
            kd=self.joint_kds,
            torques=self._zeros_f32,
            modes=self._zeros_i32,
        )
        self.backend.send_command(cmd)

    def store_original_motion(self, motion_file: str, input_fps: int):
        """Load raw CSV motion data for replay after standup.

        Args:
            motion_file: Path to the motion CSV file.
            input_fps: Original FPS of the CSV file.
        """
        self._original_qpos_data = np.loadtxt(motion_file, delimiter=",", ndmin=2)
        self._original_input_fps = input_fps
        print(f"[INFO] Stored original motion for replay: {self._original_qpos_data.shape}")

    def do_standup(self) -> bool:
        """Transition robot to standing pose using send_ref_motion. True if installed."""
        if self.motion_loader is None:
            return False

        # Current IMU quaternion (wxyz)
        state = self.backend.read_state()
        current_quat_wxyz = np.array(state.imu_quaternion, dtype=np.float64)

        # Extract yaw only — discard pitch/roll so standup targets upright
        yaw = quat_to_euler(current_quat_wxyz)[2]  # [roll, pitch, yaw]
        # Build upright quaternion with only yaw: wxyz = [cos(yaw/2), 0, 0, sin(yaw/2)]
        upright_quat_wxyz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float64)
        current_quat_xyzw = upright_quat_wxyz[[1, 2, 3, 0]]  # wxyz -> xyzw

        # Standing base position from the config; send_ref_motion rebases it onto the
        # robot's odometry pose (align_anchor="robot").
        ml = self.motion_loader
        base_pos = ml.standing_base_pos.copy().astype(np.float64)

        # standing_joint_pos is in deployment order (matches qpos_data format for send_ref_motion)
        standing_joints_deploy = ml.standing_joint_pos.copy()

        # Build synthetic qpos_data: shape (1, 36) = [xyz(3), quat_xyzw(4), joints_deploy(29)]
        qpos_data = np.zeros((1, 3 + 4 + len(standing_joints_deploy)), dtype=np.float64)
        qpos_data[0, 0:3] = base_pos
        qpos_data[0, 3:7] = current_quat_xyzw
        qpos_data[0, 7:] = standing_joints_deploy

        ok = self.send_ref_motion(qpos_data)
        if ok:
            print("[INFO] Transitioning to standing pose")
        return ok

    def replay_original_motion(self) -> bool:
        """Play the original loaded motion from its start (aligned to the robot, smooth transition)."""
        if self._original_qpos_data is None:
            print("[WARN] No original motion stored for replay")
            return False
        ok = self.send_ref_motion(self._original_qpos_data, input_fps=self._original_input_fps)
        if ok:
            print("[INFO] Playing original motion")
        return ok

    def run_real(self, timeout: float = 300.0):
        """Main state machine loop for real robot deployment.

        Args:
            timeout: Maximum runtime in seconds.
        """
        self.recording_duration = timeout
        self._start_time = time.time()

        try:
            while (time.time() - self._start_time) < timeout:
                remote = self.backend.get_remote_controller()

                # SELECT = emergency stop in any state
                if remote.button[KeyMap.select]:
                    print("[INFO] Emergency stop triggered")
                    self.robot_running = False
                    break
                # Web UI emergency stop / damping clears robot_running; honour it in
                # every state, not only in CONTROL (_should_continue).
                if not self.robot_running:
                    print("[INFO] Stop requested")
                    break
                if not self.safety_check(in_control=self.state == ControlState.CONTROL):
                    break

                # Rising edges of A and B, tracked in every state.
                a_down, b_down = bool(remote.button[KeyMap.A]), bool(remote.button[KeyMap.B])
                a_edge, b_edge = a_down and not self._prev_a_button, b_down and not self._prev_b_button
                self._prev_a_button, self._prev_b_button = int(a_down), int(b_down)

                if self.state == ControlState.INIT:
                    if remote.button[KeyMap.start]:
                        print("[INFO] Moving to the motion's first frame...")
                        if not self.move_to_default_pos():
                            break
                        self.state = ControlState.DEFAULT_POS
                        print("[INFO] At the start pose. Press B to stand up or A to start the motion...")
                    else:
                        time.sleep(self.config.control_dt)

                elif self.state == ControlState.DEFAULT_POS:
                    self.hold_default_pos()
                    # B: stand up first (safer than committing straight to motion
                    # tracking). do_standup installs a standing-pose reference via
                    # send_ref_motion (rebased onto the robot pose); the policy then
                    # drives the robot to standing on the next CONTROL ticks.
                    if b_edge:
                        print("[INFO] Standing up before motion tracking...")
                        if self.try_enter_control(self.do_standup):
                            print("[INFO] In CONTROL with standing motion. "
                                  "Press A when ready to replay original motion.")
                    # A: start motion tracking on the loaded motion, installed now
                    # (aligned to the robot, with a smooth transition from its pose).
                    elif a_edge:
                        print("[INFO] Starting motion tracking control...")
                        self.try_enter_control(self.replay_original_motion)
                    time.sleep(self.config.control_dt)

                elif self.state == ControlState.CONTROL:
                    if b_edge:
                        self.do_standup()
                    if a_edge:
                        self.replay_original_motion()
                    self._watch_depth()
                    if not self._should_continue() or not self.run_real_step():
                        break

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.print_stats()
            self.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="G1 real-robot deployment of the motion tracker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--net", type=str, default="eth0", help="Network interface")
    parser.add_argument("--config", type=str, default="configs/g1_v2.yaml", help="Config file")
    parser.add_argument("--policy_path", type=str, default='../models/g1/policy.onnx', help="Path to ONNX policy")
    parser.add_argument("--motion", type=str, required=True, help="Motion data file (CSV)")
    parser.add_argument("--motion_fps", type=int, default=30, help="Motion data FPS")
    parser.add_argument("--duration", type=float, default=3000.0, help="Max runtime (seconds)")
    parser.add_argument("--allow_no_depth", action="store_true",
                        help="Run a depth policy without a camera (far-plane input; debugging only)")
    parser.add_argument("--allow_depth_clip_mismatch", action="store_true",
                        help="Accept a depth clip different from the training clip")
    args = parser.parse_args()

    # Load config
    config_path = PROJ_DIR / args.config
    config = ControllerConfig(str(config_path))
    config.recording_duration = args.duration

    print("\n[G1 Real Robot - Motion Tracking]")
    print(f"  Config: {config_path}")
    print(f"  Policy: {args.policy_path}")
    print(f"  Motion: {args.motion}")
    print(f"  Network: {args.net}")
    print()

    print("[SAFETY] Make sure:")
    print("  1. Robot is in a safe, open area")
    print("  2. Emergency stop is within reach")
    print("  3. Policy has been tested in MuJoCo first")
    print()

    # Create backend
    backend = UnitreeBackend(config)
    if not backend.init(network_interface=args.net):
        print(f"[ERROR] Failed to connect via {args.net}")
        return 1

    # Wait for state
    print("[INFO] Waiting for robot state...")
    if not backend.wait_for_state(timeout=10.0):
        print("[ERROR] No state received from robot")
        return 1

    # Observation-variant dependencies (depth camera); refuses to start on a missing one.
    setup = variant_support.setup_real(config, backend, allow_no_depth=args.allow_no_depth,
                                       allow_depth_clip_mismatch=args.allow_depth_clip_mismatch)
    if setup is None:
        backend.cleanup()
        return 1
    _, depth_source = setup

    motion_loader = G1Controller.create_motion_loader(
        config, args.motion, input_fps=args.motion_fps, duration=args.duration
    )

    # Create controller
    controller = G1RealController(
        config, args.policy_path, backend, motion_loader,
        depth_source=depth_source,
    )
    controller.store_original_motion(motion_loader.motion_file, args.motion_fps)

    print()
    print("Controls:")
    print("  Start:  Move robot to the motion's first frame")
    print("  B:      DEFAULT_POS -> stand up first (safer), then enter CONTROL")
    print("          CONTROL    -> standup interrupt mid-run")
    print("  A:      DEFAULT_POS -> enter CONTROL on the motion (aligned, smooth transition)")
    print("          CONTROL    -> replay original motion")
    print("  Select: Emergency stop")
    print()
    print("[INFO] Recommended sequence: Start -> wait for the start pose ->")
    print("       B (verify standup) -> A (commit to motion).")
    print()

    print("[INFO] Press Start to begin (robot will move to the motion's first frame)...")
    controller.run_real(timeout=args.duration)  # cleanup (damping, depth source) in its finally

    print("[INFO] Deployment completed")
    return 0


if __name__ == "__main__":
    exit(main())
