#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
Real Robot Web Interface for G1 Motion Tracking.

Combines G1RealController (real robot, remote controller) with
MotionTrackingWebInterface (web-based motion selection from tracker/data/*.csv).
Remote B still works for standup and interrupts the current web-selected motion.

Usage (from tracker/deploy/g1; --config is relative to this directory):
    python play_real_web_interface.py --config configs/g1_v2.yaml --policy_path POLICY.onnx \
        --motion MOTION.csv --net eth0 --port 8080
"""

import argparse
import sys
import time
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import UnitreeBackend, ControllerConfig, quat_to_euler
from utils.web_interface import MotionTrackingWebInterface
from deploy_real import G1RealController
from g1_controller import G1Controller
import variant_support

PROJ_DIR = Path(__file__).resolve().parent


class RealMotionTrackingWebInterface(MotionTrackingWebInterface):
    """Web interface adapted for real robot (reads state from UnitreeBackend)."""

    def _state_monitor_loop(self):
        """Read robot state from real backend at 20Hz."""
        while self._running:
            try:
                backend = getattr(self.controller, 'backend', None)
                if backend is None:
                    time.sleep(0.1)
                    continue

                state = backend.read_state()
                if state is None:
                    time.sleep(0.05)
                    continue

                # Orientation from IMU quaternion (wxyz)
                quat = np.array(state.imu_quaternion, dtype=np.float64)
                euler = quat_to_euler(quat)
                self._robot_state['orientation'] = {
                    'roll': float(np.degrees(euler[0])),
                    'pitch': float(np.degrees(euler[1])),
                    'yaw': float(np.degrees(euler[2])),
                }

                # Angular velocity from IMU
                ang_vel = np.array(state.imu_angular_velocity, dtype=np.float64)
                self._robot_state['base_ang_vel'] = ang_vel.tolist()

                # Linear velocity and position are not shown on the real robot
                self._robot_state['base_lin_vel'] = [0.0, 0.0, 0.0]
                self._robot_state['position'] = [0.0, 0.0, 0.0]

                # Joint states
                self._robot_state['joint_pos'] = state.joint_positions.tolist()
                self._robot_state['joint_vel'] = state.joint_velocities.tolist()

                time.sleep(0.05)  # 20Hz
            except Exception:
                time.sleep(0.1)

    def _do_standup(self):
        """Standup for real robot -- delegates to controller."""
        self.controller.do_standup()
        self._current_motion = None


def main():
    parser = argparse.ArgumentParser(
        description="G1 Real Robot Web Interface for Motion Tracking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--net", type=str, default="eth0", help="Network interface")
    parser.add_argument("--config", type=str, default="configs/g1_v2.yaml", help="Config file")
    parser.add_argument(
        "--policy_path", type=str, default="../models/g1/policy.onnx",
        help="Path to ONNX policy",
    )
    parser.add_argument("--motion", type=str, required=True, help="Initial motion file (CSV)")
    parser.add_argument("--motion_fps", type=int, default=30, help="Motion data FPS")
    parser.add_argument("--duration", type=float, default=86400.0, help="Max runtime (seconds)")
    parser.add_argument("--port", type=int, default=8080, help="Web interface port")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Web interface bind address (no authentication: bind beyond loopback only on a trusted network)")
    parser.add_argument("--allow_no_depth", action="store_true",
                        help="Run a depth policy without a camera (far-plane input; debugging only)")
    parser.add_argument("--allow_depth_clip_mismatch", action="store_true",
                        help="Accept a depth clip different from the training clip")
    args = parser.parse_args()

    # Load config
    config_path = PROJ_DIR / args.config
    config = ControllerConfig(str(config_path))
    config.recording_duration = args.duration

    print("\n[G1 Real Robot - Web Interface]")
    print(f"  Config: {config_path}")
    print(f"  Policy: {args.policy_path}")
    print(f"  Motion: {args.motion}")
    print(f"  Network: {args.net}")
    print(f"  Web UI: http://{args.host}:{args.port}")
    print()

    # Connect to real robot
    backend = UnitreeBackend(config)
    if not backend.init(network_interface=args.net):
        print(f"[ERROR] Failed to connect via {args.net}")
        return 1

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

    # Motion CSVs offered by the web interface: tracker/data/*.csv
    data_dir = str(Path(__file__).resolve().parent.parent.parent / "data")

    # Start controller in background thread
    controller_thread = threading.Thread(
        target=controller.run_real,
        kwargs={"timeout": args.duration},
        daemon=True,
    )
    controller_thread.start()

    print()
    print("Controls:")
    print("  Joystick Start: Move to default position")
    print("  Joystick A: Begin control / replay original motion")
    print("  Joystick B: Standup (interrupts current motion)")
    print("  Joystick Select: Emergency stop")
    print("  Web UI: Select motions, standup, view telemetry")
    print()

    # Create and start web interface
    web = RealMotionTrackingWebInterface(
        controller, robot_name="G1 (Real)", port=args.port,
        data_dir=data_dir, initial_motion=args.motion,
        default_motion_fps=args.motion_fps, host=args.host,
    )
    print(f"[Web] G1 Real Robot Control: http://{args.host}:{args.port}")
    try:
        web.start(threaded=False)  # Block on Flask
    except KeyboardInterrupt:
        pass
    finally:
        # werkzeug's serve_forever swallows Ctrl-C and returns normally, and a
        # daemon thread killed at interpreter exit never runs its finally block
        # (cleanup -> damping). Always stop the controller, wait, and damp.
        controller.robot_running = False
        web.stop()
        controller_thread.join(timeout=5.0)
        controller.cleanup()

    return 0


if __name__ == "__main__":
    exit(main())
