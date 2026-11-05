#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
MuJoCo sim-to-sim deployment of the G1 tracker.

Runs G1Controller with SimulatorBackend; the run ends when the viewer is closed
or --duration elapses.

A config whose ``observation:`` block has an image group (the depth student, e.g.
configs/g1_v2_depth.yaml) gets the training depth camera injected from its ``depth_camera:``
block and rendered on the control thread.

Usage (from tracker/deploy/g1; --config is relative to this directory):
    python deploy_mujoco.py --config configs/g1_v2.yaml --policy_path POLICY.onnx --motion MOTION.csv
    python deploy_mujoco.py --config configs/g1_v2_depth.yaml --policy_path POLICY.onnx --motion MOTION.csv
    python deploy_mujoco.py --config configs/g1_v2.yaml --policy_path POLICY.onnx --motion MOTION.csv --web --port 8080
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import SimulatorBackend, ControllerConfig
from utils.core.observation_builder import ObservationSpec
from g1_controller import G1Controller
import variant_support


def main():
    parser = argparse.ArgumentParser(description="G1 MuJoCo Deployment")
    parser.add_argument("--policy_path", type=str, default='../models/g1/policy.onnx', help="Path to ONNX policy")
    parser.add_argument("--config", type=str, default="configs/g1_v2.yaml", help="Config file")
    parser.add_argument("--motion", type=str, default=None,
                        help="Motion CSV (path, or a name in tracker/data/); default: standing pose")
    parser.add_argument("--motion_fps", type=int, default=30, help="Motion data FPS")
    parser.add_argument("--headless", action="store_true", help="Run without viewer")
    parser.add_argument("--duration", type=float, default=3000.0, help="Max runtime (seconds)")
    parser.add_argument("--web", action="store_true", help="Enable web control interface")
    parser.add_argument("--port", type=int, default=8080, help="Web interface port")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Web interface bind address (no authentication: bind beyond loopback only on a trusted network)")
    parser.add_argument("--allow_no_depth", action="store_true",
                        help="Run a depth policy without a depth camera (far-plane input; debugging only)")
    parser.add_argument("--allow_depth_clip_mismatch", action="store_true",
                        help="Accept a depth clip different from the training clip")
    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent / args.config
    config = ControllerConfig(str(config_path))
    config.recording_duration = args.duration

    print("[G1 MuJoCo]")
    print(f"  Config: {config_path}")
    print(f"  Policy: {args.policy_path}")
    if args.web:
        print(f"  Web UI: http://{args.host}:{args.port}")

    # Create simulator backend. A depth policy needs the training depth camera in the model.
    spec = ObservationSpec.from_dict(config.observation)
    backend = SimulatorBackend(config)
    depth_source = None
    if variant_support.image_group(spec) is not None and getattr(config, "depth_camera", None):
        model, data, rig = variant_support.compile_scene_with_depth_camera(config.xml_path, config.depth_camera)
        if not backend.init_from_model(model, data, headless=args.headless):
            print("[ERROR] Failed to initialize MuJoCo")
            return 1
        depth_source = variant_support.make_mujoco_depth_source(spec, model, data, rig["name"])
        print(f"  Depth camera: '{rig['name']}' fovy={rig['fovy']} on {rig['parent_body']} at {rig['pos']}")
    elif not backend.init(headless=args.headless):
        print("[ERROR] Failed to initialize MuJoCo")
        return 1

    # The MuJoCo depth renderer is pinned to the thread that renders first (the control loop), so the
    # preflight checks the source's geometry without reading a frame.
    if not variant_support.preflight(spec, backend=backend, depth_source=depth_source,
                                     allow_no_depth=args.allow_no_depth,
                                     allow_depth_clip_mismatch=args.allow_depth_clip_mismatch,
                                     read_depth_frame=False):
        return 1

    # Set initial pose
    backend.set_initial_pose(joint_angles=config.default_angles)

    # Motion CSVs offered by the web interface: tracker/data/*.csv
    data_dir = str(Path(__file__).resolve().parent.parent.parent / 'data')

    # Load motion (standing pose if no motion specified)
    motion_loader = G1Controller.create_motion_loader(
        config, args.motion, input_fps=args.motion_fps
    )

    # Create controller (shared G1Controller works for both MuJoCo and Real)
    controller = G1Controller(
        config, args.policy_path, backend, motion_loader,
        depth_source=depth_source,
    )

    # Re-route the loaded motion through send_ref_motion so it gets SE(2)+Z-aligned to the
    # robot's spawn pose (the controller's __init__ installs the clip unaligned).
    # Not needed for the standing pose (no --motion given).
    if args.motion is not None:
        base_pos      = motion_loader.motion_base_poss.copy()
        quat_wxyz     = motion_loader.motion_base_rots.copy()
        joints_deploy = motion_loader.motion_dof_poss[:, controller.train_to_deploy_map].copy()
        quat_xyzw     = quat_wxyz[:, [1, 2, 3, 0]]
        qpos_data     = np.concatenate([base_pos, quat_xyzw, joints_deploy], axis=1)
        controller.send_ref_motion(
            qpos_data,
            input_fps=motion_loader.output_fps,
            enable_smooth_transition=False,
        )
        print("[deploy_mujoco] Loaded motion re-aligned via send_ref_motion "
              "(robot stays at spawn; motion translates/rotates to match).")

    if args.web:
        from utils.web_interface import MotionTrackingWebInterface

        controller_thread = threading.Thread(target=controller.run_sim, daemon=True)
        controller_thread.start()
        time.sleep(1.0)

        web = MotionTrackingWebInterface(
            controller, robot_name="G1", port=args.port,
            data_dir=data_dir, initial_motion=args.motion,
            default_motion_fps=args.motion_fps, host=args.host,
        )
        print(f"\n[Web] G1 Control: http://{args.host}:{args.port}")
        try:
            web.start(threaded=False)  # Block on Flask
        except KeyboardInterrupt:
            controller.robot_running = False
            web.stop()
    else:
        print()
        print("Controls: Close viewer window to exit")
        print()

        # Run
        try:
            controller.run_sim()
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
