#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Browser-based viewer for a G1 CSV motion clip (no Isaac Sim needed).

Uses viser + yourdfpy to replay a clip in the CSV layout of ``extract_sequences.py``
(``x, y, z, qx, qy, qz, qw`` + 29 joint angles in the order of ``CSV_JOINT_NAMES``).

Requirements:
    pip install viser yourdfpy trimesh numpy

Usage:
    python visualize_motion_web.py --csv motion_plan.csv
    python visualize_motion_web.py --csv motion_plan.csv --fps 30 --loop
"""

import argparse
import time
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(description="Visualize motion plan CSV in browser via viser.")
parser.add_argument("--csv", type=str, required=True, help="Path to CSV (N x 36: xyz, quat_xyzw, joints_deploy)")
parser.add_argument("--urdf", type=str, default=None, help="Path to URDF (auto-detected if not set)")
parser.add_argument("--fps", type=int, default=30, help="Playback FPS (default: 30)")
parser.add_argument("--speed", type=float, default=1.0, help="Speed multiplier")
parser.add_argument("--loop", action="store_true", help="Loop playback")
parser.add_argument("--port", type=int, default=8080, help="Server port")
args = parser.parse_args()

ROOT_DIR = Path(__file__).resolve().parents[2]

if args.urdf is None:
    args.urdf = str(ROOT_DIR.parent / "assets" / "robots" / "g1" / "urdf" / "g1_29dof_rev_1_0.urdf")

# Joint order of the CSV columns; the same list as extract_sequences.CSV_JOINT_NAMES (not imported: that script
# launches Isaac Sim on import).
CSV_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

import viser
import yourdfpy
from viser.extras import ViserUrdf


def main():
    data = np.genfromtxt(args.csv, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    n_frames, n_cols = data.shape
    assert n_cols >= 36, f"Expected >= 36 columns, got {n_cols}"
    print(f"[INFO] Loaded {n_frames} frames from {args.csv}")

    dt = 1.0 / args.fps
    print(f"[INFO] Playback: {args.fps} FPS, speed={args.speed}x")

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"[INFO] Open browser at: http://localhost:{args.port}")

    server.scene.add_grid("/ground", width=10, height=10, cell_size=0.5)

    urdf_model = yourdfpy.URDF.load(args.urdf)
    robot_vis = ViserUrdf(server, urdf_model, root_node_name="/robot")
    print(f"[INFO] URDF loaded: {args.urdf}")

    playing = True
    frame_idx = 0

    with server.gui.add_folder("Controls"):
        play_button = server.gui.add_button("Play/Pause")
        speed_slider = server.gui.add_slider("Speed", min=0.1, max=5.0, step=0.1, initial_value=args.speed)
        frame_slider = server.gui.add_slider("Frame", min=0, max=n_frames - 1, step=1, initial_value=0)

    @play_button.on_click
    def _(_):
        nonlocal playing
        playing = not playing

    @frame_slider.on_update
    def _(event):
        nonlocal frame_idx
        if not playing:
            frame_idx = int(event.target.value)
            render_frame(frame_idx)

    origin_x = data[0, 0]
    origin_y = data[0, 1]

    def render_frame(idx):
        row = data[idx]
        # Position: xyz (columns 0-2), centered at origin
        pos = np.array([row[0] - origin_x, row[1] - origin_y, row[2]])
        # Quaternion: CSV is xyzw, viser wants wxyz
        quat_xyzw = row[3:7]
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        server.scene.add_frame("/robot", position=pos, wxyz=quat_wxyz, show_axes=False)

        # Joint configuration
        joints = row[7:36]
        cfg = {}
        for i, name in enumerate(CSV_JOINT_NAMES):
            cfg[name] = float(joints[i])
        robot_vis.update_cfg(cfg)

    print("[INFO] Playing... Press Ctrl+C to stop.")
    try:
        while True:
            if playing:
                render_frame(frame_idx)
                frame_slider.value = frame_idx

                frame_idx += 1
                if frame_idx >= n_frames:
                    if args.loop:
                        frame_idx = 0
                    else:
                        playing = False
                        frame_idx = n_frames - 1

                time.sleep(dt / speed_slider.value)
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")


if __name__ == "__main__":
    main()
