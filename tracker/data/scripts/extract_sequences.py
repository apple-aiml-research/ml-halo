#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Builds on BeyondMimic whole_body_tracking (Copyright (c) 2024, The Isaac Lab Project Developers, MIT License).

"""Replay CSV motion clips in Isaac Lab and write training-ready NPZs (one Isaac Sim launch for all of them).

Input layout (one frame per CSV row):
    ``[x, y, z, qx, qy, qz, qw, joint_0, ..., joint_28]``

Output NPZ keys (the format ``mdp.MotionCommand`` consumes):
    joint_pos, joint_vel,
    body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w,
    root_pos, root_quat, root_gravity,
    root_lin_vel_w, root_ang_vel_w, root_lin_vel_b, root_ang_vel_b,
    body_names, joint_names, fps

Usage:
    python extract_sequences.py --input data/lafan1/g1/walk1_subject1.csv \
                                --output data/g1_processed/robot_only/lafan/walk1_subject1.npz \
                                --headless

``--input`` / ``--output`` take several paths (pairwise). Use ``batch_process.py`` to convert a folder of CSVs.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# CLI
parser = argparse.ArgumentParser(description="Replay CSV motion clips and dump training NPZs.")
parser.add_argument("--input", type=str, nargs="+", required=True, help="Input .csv motion clip(s).")
parser.add_argument("--output", type=str, nargs="+", required=True, help="Output .npz file(s), one per input.")
parser.add_argument("--input_fps", type=int, default=30, help="FPS of the input CSV motion.")
parser.add_argument("--output_fps", type=int, default=50, help="FPS of the output NPZ motion.")
parser.add_argument(
    "--frame_range", nargs=2, type=int, metavar=("START", "END"),
    help="1-indexed inclusive frame range. Default: all frames.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if len(args_cli.input) != len(args_cli.output):
    parser.error(f"{len(args_cli.input)} --input paths but {len(args_cli.output)} --output paths")
for _path in args_cli.input:
    if not os.path.isfile(_path):
        parser.error(f"--input does not exist: {_path}")
for _path in args_cli.output:
    os.makedirs(os.path.dirname(os.path.abspath(_path)), exist_ok=True)

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Heavy imports (require sim to be live)
import torch
from scipy.spatial.transform import Rotation as R

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp


# Load the training robot's config without pulling in the whole training package (which registers the
# gym tasks and imports Isaac-Sim-only modules at top level). Clips are replayed on the training robot so the
# recorded body poses use its exact kinematics.
_G1_PKG_DIR = (
    PROJECT_ROOT / "training" / "source" / "halo_tracker" / "halo_tracker"
    / "tasks" / "manager_based" / "tracking" / "g1"
)
_g1_pkg = types.ModuleType("_g1_pkg")
_g1_pkg.ASSETS_DIR = PROJECT_ROOT.parent / "assets"
sys.modules["_g1_pkg"] = _g1_pkg
_spec = importlib.util.spec_from_file_location(
    "_g1_pkg.robot_cfg", _G1_PKG_DIR / "robot_cfg.py", submodule_search_locations=[]
)
_robot_cfg_module = importlib.util.module_from_spec(_spec)
_robot_cfg_module.__package__ = "_g1_pkg"
sys.modules["_g1_pkg.robot_cfg"] = _robot_cfg_module
_spec.loader.exec_module(_robot_cfg_module)
ROBOT_CFG = _robot_cfg_module.get_g1_sysid_cfg()

# Joint order in the input CSV — matches the data-side retargeting convention.
CSV_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Minimal scene for kinematic motion replay."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    """Load a CSV motion clip, interpolate to ``output_fps``, and compute velocities.

    All quaternions are stored in wxyz (Isaac Lab convention).
    """

    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None = None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / input_fps
        self.output_dt = 1.0 / output_fps
        self.device = device
        self.frame_range = frame_range
        self.current_idx = 0

        self._load_csv()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_csv(self):
        """CSV layout: ``[x, y, z, qx, qy, qz, qw, joint_0..joint_28]``."""
        if self.frame_range is None:
            motion = np.loadtxt(self.motion_file, delimiter=",")
        else:
            motion = np.loadtxt(
                self.motion_file,
                delimiter=",",
                skiprows=self.frame_range[0] - 1,
                max_rows=self.frame_range[1] - self.frame_range[0] + 1,
            )
        motion = torch.from_numpy(motion).to(torch.float32).to(self.device)

        self.motion_base_poss_input = motion[:, :3]
        # CSV stores xyzw; Isaac Lab uses wxyz.
        self.motion_base_rots_input = motion[:, 3:7][:, [3, 0, 1, 2]]
        self.motion_dof_poss_input = motion[:, 7:]

        # quat_slerp downstream needs unit quats.
        self.motion_base_rots_input = self.motion_base_rots_input / torch.clamp(
            self.motion_base_rots_input.norm(dim=-1, keepdim=True), min=1e-8
        )

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"[MotionLoader] {self.motion_file}: frames={self.input_frames}, fps={self.input_fps}")

    def _interpolate_motion(self):
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        idx0, idx1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[idx0], self.motion_base_poss_input[idx1], blend.unsqueeze(1)
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[idx0], self.motion_base_rots_input[idx1], blend
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[idx0], self.motion_dof_poss_input[idx1], blend.unsqueeze(1)
        )
        print(
            f"[MotionLoader] interpolated: "
            f"{self.input_frames}@{self.input_fps}fps -> "
            f"{self.output_frames}@{self.output_fps}fps"
        )

    @staticmethod
    def _lerp(a, b, blend):
        return a * (1 - blend) + b * blend

    @staticmethod
    def _slerp(a, b, blend):
        out = torch.zeros_like(a)
        for i in range(a.shape[0]):
            out[i] = quat_slerp(a[i], b[i], blend[i])
        return out

    def _compute_frame_blend(self, times):
        phase = times / self.duration
        idx0 = (phase * (self.input_frames - 1)).floor().long()
        idx1 = torch.minimum(idx0 + 1, torch.tensor(self.input_frames - 1))
        return idx0, idx1, phase * (self.input_frames - 1) - idx0

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    @staticmethod
    def _so3_derivative(rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """SO(3) finite-difference angular velocity (B, 3) from quaternions (B, 4)."""
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        # Repeat first/last so output length matches input length.
        return torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    def get_next_state(self):
        i = self.current_idx
        state = (
            self.motion_base_poss[i : i + 1],
            self.motion_base_rots[i : i + 1],
            self.motion_base_lin_vels[i : i + 1],
            self.motion_base_ang_vels[i : i + 1],
            self.motion_dof_poss[i : i + 1],
            self.motion_dof_vels[i : i + 1],
        )
        self.current_idx += 1
        reset = self.current_idx >= self.output_frames
        if reset:
            self.current_idx = 0
        return state, reset


def replay_clip(sim, scene, robot, joint_indexes, input_csv: str, output_npz: str) -> None:
    """Replay one CSV clip on the robot and write its NPZ."""
    motion = MotionLoader(
        motion_file=input_csv,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
    )

    log: dict[str, list | np.ndarray | int] = {
        "fps": args_cli.output_fps,
        "joint_pos": [], "joint_vel": [],
        "body_pos_w": [], "body_quat_w": [],
        "body_lin_vel_w": [], "body_ang_vel_w": [],
        "root_pos": [], "root_quat": [], "root_gravity": [],
        "root_lin_vel_w": [], "root_ang_vel_w": [],
        "root_lin_vel_b": [], "root_ang_vel_b": [],
    }
    env_orig = np.array([scene.env_origins[0, 0].item(), scene.env_origins[0, 1].item(), 0.0])

    saved = False
    while simulation_app.is_running() and not saved:
        state, reset_flag = motion.get_next_state()
        base_pos, base_rot, base_lin_vel, base_ang_vel, dof_pos, dof_vel = state

        # Drive root + joint state kinematically (no physics step).
        root = robot.data.default_root_state.clone()
        root[:, :3] = base_pos
        root[:, :2] += scene.env_origins[:, :2]
        root[:, 3:7] = base_rot
        root[:, 7:10] = base_lin_vel
        root[:, 10:] = base_ang_vel
        robot.write_root_state_to_sim(root)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, joint_indexes] = dof_pos
        joint_vel[:, joint_indexes] = dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()
        scene.update(sim.get_physics_dt())

        cam_target = root[0, :3].cpu().numpy()
        sim.set_camera_view(cam_target + np.array([2.0, 2.0, 0.5]), cam_target)

        # Per-frame log
        log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())

        body_pos_w = robot.data.body_pos_w[0].cpu().numpy().copy() - env_orig
        log["body_pos_w"].append(body_pos_w)
        log["body_quat_w"].append(robot.data.body_quat_w[0].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0].cpu().numpy().copy())

        root_pos_np = base_pos[0].cpu().numpy().copy()
        root_quat_np = base_rot[0].cpu().numpy().copy()  # wxyz
        log["root_pos"].append(root_pos_np)
        log["root_quat"].append(root_quat_np)

        # Body-frame quantities (wxyz -> xyzw for scipy).
        rot = R.from_quat(np.concatenate([root_quat_np[1:], root_quat_np[0:1]]))
        rot_inv = rot.inv()
        log["root_gravity"].append(rot.apply(np.array([0.0, 0.0, -1.0]), inverse=True))

        lin_vel_w = base_lin_vel[0].cpu().numpy()
        ang_vel_w = base_ang_vel[0].cpu().numpy()
        log["root_lin_vel_w"].append(lin_vel_w.copy())
        log["root_ang_vel_w"].append(ang_vel_w.copy())
        log["root_lin_vel_b"].append(rot_inv.apply(lin_vel_w))
        log["root_ang_vel_b"].append(rot_inv.apply(ang_vel_w))

        if reset_flag:
            saved = True
            for k, v in list(log.items()):
                if isinstance(v, list):
                    log[k] = np.stack(v, axis=0)
            log["body_names"] = np.array(list(robot.data.body_names))
            log["joint_names"] = np.array(list(robot.data.joint_names))

            np.savez(output_npz, **log)
            print(f"[INFO] saved {output_npz}")
            for k, v in log.items():
                if isinstance(v, np.ndarray):
                    print(f"       {k}: {v.shape}")


def main():
    """Replay every ``--input`` CSV and write the matching ``--output`` NPZ, in one simulation."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    joint_indexes = robot.find_joints(CSV_JOINT_NAMES, preserve_order=True)[0]

    for input_csv, output_npz in zip(args_cli.input, args_cli.output):
        replay_clip(sim, scene, robot, joint_indexes, input_csv, output_npz)


if __name__ == "__main__":
    # Exit right after the NPZs are written: Isaac Sim's teardown (simulation_app.close()) can hang
    # indefinitely after a headless replay, which would stall a batch run on its first clip.
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
