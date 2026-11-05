#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Isaac kinematic replay: pose the G1 per frame, render OAK + TPV into frame sinks, write traj.npz."""

from __future__ import annotations

import math
import os

import numpy as np
import torch

from render.cameras import OAK_CAM_PITCH_DOWN_DEG, OAK_CAM_ROT, OAK_OFFSET_FROM_TORSO

# Kimodo / MuJoCo joint order of qpos[:, 7:36].
MUJOCO_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]  # fmt: skip
LAYOUT_HEIGHT = 10.0


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion product (wxyz)."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])  # fmt: skip


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` by the wxyz quaternion ``q``."""
    q_vec = q[1:4]
    t = 2.0 * torch.cross(q_vec, v)
    return v + q[0] * t + torch.cross(q_vec, t)


def update_oak_camera(oak_cam, robot, device) -> None:
    """Pose the OAK camera on the robot's torso_link: mount offset, then the 25 deg pitch-down (ROS convention)."""
    idx = robot.data.body_names.index("torso_link")
    torso_pos, torso_quat = (robot.data.body_pos_w[0, idx], robot.data.body_quat_w[0, idx])
    cam_pos = torso_pos + quat_rotate(torso_quat, torch.tensor(OAK_OFFSET_FROM_TORSO, device=device))
    cam_rot = quat_mul(torso_quat, torch.tensor(OAK_CAM_ROT, device=device))
    half = torch.tensor(math.radians(-(0.0 + OAK_CAM_PITCH_DOWN_DEG)) / 2.0, device=device)
    zero = torch.tensor(0.0, device=device)
    cam_rot = quat_mul(cam_rot, torch.stack([torch.cos(half), torch.sin(half), zero, zero]))
    oak_cam.set_world_poses(positions=cam_pos.unsqueeze(0), orientations=cam_rot.unsqueeze(0), convention="ros")


def capture_layout(sim, entities, chair_pos, robot_start_pos, run_dir, sim_dt):
    """Top-down layout diagnostic (layout_topdown_rgb.png + layout_surface_z.npy) above the robot / chair midpoint."""
    from PIL import Image, ImageDraw

    layout_cam = entities["layout_cam"]
    mid_x = (robot_start_pos[0] + chair_pos[0]) / 2.0
    mid_y = (robot_start_pos[1] + chair_pos[1]) / 2.0
    view_size = 14.0  # metres covered by the layout view
    cam_pos = torch.tensor([mid_x, mid_y, LAYOUT_HEIGHT], dtype=torch.float32, device=sim.device)
    cam_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=sim.device)
    layout_cam.set_world_poses(positions=cam_pos.unsqueeze(0), orientations=cam_quat.unsqueeze(0))
    for _ in range(5):
        sim.step()
        layout_cam.update(dt=sim_dt)
    depth = layout_cam.data.output["distance_to_image_plane"][0].cpu().numpy()[:, :, 0]
    surface_z = LAYOUT_HEIGHT - depth
    np.save(os.path.join(run_dir, "layout_surface_z.npy"), surface_z)

    rgb = layout_cam.data.output["rgb"][0].cpu().numpy()[:, :, :3].copy()
    h, w = rgb.shape[:2]

    def world_to_pixel(wx, wy):
        px = int(((wx - mid_x) / view_size + 0.5) * w)
        py = int((-(wy - mid_y) / view_size + 0.5) * h)
        return np.clip(px, 5, w - 5), np.clip(py, 5, h - 5)

    rx, ry = world_to_pixel(robot_start_pos[0], robot_start_pos[1])
    cx, cy = world_to_pixel(chair_pos[0], chair_pos[1])
    rgb[ry - 3 : ry + 4, rx - 3 : rx + 4] = [0, 255, 0]
    rgb[cy - 3 : cy + 4, cx - 3 : cx + 4] = [255, 0, 0]
    pil_img = Image.fromarray(rgb.astype(np.uint8))
    try:
        draw = ImageDraw.Draw(pil_img)
        draw.text((rx + 5, ry), "Robot", fill=(0, 255, 0))
        draw.text((cx + 5, cy), "Chair", fill=(255, 0, 0))
        draw.text((10, 10), f"Top-down  Center=({mid_x:.1f},{mid_y:.1f})", fill=(255, 255, 0))
    except Exception:
        pass
    pil_img.save(os.path.join(run_dir, "layout_topdown_rgb.png"))
    np.save(os.path.join(run_dir, "layout_surface_z.npy"), surface_z.astype(np.float32))


def _set_tpv_pose(tpv_cam, eye, target, device):
    """Look-at pose of the fixed TPV camera (ROS: +Z forward, +X right, +Y down)."""
    from scipy.spatial.transform import Rotation as Rot

    eye = torch.tensor(np.asarray(eye), dtype=torch.float32, device=device)
    target = torch.tensor(np.asarray(target), dtype=torch.float32, device=device)
    fwd = target - eye
    fwd = fwd / fwd.norm()
    rt = torch.linalg.cross(fwd, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device))
    rt = rt / rt.norm()
    cam_down = torch.linalg.cross(fwd, rt)
    q = Rot.from_matrix(torch.stack([rt, cam_down, fwd], dim=1).cpu().numpy()).as_quat()  # xyzw
    tpv_q = torch.tensor([q[3], q[0], q[1], q[2]], dtype=torch.float32, device=device)
    tpv_cam.set_world_poses(positions=eye.unsqueeze(0), orientations=tpv_q.unsqueeze(0), convention="ros")


def run_simulator(sim, entities, qpos_traj, chair_pos_zup, run_dir, tpv_eye, tpv_target, frame_sinks, fps=30):
    """Replay ``qpos_traj`` once: each saved frame's OAK / TPV RGB goes to ``frame_sinks[key](fid, rgb)``."""
    robot, oak_cam, tpv_cam = entities["g1"], entities["oak_cam"], entities["tpv_cam"]
    sim_dt = sim.get_physics_dt()
    num_frames = qpos_traj.shape[0]
    capture_layout(sim, entities, chair_pos_zup, qpos_traj[0, :3].copy(), run_dir, sim_dt)

    mj_name_to_idx = {name: i for i, name in enumerate(MUJOCO_JOINT_NAMES)}
    joint_map = [(i, 7 + mj_name_to_idx[n]) for i, n in enumerate(robot.data.joint_names) if n in mj_name_to_idx]
    print(f"[INFO] Matched {len(joint_map)}/{len(robot.data.joint_names)} joints; replaying {num_frames} frames")
    _set_tpv_pose(tpv_cam, tpv_eye, tpv_target, sim.device)

    # Warm up the stage and camera buffers. The frame loop then only calls sim.render(): a sim.step() would pull
    # joints the actuator regexes do not match toward zero and the frames would drift from qpos.
    for _ in range(6):
        sim.step()
        robot.update(sim_dt)
        oak_cam.update(dt=sim_dt)
        tpv_cam.update(dt=sim_dt)

    root_pos_l, root_quat_l, joints_l = [], [], []
    for frame_idx in range(num_frames):
        qpos = qpos_traj[frame_idx]
        root_state = robot.data.default_root_state.clone()
        root_pos = qpos[:3].copy()
        root_state[:, :3] = torch.tensor(root_pos, dtype=torch.float32, device=sim.device)
        root_state[:, 3:7] = torch.tensor(qpos[3:7], dtype=torch.float32, device=sim.device)
        target_pos = robot.data.default_joint_pos.clone()
        for sim_idx, qpos_idx in joint_map:
            target_pos[:, sim_idx] = float(qpos[qpos_idx])
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        robot.write_joint_state_to_sim(target_pos, torch.zeros_like(target_pos))
        robot.write_data_to_sim()
        update_oak_camera(oak_cam, robot, sim.device)
        sim.render()
        robot.update(0.0)
        oak_cam.update(dt=0.0)
        tpv_cam.update(dt=0.0)

        # Skip teleport frames (> 2 m/s root speed) among the first 10.
        if 0 < frame_idx < 10 and np.linalg.norm(qpos[:3] - qpos_traj[frame_idx - 1][:3]) * 30.0 > 2.0:
            continue
        fid = len(root_pos_l)
        for key, cam in (("oak_rgb", oak_cam), ("tpv_rgb", tpv_cam)):
            frame_sinks[key](fid, cam.data.output["rgb"][0].cpu().numpy()[:, :, :3].astype(np.uint8))
        root_pos_l.append(root_pos.copy())
        root_quat_l.append(qpos[3:7].copy())
        joints_l.append(qpos[7:].copy())
        if frame_idx % 50 == 0:
            print(f"[SAVE ({fid + 1}/{num_frames})] frame={frame_idx}/{num_frames} root_z={root_pos[2]:.3f}")

    np.savez_compressed(
        os.path.join(run_dir, "traj.npz"),
        qpos=qpos_traj,
        root_pos=np.array(root_pos_l),
        root_quat=np.array(root_quat_l),
        joint_angles=np.array(joints_l),
        fps=fps,
        num_frames=len(root_pos_l),
        chair_pos=chair_pos_zup,
    )
