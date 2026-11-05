#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""MuJoCo kinematic replay: render one episode to fpv / tpv / oak mp4s and traj.npz."""

from __future__ import annotations

import os
import time

import numpy as np

_WS: dict = {}  # model + renderers of the current scene XML


def init_worker(scene_xml: str, width: int, height: int, oak_width: int, oak_height: int) -> None:
    """Load the MuJoCo model, an FPV/TPV renderer at (width, height) and an OAK renderer at the pinhole_render size."""
    import mujoco

    # Free the previous episode's GL contexts before creating new ones: with OSMesa, contexts left to the
    # garbage collector corrupted every frame after the first episode.
    for key in ("renderer", "oak_renderer"):
        if key in _WS:
            _WS.pop(key).close()
    model = mujoco.MjModel.from_xml_path(scene_xml)
    # Isaac's TPV vertical FoV (10 mm focal, 20.955 mm aperture at 848x480); the free camera uses global fovy.
    model.vis.global_.fovy = 61.3
    _WS["model"] = model
    _WS["renderer"] = mujoco.Renderer(model, height=height, width=width)
    _WS["oak_renderer"] = mujoco.Renderer(model, height=oak_height, width=oak_width)


def render_episode(qpos_scene: np.ndarray, ep_dir: str, fps: int, tpv_eye, tpv_target) -> tuple[int, dict]:
    """Render one episode with the loaded model; returns (num_frames, timing).

    The TPV free camera looks from ``tpv_eye`` at ``tpv_target`` (forward = target - eye; the reverse sign
    mirrors the view in xy). oak.mp4 is the raw pinhole: the dataset writer applies the fisheye warp once.
    """
    import imageio.v2 as imageio
    import mujoco

    model, renderer, oak_renderer = _WS["model"], _WS["renderer"], _WS["oak_renderer"]
    data = mujoco.MjData(model)
    tpv_cam = mujoco.MjvCamera()
    tpv_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    fwd = np.array(tpv_target) - np.array(tpv_eye)
    tpv_cam.distance = float(np.linalg.norm(fwd))
    fwd = fwd / tpv_cam.distance
    tpv_cam.lookat[:] = np.array(tpv_target)
    tpv_cam.elevation = float(np.degrees(np.arcsin(fwd[2])))
    tpv_cam.azimuth = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
    ego_opt = mujoco.MjvOption()
    ego_opt.geomgroup[1] = 0  # hide the robot's own visual meshes in the head cameras

    T = len(qpos_scene)
    fpv_frames, tpv_frames, oak_frames = [], [], []
    t0 = time.time()
    for t in range(T):
        nq = min(len(qpos_scene[t]), model.nq)
        data.qpos[:nq] = qpos_scene[t, :nq]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="ego_camera", scene_option=ego_opt)
        fpv_frames.append(renderer.render().copy())
        renderer.update_scene(data, camera=tpv_cam)
        tpv_frames.append(renderer.render().copy())
        oak_renderer.update_scene(data, camera="oak1", scene_option=ego_opt)
        oak_frames.append(oak_renderer.render().copy())
    t_render = time.time() - t0

    t1 = time.time()
    for name, frames in (("fpv.mp4", fpv_frames), ("tpv.mp4", tpv_frames), ("oak.mp4", oak_frames)):
        imageio.mimsave(os.path.join(ep_dir, name), frames, fps=fps, macro_block_size=1)
    t_encode = time.time() - t1
    np.savez(os.path.join(ep_dir, "traj.npz"), qpos=qpos_scene.astype(np.float32), num_frames=np.int64(T))
    return T, {"render_s": t_render, "encode_wait_s": t_encode, "mode": "mujoco"}
