# Coordinate frames

Two frames: SMPL / Kimodo internals are **Y-up**; everything else (Kimodo's `MujocoQposConverter` output,
`g1_qpos.npy`, parquet, MuJoCo and Isaac scenes) is world **Z-up**, gravity-aligned. Motion is generated in
world Z-up with the floor at z=0 and the chair at the origin. The render places it into a scene with one rigid
SE(2) transform plus a lift to the scene's `floor_z` (`render/synthetic_scene/robot_placement.py`); the parquet
poses are in that scene frame.

Y-up to Z-up is `[x, y, z] -> [x, -z, y]` (and back `[x, y, z] -> [x, z, -y]`).

## Quaternions

All code uses **wxyz**: `g1_qpos.npy` `[:, 3:7]`, parquet `observation.body_quat`, MuJoCo `data.qpos[3:7]`,
Isaac `set_world_poses(orientations=...)`. scipy's `Rotation` uses xyzw: convert at the call
(`q[[1, 2, 3, 0]]` / `q[[3, 0, 1, 2]]`) and never store xyzw.

## Cameras

The TPV camera is picked per episode from the scene's ranked views (`render/tpv.py::select_infinigen_tpv`). For a
free MuJoCo camera, `forward = (target - eye) / |·|`, `elevation = asin(f_z)`, `azimuth = atan2(f_y, f_x)`
(`render/mujoco/replay.py`); `eye - target` turns it 180° and breaks parity with Isaac.
