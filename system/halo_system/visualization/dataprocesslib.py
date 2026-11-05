#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Scene-mesh helpers for the web viewer: G1 joint names and MuJoCo geoms -> trimesh (textured where possible)."""
from __future__ import annotations

import numpy as np


DEFAULT_G1_JOINT_NAMES = [
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


def _robot_geom_ids(mj_model) -> set:
    """Geom ids of the 'pelvis' subtree (drawn from the URDF, so excluded from the scene)."""
    import mujoco
    pelvis_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id < 0:
        return set()
    robot_bodies = set()
    queue = [pelvis_id]
    while queue:
        bid = queue.pop()
        robot_bodies.add(bid)
        for child in range(mj_model.nbody):
            if mj_model.body_parentid[child] == bid and child not in robot_bodies:
                queue.append(child)
    return {g for g in range(mj_model.ngeom)
            if mj_model.geom_bodyid[g] in robot_bodies}


def _geom_texture_id(mj_model, g) -> int:
    """RGB texture id of geom ``g``'s material, or -1 (first valid role; version-robust)."""
    matid = int(mj_model.geom_matid[g])
    if matid < 0:
        return -1
    texids = np.atleast_1d(mj_model.mat_texid[matid]).ravel()
    valid = [int(t) for t in texids if int(t) >= 0]
    return valid[0] if valid else -1


def _build_textured_mesh(mj_model, g):
    """Textured trimesh for mesh geom ``g`` (local frame), or None if it has no UVs/texture."""
    import trimesh
    from PIL import Image

    mid = int(mj_model.geom_dataid[g])
    if mid < 0:
        return None
    tn = int(mj_model.mesh_texcoordnum[mid])
    tid = _geom_texture_id(mj_model, g)
    if tn <= 0 or tid < 0:
        return None

    va, vn = int(mj_model.mesh_vertadr[mid]), int(mj_model.mesh_vertnum[mid])
    fa, fn = int(mj_model.mesh_faceadr[mid]), int(mj_model.mesh_facenum[mid])
    ta = int(mj_model.mesh_texcoordadr[mid])

    verts = np.asarray(mj_model.mesh_vert[va:va + vn], np.float32)
    faces = np.asarray(mj_model.mesh_face[fa:fa + fn], np.int64)        # local vert idx
    ftc = np.asarray(mj_model.mesh_facetexcoord[fa:fa + fn], np.int64)  # local uv idx
    tc = np.asarray(mj_model.mesh_texcoord[ta:ta + tn], np.float32)

    # MuJoCo stores UVs per face-corner; dedupe to unique (vertex, uv) pairs to keep the mesh small.
    cv = faces.reshape(-1)
    ct = ftc.reshape(-1)
    key = cv.astype(np.int64) * (tn + 1) + ct
    uniq, inv = np.unique(key, return_inverse=True)
    P = verts[(uniq // (tn + 1)).astype(np.int64)]
    UV = tc[(uniq % (tn + 1)).astype(np.int64)].copy()
    UV[:, 1] = 1.0 - UV[:, 1]                    # MuJoCo texel rows are top-down
    new_faces = inv.reshape(-1, 3).astype(np.int64)

    adr = int(mj_model.tex_adr[tid])
    h, w = int(mj_model.tex_height[tid]), int(mj_model.tex_width[tid])
    nch = int(mj_model.tex_nchannel[tid])
    img = np.asarray(mj_model.tex_data[adr:adr + h * w * nch], np.uint8).reshape(h, w, nch)
    pil = Image.fromarray(img)
    visual = trimesh.visual.TextureVisuals(uv=UV, image=pil)
    return trimesh.Trimesh(vertices=P, faces=new_faces, visual=visual, process=False)


def _geom_to_trimesh(mj_model, g):
    """Trimesh for MuJoCo geom ``g`` in its local frame, or None (planes/hfields skipped)."""
    import mujoco
    import trimesh

    gtype = mj_model.geom_type[g]
    size = mj_model.geom_size[g]
    T = mujoco.mjtGeom

    if gtype == T.mjGEOM_PLANE or gtype == T.mjGEOM_HFIELD:
        return None
    if gtype == T.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(subdivisions=2, radius=float(size[0]))
    if gtype == T.mjGEOM_ELLIPSOID:
        m = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        m.apply_scale([float(size[0]), float(size[1]), float(size[2])])
        return m
    if gtype == T.mjGEOM_CAPSULE:
        return trimesh.creation.capsule(radius=float(size[0]),
                                        height=2.0 * float(size[1]))
    if gtype == T.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(radius=float(size[0]),
                                         height=2.0 * float(size[1]), sections=24)
    if gtype == T.mjGEOM_BOX:
        return trimesh.creation.box(extents=2.0 * np.asarray(size[:3], float))
    if gtype == T.mjGEOM_MESH:
        mid = int(mj_model.geom_dataid[g])
        if mid < 0:
            return None
        va, vn = int(mj_model.mesh_vertadr[mid]), int(mj_model.mesh_vertnum[mid])
        fa, fn = int(mj_model.mesh_faceadr[mid]), int(mj_model.mesh_facenum[mid])
        verts = np.asarray(mj_model.mesh_vert[va:va + vn], float).reshape(-1, 3)
        faces = np.asarray(mj_model.mesh_face[fa:fa + fn], np.int64).reshape(-1, 3)
        return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return None
