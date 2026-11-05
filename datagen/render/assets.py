#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Prop assets (the sit chairs in assets/objects/chairs): locate a spec's asset and read its USDZ mesh."""

from __future__ import annotations

import os
import zipfile

import numpy as np

CHAIRS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "assets", "objects", "chairs"))
_DIFFUSE_KEYS = ("_basecolor", "_albedo", "_diffuse", "_diff", "_color", "_tex")


def resolve_asset(path: str) -> str:
    """The repo copy of ``path`` (matched by file name) if there is one, else ``path`` itself."""
    cand = os.path.join(CHAIRS_DIR, os.path.basename(path or ""))
    return cand if path and os.path.isfile(cand) else os.path.expanduser(path)


def usdz_mesh(usdz_path: str) -> dict | None:
    """Triangle mesh of a USDZ in Z-up metres, centred in xy with its base at z=0.

    Returns ``{"verts" (V,3), "faces" (F,3) int64, "uvs" (F,3,2) float32 or None, "tex": {"diffuse"|"normal"|"ao":
    zip member}}``, or None if the file cannot be opened.
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(usdz_path)
    if stage is None:
        return None
    mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    up = UsdGeom.GetStageUpAxis(stage) or "Y"
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    verts_list, faces_list, uvs_list = [], [], []
    n_verts = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        fvc = mesh.GetFaceVertexCountsAttr().Get()
        fvi = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not fvc or not fvi:
            continue
        if any(c != 3 for c in fvc):
            raise ValueError(f"{usdz_path}: {prim.GetPath()} is not triangulated")
        xform = cache.GetLocalToWorldTransform(prim)
        m = np.array([[xform[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)
        pts = np.asarray([(p[0], p[1], p[2]) for p in points], dtype=np.float64)
        verts = (np.hstack([pts, np.ones((len(pts), 1))]) @ m)[:, :3] * mpu
        if up == "Y":
            verts = np.stack([verts[:, 0], -verts[:, 2], verts[:, 1]], axis=1)
        tri = np.asarray(list(fvi), dtype=np.int64).reshape(-1, 3)

        pv_api = UsdGeom.PrimvarsAPI(prim)
        st = pv_api.GetPrimvar("st")
        for alt in ("UVMap", "uv", "Texture_uv"):
            if st.IsDefined():
                break
            st = pv_api.GetPrimvar(alt)
        if st.IsDefined() and st.Get() is not None:
            st_arr = np.array([(s[0], s[1]) for s in st.Get()], dtype=np.float32)
            if st.GetInterpolation() == "vertex":
                uvs_list.append(st_arr[tri.reshape(-1)].reshape(-1, 3, 2))
            else:  # faceVarying
                idx = np.asarray(list(st.GetIndices()), dtype=np.int64) if st.IsIndexed() else np.arange(tri.size)
                uvs_list.append(st_arr[idx[: tri.size]].reshape(-1, 3, 2))
        verts_list.append(verts)
        faces_list.append(tri + n_verts)
        n_verts += len(verts)
    if not verts_list:
        return None
    verts = np.concatenate(verts_list, axis=0)
    bb_min, bb_max = verts.min(0), verts.max(0)
    verts[:, 0] -= 0.5 * (bb_min[0] + bb_max[0])
    verts[:, 1] -= 0.5 * (bb_min[1] + bb_max[1])
    verts[:, 2] -= bb_min[2]
    uvs = np.concatenate(uvs_list, axis=0) if len(uvs_list) == len(verts_list) else None

    tex = {}
    with zipfile.ZipFile(usdz_path) as zf:
        images = [i.filename for i in zf.infolist() if i.filename.lower().endswith((".png", ".jpg", ".jpeg"))]
    for name in images:
        low = os.path.basename(name).lower()
        if "_norm" in low:
            tex.setdefault("normal", name)
        elif "_ao" in low:
            tex.setdefault("ao", name)
        elif any(k in low for k in _DIFFUSE_KEYS):
            tex.setdefault("diffuse", name)
    return {"verts": verts, "faces": np.concatenate(faces_list, axis=0), "uvs": uvs, "tex": tex}


def extract_member(usdz_path: str, member: str, dst: str) -> str:
    """Write one USDZ archive member to ``dst`` (unless it already exists); returns ``dst``."""
    if not os.path.isfile(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with zipfile.ZipFile(usdz_path) as zf, zf.open(member) as src, open(dst, "wb") as out:
            out.write(src.read())
    return dst
