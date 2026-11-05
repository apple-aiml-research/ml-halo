#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Isaac-side loader for infinigen baked rooms."""

from __future__ import annotations

import os


def split_subsets_two_sided(stage, tag: str = "[infinigen-usd]") -> int:
    """Split every GeomSubset-carrying mesh on ``stage`` into one doubleSided mesh per material."""
    import numpy as np
    from pxr import UsdGeom, UsdShade, Sdf

    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    n_new = 0
    for prim in meshes:
        m = UsdGeom.Mesh(prim)
        subs = [c for c in prim.GetChildren() if c.IsA(UsdGeom.Subset)]
        counts = m.GetFaceVertexCountsAttr().Get()
        idx = m.GetFaceVertexIndicesAttr().Get()
        if not subs or not counts:
            m.CreateDoubleSidedAttr().Set(True)  # no subsets → just make it two-sided
            continue
        if not all(c == 3 for c in counts):
            print(f"{tag} {prim.GetPath()}: non-triangulated, skipping split")
            continue
        idx = np.asarray(idx, dtype=np.int64).reshape(-1, 3)
        pv = UsdGeom.PrimvarsAPI(m).GetPrimvar("st")
        st = np.asarray(pv.Get()).reshape(-1, 3, 2) if (pv and pv.Get()) else None
        pts = m.GetPointsAttr().Get()
        P = np.asarray(pts, dtype=np.float64)
        centroid = P.mean(0)
        parent = prim.GetParent().GetPath()
        split_root = parent.AppendChild(prim.GetName() + "_split")
        UsdGeom.Xform.Define(stage, split_root)
        for si, sub in enumerate(subs):
            faces = np.asarray(UsdGeom.Subset(sub).GetIndicesAttr().Get(), dtype=np.int64)
            if len(faces) == 0:
                continue
            tri = idx[faces].copy()  # (F,3) vertex indices
            a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
            nrm = np.cross(b - a, c - a)
            fc = (a + b + c) / 3.0
            outward = (nrm * (centroid - fc)).sum(1) < 0.0
            tri[outward] = tri[outward][:, ::-1]
            nm = UsdGeom.Mesh.Define(stage, split_root.AppendChild(f"m{si:03d}"))
            nm.CreatePointsAttr(pts)  # share full point set
            nm.CreateFaceVertexCountsAttr([3] * len(faces))
            nm.CreateFaceVertexIndicesAttr(tri.reshape(-1).tolist())
            nm.CreateDoubleSidedAttr().Set(True)  # ← honoured (no subsets)
            if st is not None:
                sst = st[faces].copy()
                sst[outward] = sst[outward][:, ::-1]  # keep UV aligned to flipped winding
                sp = UsdGeom.PrimvarsAPI(nm).CreatePrimvar(
                    "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
                )
                sp.Set(sst.reshape(-1, 2))
            mat = UsdShade.MaterialBindingAPI(sub).GetDirectBinding().GetMaterial()
            if mat:
                UsdShade.MaterialBindingAPI(nm).Bind(mat)
            n_new += 1
        prim.SetActive(False)  # hide the original subset mesh; only the split meshes render
        print(
            f"{tag} {prim.GetPath()}: split {len(subs)} subsets → "
            f"{n_new} inward-wound doubleSided meshes (no GeomSubsets)"
        )
    return n_new


def _orient_usd_inward(src_usdc: str, dst_usdc: str) -> str:
    """Rewrite ``src_usdc`` → ``dst_usdc`` so Isaac renders the room from an interior ego camera."""
    from pxr import Usd, UsdShade, Sdf

    stage = Usd.Stage.Open(src_usdc)
    n_new = split_subsets_two_sided(stage)

    n_mat = 0
    for mp in [p for p in stage.Traverse() if p.IsA(UsdShade.Material)]:
        surf = None
        for sh in Usd.PrimRange(mp):
            if sh.IsA(UsdShade.Shader) and UsdShade.Shader(sh).GetIdAttr().Get() == "UsdPreviewSurface":
                surf = UsdShade.Shader(sh)
                break
        if surf is None:
            continue
        dif = surf.GetInput("diffuseColor") or surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        emi = surf.GetInput("emissiveColor")
        srcs = emi.GetConnectedSources() if emi else None
        if srcs and srcs[0]:
            si = srcs[0][0]  # UsdShadeConnectionSourceInfo(source, sourceName, ...)
            dif.ConnectToSource(si.source.GetOutput(si.sourceName))
            n_mat += 1
    print(f"[infinigen-usd] emissive→diffuse texture on {n_mat} materials (interior LIT like the lab scan)")

    stage.GetRootLayer().Export(dst_usdc)
    print(f"[infinigen-usd] wrote two-sided + diffuse-lit split USD → {dst_usdc} ({n_new} meshes)")
    return dst_usdc


def _load_usd_scene(sim_utils, usd_path, offset=(0.0, 0.0, 0.0)):
    """Reference ``usd_path`` under /World/SceneBackground, scaled by its metersPerUnit."""
    import omni.usd
    from pxr import Gf, Usd, UsdGeom

    total_scale = UsdGeom.GetStageMetersPerUnit(Usd.Stage.Open(usd_path))
    stage = omni.usd.get_context().get_stage()
    xf = UsdGeom.Xformable(stage.DefinePrim("/World/SceneBackground", "Xform"))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(offset[0], offset[1], offset[2]))
    xf.AddScaleOp().Set(Gf.Vec3f(total_scale, total_scale, total_scale))
    cfg = sim_utils.UsdFileCfg(usd_path=usd_path)
    cfg.func("/World/SceneBackground/Mesh", cfg, translation=(0, 0, 0), orientation=(1.0, 0.0, 0.0, 0.0))


def build_infinigen_scene(sim_utils, scene_dir: str, offset=(0.0, 0.0, 0.0)) -> str:
    """Load ``<scene_dir>/asset/scan.usdc`` (two-sided, diffuse-lit) as /World/SceneBackground plus room lights."""
    import json

    scene_dir = os.path.expanduser(scene_dir)
    usd = os.path.join(scene_dir, "asset", "scan.usdc")
    if not os.path.exists(usd):
        raise FileNotFoundError(f"{usd} not found")
    try:
        usd = _orient_usd_inward(usd, os.path.join(scene_dir, "asset", "scan_twosided.usdc"))
    except Exception as e:
        print(f"[infinigen-usd] WARN: two-sided split failed ({e}); loading original scan.usdc")
    _load_usd_scene(sim_utils, usd, offset=tuple(offset))

    room = {"W": 5.0, "D": 4.0, "H": 2.8, "floor_z": 0.12}
    try:
        room.update(json.load(open(os.path.join(scene_dir, "scene_analysis.json")))["room"])
    except Exception:
        pass
    W, D, H, fz = room["W"], room["D"], room.get("H", 2.8), room.get("floor_z", 0.12)
    lz = fz + max(1.5, H - 0.4)  # just below the ceiling, inside the sealed box
    dome = sim_utils.DomeLightCfg(intensity=200.0, color=(0.95, 0.95, 0.97))
    dome.func("/World/SceneDomeLight", dome)
    k = 0
    for fx_ in (0.28, 0.72):
        for fy_ in (0.28, 0.72):
            lc = sim_utils.SphereLightCfg(intensity=40000.0, color=(1.0, 0.98, 0.95), radius=0.15)
            lc.func(f"/World/SceneInteriorLight_{k}", lc, translation=(W * fx_, D * fy_, lz))
            k += 1
    print(f"[infinigen-usd] loaded {usd} + {k} interior lights at z={lz:.2f}")
    return usd
