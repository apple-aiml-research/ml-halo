#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Place EpisodeSpec props into an Isaac stage and set per-episode lighting."""

from __future__ import annotations

import os
import zlib
from typing import Sequence

from render.assets import extract_member, resolve_asset, usdz_mesh
from motion_gen.episode_spec import EpisodeSpec

_HIDDEN_Z = -100.0  # unused pre-spawned props are parked here, well below the floor
_TEX_TMP_DIR = "/tmp/halo_syndata_tex"


def clear_episode_prims(root: str = "/World") -> None:
    """Delete the per-episode prims (``Episode*``) under ``root``."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root)
    if not root_prim.IsValid():
        return
    for path in [p.GetPath() for p in root_prim.GetChildren() if p.GetName().startswith(("Box", "Episode"))]:
        stage.RemovePrim(path)


def _sanitize(name: str) -> str:
    s = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    return ("X_" + s if s and s[0].isdigit() else s) or "Asset"


def _uv_texture(stage, mat_path, name, file, uv_out, raw=False):
    """A UsdUVTexture shader reading ``file`` at the mesh ``st`` coordinates."""
    from pxr import Sdf, UsdShade

    tex = UsdShade.Shader.Define(stage, f"{mat_path}/{name}")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(file))
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_out)
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    if raw:
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
    return tex


def _place_prop_mesh(sim_utils, usdz_path: str, prim_path: str, pos_zup: Sequence[float]) -> bool:
    """Spawn one USDZ prop as a mesh prim with a UsdPreviewSurface (diffuse + normal + AO maps)."""
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    data = usdz_mesh(usdz_path)
    if data is None or data["uvs"] is None or "diffuse" not in data["tex"]:
        return False
    tex = {
        k: extract_member(
            usdz_path, m, os.path.join(_TEX_TMP_DIR, f"{zlib.crc32(usdz_path.encode('utf-8')):x}_{os.path.basename(m)}")
        )
        for k, m in data["tex"].items()
    }
    verts, faces = data["verts"], data["faces"].astype("int32")

    from render.isaac.scene_builder import _create_prim_compat

    stage = omni.usd.get_context().get_stage()
    _create_prim_compat(
        sim_utils, prim_path, "Xform", translation=tuple(float(x) for x in pos_zup), orientation=(1.0, 0.0, 0.0, 0.0)
    )
    mesh_prim = UsdGeom.Mesh.Define(stage, f"{prim_path}/Mesh")
    mesh_prim.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*v) for v in verts]))
    mesh_prim.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(faces)))
    mesh_prim.GetFaceVertexIndicesAttr().Set(Vt.IntArray(faces.flatten().tolist()))
    mesh_prim.GetDoubleSidedAttr().Set(True)
    primvar = UsdGeom.PrimvarsAPI(mesh_prim).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    primvar.Set(Vt.Vec2fArray([Gf.Vec2f(float(u), float(v)) for u, v in data["uvs"].reshape(-1, 2)]))

    mat_path = f"{prim_path}/Material"
    material = UsdShade.Material.Define(stage, mat_path)
    surf = UsdShade.Shader.Define(stage, f"{mat_path}/Surface")
    surf.CreateIdAttr("UsdPreviewSurface")
    surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.02)
    uv_reader = UsdShade.Shader.Define(stage, f"{mat_path}/UVReader")
    uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
    uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    uv_out = uv_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    diffuse = _uv_texture(stage, mat_path, "Tex", tex["diffuse"], uv_out)
    diffuse.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(1.0, 1.0, 1.0, 1.0))
    surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        diffuse.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )
    if "normal" in tex:
        ntex = _uv_texture(stage, mat_path, "NormalTex", tex["normal"], uv_out, raw=True)
        ntex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2.0, 2.0, 2.0, 1.0))
        ntex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1.0, -1.0, -1.0, 0.0))
        surf.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
            ntex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        )
    if "ao" in tex:
        atex = _uv_texture(stage, mat_path, "AOTex", tex["ao"], uv_out, raw=True)
        surf.CreateInput("occlusion", Sdf.ValueTypeNames.Float).ConnectToSource(
            atex.CreateOutput("r", Sdf.ValueTypeNames.Float)
        )
    material.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)
    return True


def prespawn_prop_pool(sim_utils, asset_paths: Sequence[str], root: str = "/World/PreSpawnPool") -> dict[str, str]:
    """Spawn every distinct prop once, parked below the floor; returns {resolved asset path: prim path}."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(root):
        UsdGeom.Xform.Define(stage, root)
    pool: dict[str, str] = {}
    for raw_path in asset_paths:
        usdz = resolve_asset(raw_path)
        if usdz in pool:
            continue
        if not os.path.isfile(usdz):
            print(f"  [prespawn] WARN missing USDZ: {raw_path}", flush=True)
            continue
        i = len(pool)
        prim_path = f"{root}/Prop_{i:03d}_{_sanitize(os.path.basename(usdz).split('.')[0])}"
        if not _place_prop_mesh(sim_utils, usdz, prim_path, pos_zup=(0.0, 0.0, _HIDDEN_Z - 0.5 * i)):
            print(f"  [prespawn] WARN failed to extract {os.path.basename(usdz)}", flush=True)
            continue
        pool[usdz] = prim_path
    print(f"  [prespawn] pre-spawned {len(pool)} prop mesh(es) under {root}", flush=True)
    return pool


def _set_prim_xform(stage, prim_path: str, translation: Sequence[float], quat_wxyz: Sequence[float] | None = None):
    """Set translate (and orient) on an existing prim, matching each op's precision."""
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
    ops = xf.GetOrderedXformOps()
    t_op = next((o for o in ops if o.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
    o_op = next((o for o in ops if o.GetOpType() == UsdGeom.XformOp.TypeOrient), None)
    if t_op is None:
        t_op = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat)
    double = t_op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble
    t_op.Set((Gf.Vec3d if double else Gf.Vec3f)(*[float(v) for v in translation]))
    if quat_wxyz is not None:
        if o_op is None:
            o_op = xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
        w, x, y, z = [float(v) for v in quat_wxyz]
        if o_op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
            o_op.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
        else:
            o_op.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def place_episode_objects(spec: EpisodeSpec, prop_pool: dict[str, str]) -> None:
    """Park every pooled prop, then move the spec's placements into place."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    for prim_path in prop_pool.values():
        _set_prim_xform(stage, prim_path, (0.0, 0.0, _HIDDEN_Z))
    seen: set[str] = set()
    for p in spec.placements:
        usdz = resolve_asset(p.asset_path)
        if usdz in seen:
            continue
        if usdz not in prop_pool:
            print(f"  [place] WARN placement USDZ not in prop_pool: {usdz}", flush=True)
            continue
        _set_prim_xform(stage, prop_pool[usdz], [float(x) for x in p.pos_xyz[:3]], p.quat_wxyz)
        seen.add(usdz)


def set_episode_lighting(sim_utils, spec: EpisodeSpec, height: float = 2.7, root: str = "/World") -> None:
    """The spec's dome + overhead sphere light as a fresh ``Episode*`` light pair."""
    dome = sim_utils.DomeLightCfg(intensity=spec.dome_intensity, color=spec.dome_color)
    dome.func(f"{root}/EpisodeDomeLight", dome)
    overhead = sim_utils.SphereLightCfg(intensity=spec.overhead_intensity, color=(1.0, 0.98, 0.95), radius=0.3)
    overhead.func(f"{root}/EpisodeOverheadLight", overhead, translation=(0.0, 0.0, max(0.5, height - 0.2)))
