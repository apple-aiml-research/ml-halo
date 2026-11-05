#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""obj_to_usdc.py -- multi-material baked OBJ -> binary .usdc (double-sided, unlit emissive materials)."""

import argparse, os, shutil
from collections import OrderedDict

from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf

from project_gt import parse_mtl, parse_obj


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("obj", help="input OBJ (with its .mtl and textures next to it)")
    ap.add_argument("out", help="output path; forced to .usdc (ASCII .usda makes Isaac render black)")
    a = ap.parse_args()
    if not a.out.endswith(".usdc"):
        a.out = os.path.splitext(a.out)[0] + ".usdc"
    outdir = os.path.dirname(os.path.abspath(a.out)) or "."
    objdir = os.path.dirname(os.path.abspath(a.obj)) or "."
    V, VT, faces, mtllib = parse_obj(a.obj)
    mtl_tex = parse_mtl(os.path.join(objdir, mtllib)) if mtllib else {}
    texout = os.path.join(outdir, "tex")
    os.makedirs(texout, exist_ok=True)

    stage = Usd.Stage.CreateNew(a.out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/World/scan")
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in V])

    fv, fvt, subsets = [], [], OrderedDict()
    tri = 0
    for mat, tris in faces.items():
        if tris:
            for vi, ti in tris:
                fv.extend(vi)
                fvt.extend(ti)
            subsets[mat] = (tri, tri + len(tris))
            tri += len(tris)
    mesh.CreateFaceVertexCountsAttr([3] * tri)
    mesh.CreateFaceVertexIndicesAttr(fv)
    if len(VT) and any(i >= 0 for i in fvt):
        pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        )
        pv.Set([Gf.Vec2f(float(VT[i][0]), float(VT[i][1])) if i >= 0 else Gf.Vec2f(0, 0) for i in fvt])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)  # thin single-sided sheets; bake.py already made the normals consistent

    n_tex = 0
    for mi, (mat, (s, e)) in enumerate(subsets.items()):
        mpath = f"/World/mats/m{mi}_" + "".join(c if c.isalnum() else "_" for c in mat)
        usdmat = UsdShade.Material.Define(stage, mpath)
        surf = UsdShade.Shader.Define(stage, mpath + "/surf")
        surf.CreateIdAttr("UsdPreviewSurface")
        surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        surf.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
        surf.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set((0.0, 0.0, 0.0))
        src = os.path.normpath(os.path.join(objdir, mtl_tex[mat])) if mtl_tex.get(mat) else None
        if src and os.path.isfile(src):
            dst = os.path.join(texout, os.path.basename(src))
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copy(src, dst)
            reader = UsdShade.Shader.Define(stage, mpath + "/streader")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            st_out = reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            t = UsdShade.Shader.Define(stage, mpath + "/tex")
            t.CreateIdAttr("UsdUVTexture")
            t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./tex/{os.path.basename(src)}")
            t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
            rgb = t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            surf.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb)
            surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.0, 0.0, 0.0))
            n_tex += 1
        usdmat.CreateSurfaceOutput().ConnectToSource(surf.CreateOutput("surface", Sdf.ValueTypeNames.Token))
        sub = UsdGeom.Subset.CreateGeomSubset(
            mesh, f"sub{mi}", UsdGeom.Tokens.face, list(range(s, e)), familyName="materialBind"
        )
        UsdShade.MaterialBindingAPI(sub.GetPrim()).Bind(usdmat)

    stage.GetRootLayer().Save()
    print(f"wrote {a.out} tris={tri} materials={len(subsets)} textured={n_tex} verts={len(V)}", flush=True)


if __name__ == "__main__":
    main()
