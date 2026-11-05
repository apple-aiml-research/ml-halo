#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Baked multi-material OBJ → MuJoCo mesh parts + unlit/emissive scene XML."""

from __future__ import annotations

import logging
import os
from collections import OrderedDict

import numpy as np

log = logging.getLogger("obj_to_mjcf")

_VLIMIT = 60000

OUTLIER_QUANTILE = 0.001
OUTLIER_MARGIN_FRAC = 1.0
OUTLIER_MARGIN_MIN_M = 5.0


def outlier_vertex_mask(xyz: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Vertices outside the [0.1%, 99.9%] quantile box widened by max(5 m, its extent), and that box."""
    lo = np.quantile(xyz, OUTLIER_QUANTILE, axis=0)
    hi = np.quantile(xyz, 1.0 - OUTLIER_QUANTILE, axis=0)
    margin = np.maximum(OUTLIER_MARGIN_MIN_M, OUTLIER_MARGIN_FRAC * (hi - lo))
    lo, hi = lo - margin, hi + margin
    return np.any((xyz < lo) | (xyz > hi), axis=1), (lo, hi)


def split_obj(obj_path: str, workdir: str) -> list:
    """Split a multi-material OBJ into per-material, de-indexed, double-sided OBJ parts (outlier faces dropped)."""
    V, VT, VN = [], [], []
    groups: "OrderedDict[str, list]" = OrderedDict()
    cur = "default"
    mtllib = None
    with open(obj_path, errors="ignore") as fh:
        for ln in fh:
            if ln.startswith("v "):
                V.append(ln.split()[1:4])
            elif ln.startswith("vt "):
                VT.append(ln.split()[1:3])
            elif ln.startswith("vn "):
                VN.append(ln.split()[1:4])
            elif ln.startswith("mtllib"):
                mtllib = ln.split(maxsplit=1)[1].strip()
            elif ln.startswith("usemtl"):
                cur = ln.split(maxsplit=1)[1].strip()
                groups.setdefault(cur, [])
            elif ln.startswith("f "):
                groups.setdefault(cur, []).append(ln.split()[1:])

    tex: dict = {}
    if mtllib:
        mp = os.path.join(os.path.dirname(obj_path), mtllib)
        curm = None
        if os.path.isfile(mp):
            for ln in open(mp, errors="ignore"):
                p = ln.split()
                if not p:
                    continue
                if p[0] == "newmtl":
                    curm = p[1]
                elif p[0] in ("map_Kd", "map_Ke") and curm:
                    tex.setdefault(curm, p[-1])

    bad = None
    if V:
        mask, (lo, hi) = outlier_vertex_mask(np.asarray(V, dtype=np.float64))
        if mask.any():
            bad = mask
            log.warning(
                "%s: %d of %d vertices outside %s .. %s (outliers); dropping their faces",
                obj_path,
                int(mask.sum()),
                len(V),
                np.round(lo, 2).tolist(),
                np.round(hi, 2).tolist(),
            )

    os.makedirs(workdir, exist_ok=True)
    out = []
    objdir = os.path.dirname(os.path.abspath(obj_path))
    for i, (mat, faces) in enumerate(groups.items()):
        if bad is not None:
            kept = [toks for toks in faces if not any(bad[int(t.split("/")[0]) - 1] for t in toks)]
            if len(kept) != len(faces):
                log.warning("%s: dropped %d of %d faces (outlier vertices)", mat, len(faces) - len(kept), len(faces))
            faces = kept
        if not faces:
            continue
        chunks, remap, ov, ovt, ovn, of = [], {}, [], [], [], []

        def flush():
            if of:
                chunks.append((ov[:], ovt[:], ovn[:], of[:]))

        for toks in faces:
            if len(ov) + 3 > _VLIMIT:
                flush()
                remap, ov, ovt, ovn, of = {}, [], [], [], []
            idx = []
            for t in toks:
                b = (t.split("/") + ["", ""])[:3]
                vi = int(b[0]) - 1
                ti = int(b[1]) - 1 if b[1] else -1
                ni = int(b[2]) - 1 if b[2] else -1
                key = (vi, ti, ni)
                j = remap.get(key)
                if j is None:
                    j = len(ov) + 1  # 1-based OBJ index
                    remap[key] = j
                    ov.append(V[vi])
                    ovt.append(VT[ti] if 0 <= ti < len(VT) else ["0", "0"])
                    ovn.append(VN[ni] if 0 <= ni < len(VN) else None)
                idx.append(j)
            for k in range(1, len(idx) - 1):  # fan triangulate
                of.append((idx[0], idx[k], idx[k + 1]))
                of.append((idx[0], idx[k + 1], idx[k]))  # reversed winding: double-sided
        flush()

        t = tex.get(mat)
        tp = os.path.normpath(os.path.join(objdir, t)) if t else None
        if tp and not os.path.isfile(tp):
            tp = None
        for ci, (cv_, cvt, cvn, cf) in enumerate(chunks):
            nm = f"g{i:03d}" if len(chunks) == 1 else f"g{i:03d}_{ci}"
            p = os.path.join(workdir, f"{nm}.obj")
            with open(p, "w") as f:
                f.writelines(f"v {x} {y} {z}\n" for x, y, z in cv_)
                f.writelines(f"vt {x} {y}\n" for x, y in cvt)
                have_n = all(n is not None for n in cvn)
                if have_n:
                    f.writelines(f"vn {x} {y} {z}\n" for x, y, z in cvn)
                    f.writelines(f"f {x}/{x}/{x} {y}/{y}/{y} {z}/{z}/{z}\n" for x, y, z in cf)
                else:
                    f.writelines(f"f {x}/{x} {y}/{y} {z}/{z}\n" for x, y, z in cf)
            out.append((nm, p, tp))
        if len(chunks) > 1:
            log.info("%s split into %d chunks to stay under %d verts", mat, len(chunks), _VLIMIT)
    return out


def geometry_assets_and_geoms(parts, geom_group: int = 2) -> tuple[str, str]:
    """Build the ``<asset>`` fragment and ``<geom>`` list for a set of split parts."""
    assets, geoms = [], []
    for name, objf, texf in parts:
        assets.append(f'    <mesh name="m_{name}" file="{objf}" inertia="shell"/>')
        if texf:
            assets.append(f'    <texture name="t_{name}" type="2d" file="{texf}"/>')
            assets.append(
                f'    <material name="mat_{name}" texture="t_{name}" '
                f'specular="0" shininess="0" reflectance="0" emission="1"/>'
            )
            geoms.append(
                f'    <geom type="mesh" mesh="m_{name}" material="mat_{name}" '
                f'group="{geom_group}" contype="0" conaffinity="0"/>'
            )
        else:
            geoms.append(
                f'    <geom type="mesh" mesh="m_{name}" rgba="0.7 0.7 0.7 1" '
                f'group="{geom_group}" contype="0" conaffinity="0"/>'
            )
    return "\n".join(assets), "\n".join(geoms)
