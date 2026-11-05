#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""project_gt.py -- texture a baked asset by photo-projection of the ground-truth renders."""

import argparse, json, os, shutil, time
from collections import OrderedDict

import numpy as np


def parse_obj(path):
    """Return world verts, uvs, per-material fan-triangulated ((vi,vi,vi),(ti,ti,ti)) lists, and the mtllib."""
    V, VT = [], []
    tris = OrderedDict()
    cur, mtllib = "default", None
    for ln in open(path, errors="ignore"):
        if ln.startswith("v "):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("vt "):
            VT.append([float(x) for x in ln.split()[1:3]])
        elif ln.startswith("mtllib"):
            mtllib = ln.split(maxsplit=1)[1].strip()
        elif ln.startswith("usemtl"):
            cur = ln.split(maxsplit=1)[1].strip()
            tris.setdefault(cur, [])
        elif ln.startswith("f "):
            vi, ti = [], []
            for t in ln.split()[1:]:
                b = t.split("/")
                vi.append(int(b[0]) - 1)
                ti.append(int(b[1]) - 1 if len(b) > 1 and b[1] else -1)
            g = tris.setdefault(cur, [])
            for j in range(1, len(vi) - 1):
                g.append(((vi[0], vi[j], vi[j + 1]), (ti[0], ti[j], ti[j + 1])))
    return np.array(V, np.float64), np.array(VT, np.float64), tris, mtllib


def parse_mtl(path):
    """Material name -> texture filename (first map_Kd / map_Ke)."""
    tex, cur = {}, None
    if not os.path.isfile(path):
        return tex
    for ln in open(path, errors="ignore"):
        p = ln.split()
        if not p:
            continue
        if p[0] == "newmtl":
            cur = p[1]
        elif p[0] in ("map_Kd", "map_Ke") and cur:
            tex.setdefault(cur, p[-1])
    return tex


def project(P, pos, look, W, H, focal_mm, sensor_mm):
    """World points -> (u_px, v_px, depth) for a pinhole camera at pos looking at look. depth>0 is in front."""
    f = look - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, np.array([0.0, 0.0, 1.0]))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    d = P - pos
    zc, xc, yc = d @ f, d @ r, d @ u
    fx = W * focal_mm / sensor_mm
    with np.errstate(divide="ignore", invalid="ignore"):
        return W * 0.5 + fx * xc / zc, H * 0.5 - fx * yc / zc, zc


def rasterize(V, VT, tris, mats, cam, SW, SH):
    """Z-buffer every triangle into the camera; per pixel: material index and perspective-correct uv."""
    zbuf = np.full((SH, SW), np.inf)
    mbuf = np.full((SH, SW), -1, np.int32)
    ubuf = np.zeros((SH, SW))
    vbuf = np.zeros((SH, SW))
    up_all, vp_all, z_all = project(V, cam["pos"], cam["look"], SW, SH, cam["lens"], cam["sensor"])
    for mi, mat in enumerate(mats):
        for vi, ti in tris[mat]:
            z0, z1, z2 = z_all[vi[0]], z_all[vi[1]], z_all[vi[2]]
            if z0 <= 1e-6 or z1 <= 1e-6 or z2 <= 1e-6:
                continue  # behind / on the camera plane
            x0, x1, x2 = up_all[vi[0]], up_all[vi[1]], up_all[vi[2]]
            y0, y1, y2 = vp_all[vi[0]], vp_all[vi[1]], vp_all[vi[2]]
            xmin = int(max(np.floor(min(x0, x1, x2)), 0))
            xmax = int(min(np.ceil(max(x0, x1, x2)), SW - 1))
            ymin = int(max(np.floor(min(y0, y1, y2)), 0))
            ymax = int(min(np.ceil(max(y0, y1, y2)), SH - 1))
            if xmin > xmax or ymin > ymax:
                continue
            den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(den) < 1e-12:
                continue
            gx, gy = np.meshgrid(np.arange(xmin, xmax + 1) + 0.5, np.arange(ymin, ymax + 1) + 0.5)
            w0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / den
            w1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / den
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
            if not inside.any():
                continue
            iz = w0 / z0 + w1 / z1 + w2 / z2
            zi = np.where(iz != 0, 1.0 / iz, np.inf)
            sub = zbuf[ymin : ymax + 1, xmin : xmax + 1]
            better = inside & (zi < sub)
            if not better.any():
                continue
            if ti[0] >= 0 and ti[1] >= 0 and ti[2] >= 0:
                tu = (w0 * VT[ti[0], 0] / z0 + w1 * VT[ti[1], 0] / z1 + w2 * VT[ti[2], 0] / z2) * zi
                tv = (w0 * VT[ti[0], 1] / z0 + w1 * VT[ti[1], 1] / z1 + w2 * VT[ti[2], 1] / z2) * zi
            else:
                tu = np.zeros_like(zi)
                tv = np.zeros_like(zi)
            sub[better] = zi[better]
            mbuf[ymin : ymax + 1, xmin : xmax + 1][better] = mi
            ubuf[ymin : ymax + 1, xmin : xmax + 1][better] = tu[better]
            vbuf[ymin : ymax + 1, xmin : xmax + 1][better] = tv[better]
    return mbuf, ubuf, vbuf


def main():
    import cv2

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obj", required=True, help="baked multi-material OBJ")
    ap.add_argument("--multi", required=True, help="JSON list of {gt, pos, look}: the GT frames and their cameras")
    ap.add_argument("--out", required=True, help="output dir for the projected asset")
    ap.add_argument("--ss", type=int, default=4, help="screen supersample; higher = more texels get a measured pixel")
    a = ap.parse_args()
    anchor = json.load(open(os.environ["ANCHOR_JSON"]))  # lens, sensor and GT frame size
    W, H = (int(v) for v in anchor.get("res", (1000, 625)))
    lens, sensor = float(anchor.get("lens", 35.0)), float(anchor.get("sensor", 36.0))
    SW, SH = W * a.ss, H * a.ss
    T0 = time.time()
    texout = os.path.join(a.out, "tex")
    os.makedirs(texout, exist_ok=True)

    objdir = os.path.dirname(os.path.abspath(a.obj))
    V, VT, tris, mtllib = parse_obj(a.obj)
    mtl_tex = parse_mtl(os.path.join(objdir, mtllib)) if mtllib else {}
    views = json.load(open(a.multi))
    if not views:
        raise SystemExit(f"--multi {a.multi} is empty")
    print(f"[pg] MULTI-VIEW: {len(views)} cameras, {len(V)} verts, {len(tris)} materials", flush=True)

    mats = list(tris.keys())
    tex_img = {}
    for mat in mats:
        tf = mtl_tex.get(mat)
        src = os.path.normpath(os.path.join(objdir, tf)) if tf else None
        img = cv2.imread(src) if src and os.path.isfile(src) else np.full((256, 256, 3), 128, np.uint8)
        tex_img[mat] = img.astype(np.float64)
    # Each texel takes the mean colour from the ONE view that hit it with the most screen samples.
    best_col = {m: np.zeros((t.shape[0] * t.shape[1], 3)) for m, t in tex_img.items()}
    best_w = {m: np.zeros(t.shape[0] * t.shape[1]) for m, t in tex_img.items()}
    for vw_i, e in enumerate(views):
        cam = dict(pos=np.array(e["pos"], dtype=float), look=np.array(e["look"], dtype=float), lens=lens, sensor=sensor)
        gt = cv2.imread(e["gt"])
        if gt is None:
            print(f"[pg] view {vw_i}: cannot read GT {e['gt']}, skipping", flush=True)
            continue
        if gt.shape[1] != W or gt.shape[0] != H:
            gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_AREA)
        gts = cv2.resize(gt, (SW, SH), interpolation=cv2.INTER_LINEAR)
        mbuf, ubuf, vbuf = rasterize(V, VT, tris, mats, cam, SW, SH)
        print(f"[pg] view {vw_i + 1}/{len(views)} screen coverage {float((mbuf >= 0).mean()):.3f}", flush=True)
        for mi, mat in enumerate(mats):
            sel = mbuf == mi
            if not sel.any():
                continue
            th, tw = tex_img[mat].shape[:2]
            px = np.clip((ubuf[sel] * tw).astype(np.int64), 0, tw - 1)
            py = np.clip(((1.0 - vbuf[sel]) * th).astype(np.int64), 0, th - 1)
            flat = py * tw + px
            vc = np.zeros_like(best_w[mat])
            va = np.zeros_like(best_col[mat])
            np.add.at(vc, flat, 1.0)
            np.add.at(va, flat, gts[sel].astype(np.float64))
            better = (vc > 0) & (vc > best_w[mat])
            if better.any():
                best_col[mat][better] = va[better] / vc[better][:, None]
                best_w[mat][better] = vc[better]

    total_w = total_t = 0
    for mat in mats:  # unmeasured texels keep the bake
        th, tw = tex_img[mat].shape[:2]
        hit = best_w[mat] > 0
        out = tex_img[mat].reshape(-1, 3).copy()
        out[hit] = best_col[mat][hit]
        total_w += int(hit.sum())
        total_t += hit.size
        tf = mtl_tex.get(mat)
        name = os.path.basename(tf) if tf else f"{mat}.png"
        cv2.imwrite(os.path.join(texout, name), np.clip(out.reshape(th, tw, 3), 0, 255).astype(np.uint8))

    shutil.copy(a.obj, os.path.join(a.out, "scan.obj"))
    if mtllib and os.path.isfile(os.path.join(objdir, mtllib)):
        txt = open(os.path.join(objdir, mtllib)).read()
        for tf in mtl_tex.values():
            txt = txt.replace(tf, f"tex/{os.path.basename(tf)}")
        open(os.path.join(a.out, mtllib), "w").write(txt)
    print(f"[pg] wrote {a.out}: {total_w}/{total_t} texels measured, rest keep the bake. {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
