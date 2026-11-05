#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Per-scene analysis pass for infinigen baked rooms."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field

import numpy as np

log = logging.getLogger("analyze_scene")

_NON_OBSTACLE_SUBSTR = (
    ".floor",
    ".wall",
    ".ceiling",
    ".exterior",
    "rugfactory",
    "wallartfactory",
    "mirrorfactory",
    "ceilinglightfactory",
    "windowfactory",
)
_STRUCTURAL_SUBSTR = (".floor", ".wall", ".ceiling", ".exterior", ".interior")
ROBOT_HEIGHT_M = 1.7
OBSTACLE_MIN_HEIGHT_M = 0.08  # >this off floor blocks
ROBOT_RADIUS_M = 0.35  # planar clearance the robot needs to stand/walk
WALL_MARGIN_M = 0.30  # keep spawn/sofa this far off the walls


@dataclass
class Asset:
    name: str  # full OBJ object name
    cls: str  # infinigen factory class (or room-structural token)
    bbox_min: list  # [x,y,z] metres
    bbox_max: list
    is_obstacle: bool
    is_structural: bool  # wall / floor / ceiling
    seed: str = ""  # infinigen instance/factory seed from the object name
    material: str = ""  # OBJ usemtl name (u0XX)
    size: list = field(default_factory=list)  # [w,d,h] metres (bbox extent)
    center: list = field(default_factory=list)  # [x,y,z] metres (bbox centre)
    texture: str = ""  # baked texture, scene-relative (e.g. asset/tex/obj007.png)
    color_hex: str = ""  # mean colour of the baked texture


@dataclass
class SceneAnalysis:
    scene: str
    scene_dir: str
    room: dict  # {W, D, H, floor_z}
    floor_z: float
    scene_bbox_min: list
    scene_bbox_max: list
    assets: list = field(default_factory=list)
    occupancy: dict = field(default_factory=dict)  # grid meta + base64-free nested list
    spawn: dict = field(default_factory=dict)  # {pos_zup, yaw}
    sofa: dict = field(default_factory=dict)  # {pos_zup, yaw}
    tpv_views: list = field(default_factory=list)  # ranked [{pos, look_at, score, kind}]
    notes: list = field(default_factory=list)


def _factory_class(name: str) -> str:
    """Extract the infinigen factory / structural token from an OBJ object name."""
    m = re.match(r"([A-Za-z0-9_\-]+Factory)", name)
    if m:
        return m.group(1)
    for tok in (".floor", ".wall", ".ceiling"):
        if tok in name:
            return tok.lstrip(".")
    m = re.match(r"([A-Za-z][A-Za-z0-9_\-]*)", name)
    return m.group(1) if m else name


def parse_assets(obj_path: str) -> tuple[list, np.ndarray, np.ndarray]:
    """Parse per-object axis-aligned bboxes from a multi-object OBJ."""
    verts: list[tuple[float, float, float]] = []
    obj_vidx: dict[str, set] = {}
    obj_mtl: dict[str, str] = {}
    order: list[str] = []
    cur = None
    with open(obj_path, errors="ignore") as f:
        for ln in f:
            if ln.startswith("v "):
                p = ln.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif ln.startswith("o "):
                cur = ln[2:].strip()
                obj_vidx[cur] = set()
                order.append(cur)
            elif ln.startswith("usemtl ") and cur is not None:
                obj_mtl[cur] = ln.split(None, 1)[1].strip()
            elif ln.startswith("f ") and cur is not None:
                for tok in ln.split()[1:]:
                    obj_vidx[cur].add(int(tok.split("/")[0]) - 1)
    V = np.asarray(verts, dtype=np.float64)
    if V.size == 0:
        raise ValueError(f"{obj_path}: no vertices parsed")

    assets: list[Asset] = []
    for name in order:
        idx = obj_vidx.get(name)
        if not idx:
            continue
        P = V[np.fromiter(idx, dtype=np.int64)]
        bmin, bmax = P.min(0), P.max(0)
        cls = _factory_class(name)
        structural = any(t in name.lower() for t in _STRUCTURAL_SUBSTR)
        non_obst = any(s in name.lower() for s in _NON_OBSTACLE_SUBSTR)
        sm = re.search(r"\((\d+)\)", name)  # first parenthesised id = factory seed
        assets.append(
            Asset(
                name=name,
                cls=cls,
                bbox_min=bmin.round(4).tolist(),
                bbox_max=bmax.round(4).tolist(),
                is_obstacle=(not non_obst),
                is_structural=structural,
                seed=(sm.group(1) if sm else ""),
                material=obj_mtl.get(name, ""),
                size=(bmax - bmin).round(4).tolist(),
                center=((bmin + bmax) / 2).round(4).tolist(),
            )
        )
    return assets, V.min(0), V.max(0)


def enrich_textures(scene_dir: str, assets: list) -> None:
    """Resolve each asset's baked texture (via the MTL usemtl→map_Kd map) and its mean colour."""
    asset_dir = os.path.join(scene_dir, "asset")
    mtl_path = os.path.join(asset_dir, "scan.mtl")
    mat_png: dict[str, str] = {}
    if os.path.isfile(mtl_path):
        cur = None
        with open(mtl_path, errors="ignore") as f:
            for ln in f:
                if ln.startswith("newmtl "):
                    cur = ln.split(None, 1)[1].strip()
                elif ln.startswith("map_Kd ") and cur:
                    mat_png[cur] = ln.split(None, 1)[1].strip()  # e.g. tex/obj007.png
    try:
        import imageio.v2 as imageio
    except Exception:
        imageio = None
    color_cache: dict[str, str] = {}
    for a in assets:
        rel = mat_png.get(a.material)
        if not rel:
            continue
        a.texture = os.path.join("asset", rel)  # scene-relative
        if imageio is None:
            continue
        if rel not in color_cache:
            png = os.path.join(asset_dir, rel)
            try:
                im = imageio.imread(png)
                arr = np.asarray(im)[..., :3].reshape(-1, 3).mean(0).astype(int)
                color_cache[rel] = "#%02x%02x%02x" % tuple(int(c) for c in arr)
            except Exception:
                color_cache[rel] = ""
        a.color_hex = color_cache[rel]


def find_floor_z(assets: list) -> tuple[float, list]:
    """Walkable floor height = z of the largest ``.floor`` object (fallback: baseboard/min-z)."""
    notes = []
    floors = [a for a in assets if ".floor" in a.name.lower()]
    if floors:

        def area(a):
            return (a.bbox_max[0] - a.bbox_min[0]) * (a.bbox_max[1] - a.bbox_min[1])

        f = max(floors, key=area)
        floor_z = float(round((f.bbox_min[2] + f.bbox_max[2]) / 2.0, 4))
        notes.append(f"floor_z={floor_z:.4f} from '{f.name}' (NOT the z=0 baseboard)")
    else:
        floor_z = float(min(a.bbox_min[2] for a in assets))
        notes.append(f"no .floor object found; floor_z={floor_z:.4f} from scene min-z (approx)")
    return floor_z, notes


def build_occupancy(assets, room_w, room_d, floor_z, cell=0.10):
    """Rasterize furniture footprints into a 2D occupied grid over [0,W]x[0,D]."""
    nx = max(1, int(np.ceil(room_w / cell)))
    ny = max(1, int(np.ceil(room_d / cell)))
    occ = np.zeros((ny, nx), dtype=bool)
    band_top = floor_z + ROBOT_HEIGHT_M
    room_area = room_w * room_d
    for a in assets:
        if not a.is_obstacle or a.is_structural:
            continue
        if (a.bbox_max[2] - floor_z) < OBSTACLE_MIN_HEIGHT_M:
            continue  # flat thing (rug); ignore
        if a.bbox_min[2] > band_top:
            continue  # entirely above the robot (hanging light); ignore
        foot = (a.bbox_max[0] - a.bbox_min[0]) * (a.bbox_max[1] - a.bbox_min[1])
        if foot > 0.9 * room_area:
            continue  # room-spanning shell that slipped structural detection; never an obstacle
        x0 = int(np.floor(a.bbox_min[0] / cell))
        x1 = int(np.ceil(a.bbox_max[0] / cell))
        y0 = int(np.floor(a.bbox_min[1] / cell))
        y1 = int(np.ceil(a.bbox_max[1] / cell))
        x0, x1 = max(0, x0), min(nx, x1)
        y0, y1 = max(0, y0), min(ny, y1)
        occ[y0:y1, x0:x1] = True

    m = int(np.ceil(WALL_MARGIN_M / cell))
    if m > 0:
        occ[:m, :] = True
        occ[-m:, :] = True
        occ[:, :m] = True
        occ[:, -m:] = True

    clearance = _clearance_map(occ, cell)
    return occ, clearance


def _clearance_map(occ, cell):
    """Metric distance from each cell to the nearest occupied cell (0 if occupied)."""
    ny, nx = occ.shape
    try:
        from scipy import ndimage

        dist = ndimage.distance_transform_edt(~occ) * cell
        return dist.astype(np.float32)
    except Exception:
        occ_yx = np.argwhere(occ)
        out = np.zeros((ny, nx), dtype=np.float32)
        if len(occ_yx) == 0:
            out[:] = max(nx, ny) * cell
            return out
        for j in range(ny):
            for i in range(nx):
                if occ[j, i]:
                    continue
                d = np.min(np.hypot(occ_yx[:, 0] - j, occ_yx[:, 1] - i)) * cell
                out[j, i] = d
        return out


def _cell_to_xy(j, i, cell):
    return ((i + 0.5) * cell, (j + 0.5) * cell)


def _footprint_fits(occ, cell, cx, cy, half_x, half_y, margin=0.10):
    """True iff an axis-aligned footprint (half_x,half_y + margin) around (cx,cy) is all-free."""
    ny, nx = occ.shape
    x0 = int(np.floor((cx - half_x - margin) / cell))
    x1 = int(np.ceil((cx + half_x + margin) / cell))
    y0 = int(np.floor((cy - half_y - margin) / cell))
    y1 = int(np.ceil((cy + half_y + margin) / cell))
    if x0 < 0 or y0 < 0 or x1 > nx or y1 > ny:
        return False
    return not occ[y0:y1, x0:x1].any()


def pick_spawn_and_sofa(occ, clearance, cell, floor_z, sofa_extent=(1.61, 1.27)):
    """Pick a collision-free spawn pose and (if room allows) a free sofa pose."""
    free = clearance >= ROBOT_RADIUS_M
    if not free.any():
        raise RuntimeError("no free cell with robot clearance; scene too cluttered")
    sj, si = np.unravel_index(np.argmax(np.where(free, clearance, -1)), clearance.shape)
    spawn_xy = np.array(_cell_to_xy(sj, si, cell))

    def yaw_toward(src, dst):
        d = dst - src
        return float(np.arctan2(d[1], d[0]))

    L, Wd = sofa_extent
    best, best_d = None, -1.0
    ny, nx = occ.shape
    for j in range(ny):
        for i in range(nx):
            xy = np.array(_cell_to_xy(j, i, cell))
            d = float(np.linalg.norm(xy - spawn_xy))
            if not (1.0 <= d <= 3.2):
                continue
            yaw = yaw_toward(xy, spawn_xy)
            c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
            half_x = 0.5 * (L * c + Wd * s)  # rotated AABB half-extents
            half_y = 0.5 * (L * s + Wd * c)
            if not _footprint_fits(occ, cell, xy[0], xy[1], half_x, half_y):
                continue
            if d > best_d:
                best_d, best = d, (xy, yaw)

    spawn = {"pos_zup": [float(spawn_xy[0]), float(spawn_xy[1]), floor_z], "yaw": 0.0}
    if best is None:
        center = np.array([occ.shape[1] * cell / 2.0, occ.shape[0] * cell / 2.0])
        spawn["yaw"] = yaw_toward(spawn_xy, center)
        return spawn, None

    sofa_xy, sofa_yaw = best
    spawn["yaw"] = yaw_toward(spawn_xy, sofa_xy)
    sofa = {"pos_zup": [float(sofa_xy[0]), float(sofa_xy[1]), floor_z], "yaw": float(sofa_yaw)}
    return spawn, sofa


def rank_tpv_views(scene_dir, top_k=5):
    """Rank the pre-scored candidate cameras in views_64.json for use as a TPV."""
    vp = os.path.join(scene_dir, "views_64.json")
    ranked = []
    if os.path.isfile(vp):
        v = json.load(open(vp))
        for x in v.get("views", []):
            content = x.get("content", 0.0)
            hit = x.get("hit_frac", 0.0)
            trans = x.get("transmissive_frac", 0.0)
            kind_bonus = 0.1 if x.get("kind") == "interior" else 0.0
            score = content * hit - trans + kind_bonus
            ranked.append(
                {
                    "pos": x["pos"],
                    "look_at": x["look_at"],
                    "kind": x.get("kind", "?"),
                    "score": round(float(score), 4),
                    "content": content,
                    "hit_frac": hit,
                    "transmissive_frac": trans,
                }
            )
        ranked.sort(key=lambda r: r["score"], reverse=True)
        ranked = ranked[:top_k]
    ap = os.path.join(scene_dir, "anchor.json")
    if os.path.isfile(ap):
        a = json.load(open(ap))
        ranked.append(
            {
                "pos": a["pos"],
                "look_at": a["look"],
                "kind": "anchor",
                "score": 0.0,
                "content": None,
                "hit_frac": None,
                "transmissive_frac": None,
            }
        )
    return ranked


def synth_tpv_for_bbox(center_xy, floor_z, target_z=0.4, dist=3.4, height=2.7):
    """Fallback TPV that frames a target point (e.g. the sit location) from a fixed offset."""
    cx, cy = center_xy
    eye = [cx - 1.6, cy - 3.4, floor_z + height]
    look = [cx, cy, floor_z + target_z]
    return {"pos": eye, "look_at": look, "kind": "synth", "score": None}


def draw_debug_png(analysis: SceneAnalysis, occ, clearance, cell, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    W = analysis.room["W"]
    D = analysis.room["D"]
    fig, ax = plt.subplots(figsize=(8, 8 * D / max(W, 1e-6)))
    ax.imshow(
        clearance,
        origin="lower",
        extent=[0, occ.shape[1] * cell, 0, occ.shape[0] * cell],
        cmap="Greens",
        alpha=0.6,
        aspect="equal",
    )
    ax.imshow(
        np.where(occ, 1.0, np.nan),
        origin="lower",
        extent=[0, occ.shape[1] * cell, 0, occ.shape[0] * cell],
        cmap="Reds",
        alpha=0.35,
        aspect="equal",
        vmin=0,
        vmax=1,
    )
    for a in analysis.assets:
        if a["is_structural"]:
            continue
        x0, y0 = a["bbox_min"][0], a["bbox_min"][1]
        w, h = a["bbox_max"][0] - x0, a["bbox_max"][1] - y0
        is_sofa = "sofafactory" in a["name"].lower()
        ax.add_patch(
            Rectangle(
                (x0, y0), w, h, fill=False, edgecolor="purple" if is_sofa else "black", lw=1.6 if is_sofa else 0.7
            )
        )
        ax.text(x0, y0, a["cls"].replace("Factory", ""), fontsize=5, color="dimgray")

    def _arrow(pose, color, label):
        x, y, _ = pose["pos_zup"]
        yaw = pose["yaw"]
        ax.plot(x, y, "o", color=color, ms=10)
        ax.arrow(x, y, 0.5 * np.cos(yaw), 0.5 * np.sin(yaw), head_width=0.12, color=color, length_includes_head=True)
        ax.text(x, y + 0.15, label, color=color, fontsize=9, weight="bold")

    if analysis.spawn:
        _arrow(analysis.spawn, "blue", "spawn")
    if analysis.sofa:
        _arrow(analysis.sofa, "purple", "sofa")
    for k, v in enumerate(analysis.tpv_views[:3]):
        px, py = v["pos"][0], v["pos"][1]
        lx, ly = v["look_at"][0], v["look_at"][1]
        ax.plot([px, lx], [py, ly], "--", color="orange", lw=1.0, alpha=0.8)
        ax.plot(px, py, "^", color="orange", ms=7)
        ax.text(px, py, f"TPV{k}", color="darkorange", fontsize=7)

    ax.set_xlim(-0.3, W + 0.3)
    ax.set_ylim(-0.3, D + 0.3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"{analysis.scene}  W={W:.1f} D={D:.1f}  floor_z={analysis.floor_z:.3f} m\n"
        f"green=clearance red=obstacle  blue=spawn purple=sofa orange=TPV"
    )
    ax.grid(True, lw=0.2, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def analyze_scene(scene_dir, cell=0.10, top_k=5, write=True, debug_png=True):
    """Analyze one baked room (``<scene_dir>/asset/scan.obj`` + sidecars) into scene_analysis.json + a top-down PNG."""
    scene_dir = os.path.expanduser(scene_dir)
    scene = os.path.basename(scene_dir.rstrip("/"))
    obj_path = os.path.join(scene_dir, "asset", "scan.obj")
    if not os.path.isfile(obj_path):
        raise FileNotFoundError(obj_path)
    log.info("[%s] parsing %s", scene, obj_path)
    assets, bmin, bmax = parse_assets(obj_path)
    enrich_textures(scene_dir, assets)
    floor_z, fnotes = find_floor_z(assets)

    room = {}
    vp = os.path.join(scene_dir, "views_64.json")
    if os.path.isfile(vp):
        room = dict(json.load(open(vp)).get("room", {}))
    W = room.get("W") or float(bmax[0] - bmin[0])
    D = room.get("D") or float(bmax[1] - bmin[1])
    H = room.get("H") or float(bmax[2] - bmin[2])
    room = {"W": round(W, 3), "D": round(D, 3), "H": round(H, 3), "floor_z": floor_z}

    occ, clearance = build_occupancy(assets, W, D, floor_z, cell=cell)
    spawn, sofa = pick_spawn_and_sofa(occ, clearance, cell, floor_z)
    tpv = rank_tpv_views(scene_dir, top_k=top_k)
    focus = sofa if sofa else spawn
    tpv.append(synth_tpv_for_bbox((focus["pos_zup"][0], focus["pos_zup"][1]), floor_z))

    n_obst = sum(
        1
        for a in assets
        if a.is_obstacle and not a.is_structural and (a.bbox_max[2] - floor_z) >= OBSTACLE_MIN_HEIGHT_M
    )
    notes = fnotes + [
        f"{len(assets)} objects, {n_obst} counted as obstacles",
        f"occupancy grid {occ.shape} @ {cell} m/cell, {int((~occ).sum())} free cells",
        f"free area {(clearance >= ROBOT_RADIUS_M).sum() * cell * cell:.1f} m^2 with robot clearance",
    ]
    for nt in notes:
        log.info("[%s] %s", scene, nt)

    analysis = SceneAnalysis(
        scene=scene,
        scene_dir=scene_dir,
        room=room,
        floor_z=floor_z,
        scene_bbox_min=bmin.round(4).tolist(),
        scene_bbox_max=bmax.round(4).tolist(),
        assets=[asdict(a) for a in assets],
        occupancy={
            "cell": cell,
            "shape": list(occ.shape),
            "n_free": int((~occ).sum()),
            "free_area_m2": round(float((clearance >= ROBOT_RADIUS_M).sum() * cell * cell), 2),
            "grid": occ.astype(np.uint8).tolist(),
        },
        spawn=spawn,
        sofa=(sofa or {}),
        tpv_views=tpv,
        notes=notes,
    )

    if write:
        out_json = os.path.join(scene_dir, "scene_analysis.json")
        with open(out_json, "w") as f:
            json.dump(asdict(analysis), f, indent=1)
        log.info("[%s] wrote %s", scene, out_json)
    if debug_png:
        out_png = os.path.join(scene_dir, "scene_analysis_topdown.png")
        draw_debug_png(analysis, occ, clearance, cell, out_png)
        log.info("[%s] wrote %s", scene, out_png)
    return analysis


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene-dir", required=True, help="one scene dir (with asset/scan.obj)")
    ap.add_argument("--cell", type=float, default=0.10, help="occupancy cell size (m)")
    ap.add_argument("--top-k", type=int, default=5, help="TPV views to keep after ranking")
    args = ap.parse_args()
    try:  # exit 0 regardless: scene_gen ships the scene and logs "no scene_analysis.json"
        analyze_scene(args.scene_dir, cell=args.cell, top_k=args.top_k)
    except Exception as e:
        log.error("FAILED %s: %s", args.scene_dir, e)


if __name__ == "__main__":
    main()
