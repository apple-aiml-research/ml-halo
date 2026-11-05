#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""prep_and_gt.py -- one Blender load: measure the room, sample 64 views, render the GT frames."""

import bpy, sys, os, json, math, time
from mathutils import Vector, Matrix

T0 = time.time()
OUT, SCENE_NAME, N_INTERIOR, N_ORBIT, SAMPLES, RES, SEED = sys.argv[sys.argv.index("--") + 1 :][:7]
N_INTERIOR, N_ORBIT, SAMPLES, SEED = int(N_INTERIOR), int(N_ORBIT), int(SAMPLES), int(SEED)
RW, RH = (int(v) for v in RES.split("x"))
LENS, SENSOR = 35.0, 36.0
GT_DIR = os.path.join(OUT, "gt")
os.makedirs(GT_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scene_gen import viewpoints as vp
from scene_gen.blender_util import cycles_gpu, show_all_meshes


def log(*a):
    print(f"[prep {time.time() - T0:7.1f}s]", *a, flush=True)


sc = bpy.context.scene
show_all_meshes()
bpy.context.view_layer.update()


def world_bbox(objs):
    pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
    return [min(p[i] for p in pts) for i in range(3)], [max(p[i] for p in pts) for i in range(3)]


meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.data.vertices]
if not meshes:
    raise SystemExit("prep_and_gt: no mesh geometry in the scene")
(x0, y0, z0), (x1, y1, z1) = world_bbox(meshes)
# The room is the '.floor' objects' footprint (the scene bbox includes the exterior shell); its floor
# height is the lowest floor top (the global min z sits under the slab).
floor_objs = [o for o in meshes if ".floor" in o.name.lower()]
if floor_objs:
    floor_z = min(max((o.matrix_world @ Vector(c)).z for c in o.bound_box) for o in floor_objs)
    (rx0, ry0, _), (rx1, ry1, _) = world_bbox(floor_objs)
else:
    log(f"WARN no '.floor' object found; using the scene bbox (floor_z = min z = {z0:.3f})")
    floor_z, rx0, rx1, ry0, ry1 = z0, x0, x1, y0, y1
W_out, D_out, H_out = rx1 - rx0, ry1 - ry0, z1 - floor_z
log(f"ROOM {W_out:.2f} x {D_out:.2f} x {H_out:.2f} (floor_z={floor_z:.3f})")

ORIGIN = (rx0, ry0, floor_z)
room = {"W": round(W_out, 3), "D": round(D_out, 3), "H": round(H_out, 3), "floor_z": 0.0}


def to_world(p):
    return [round(p[0] + ORIGIN[0], 4), round(p[1] + ORIGIN[1], 4), round(p[2] + ORIGIN[2], 4)]


cx, cy = W_out / 2.0, D_out / 2.0
eye_back = min(2.5, max(0.6, D_out / 2 - 0.6))
anchor_local = {"pos": [cx, max(0.6, cy - eye_back), vp.CAM_H], "look": [cx, min(D_out - 0.3, cy + 0.5), 0.70]}
anchor = dict(name=SCENE_NAME, pos=to_world(anchor_local["pos"]), look=to_world(anchor_local["look"]), lens=LENS)
anchor.update(sensor=SENSOR, res=[RW, RH], keep_radius=round(0.5 * math.hypot(W_out, D_out) + 1.5, 3))
json.dump(anchor, open(os.path.join(OUT, "anchor.json"), "w"), indent=1)
log(f"anchor: pos={anchor['pos']} look={anchor['look']} keep_radius={anchor['keep_radius']}")

dg = bpy.context.evaluated_depsgraph_get()
_raw = vp.blender_raycast(sc, dg)


def raycast_local(origin, direction, max_dist):
    return _raw((origin[0] + ORIGIN[0], origin[1] + ORIGIN[1], origin[2] + ORIGIN[2]), direction, max_dist)


doc = vp.sample_views(room, raycast_local, anchor_local, N_INTERIOR, N_ORBIT, seed=SEED)
for v in doc["views"]:
    v["pos"] = to_world(v["pos"])
    v["look_at"] = to_world(v["look_at"])
doc["scene"] = SCENE_NAME
doc["origin"] = [round(c, 4) for c in ORIGIN]
doc["room"]["H"] = room["H"]


def view_dependence(o):
    """max(transmission, metallic * (1 - roughness)) over the object's unlinked Principled inputs."""
    best = 0.0
    for mat in o.data.materials or []:
        if not mat or not mat.use_nodes or not mat.node_tree:
            continue
        for nd in mat.node_tree.nodes:
            if nd.type != "BSDF_PRINCIPLED":
                continue

            def val(name, dflt):
                inp = nd.inputs.get(name)
                if inp is None or inp.is_linked:
                    return dflt
                try:
                    return float(inp.default_value)
                except Exception:
                    return dflt

            tr = max(val("Transmission", 0.0), val("Transmission Weight", 0.0))
            best = max(best, tr, val("Metallic", 0.0) * (1.0 - val("Roughness", 0.5)))
    return best


# transmissive_frac (fraction of the frustum fan whose first hit is glass/mirror/metal) is read by
# analyze_scene to rank TPV cameras; it does not change the sampling.
try:
    trans = {o.name for o in bpy.data.objects if o.type == "MESH" and o.data and view_dependence(o) > 0.3}
    rc_obj = vp.blender_raycast(sc, dg, want_object=True)
    if trans:
        for v in doc["views"]:
            eye = [v["pos"][i] - ORIGIN[i] for i in range(3)]
            look = [v["look_at"][i] - ORIGIN[i] for i in range(3)]
            origin = (eye[0] + ORIGIN[0], eye[1] + ORIGIN[1], eye[2] + ORIGIN[2])
            hits = [rc_obj(origin, d, vp.MAX_HIT) for d in vp.frustum_dirs(eye, look)]
            hits = [h for h in hits if h is not None]
            v["transmissive_frac"] = round(sum(h in trans for h in hits) / len(hits), 3) if hits else 0.0
except Exception as e:
    log(f"transmissive_frac skipped ({e})")

json.dump(doc, open(os.path.join(OUT, f"views_{N_INTERIOR + N_ORBIT}.json"), "w"), indent=1)
log(f"views: {len(doc['views'])}, n_interior_actual={doc['n_interior_actual']}/{N_INTERIOR}")

sc.render.engine = "CYCLES"
sc.cycles.samples = SAMPLES
sc.cycles.use_denoising = True
cycles_gpu(sc, log)
sc.render.resolution_x, sc.render.resolution_y = RW, RH
sc.render.image_settings.file_format = "PNG"
try:
    sc.view_settings.view_transform = "AgX"
except Exception:
    pass


def look_at(p, t, up=(0, 0, 1)):
    p, t, up = Vector(p), Vector(t), Vector(up)
    f = (t - p).normalized()
    r = f.cross(up).normalized()
    u = r.cross(f).normalized()
    z = -f
    return Matrix(((r.x, u.x, z.x, p.x), (r.y, u.y, z.y, p.y), (r.z, u.z, z.z, p.z), (0, 0, 0, 1)))


cd = bpy.data.cameras.new("gtcam")
cd.lens, cd.sensor_width, cd.sensor_fit = LENS, SENSOR, "HORIZONTAL"
co = bpy.data.objects.new("gtcam", cd)
sc.collection.objects.link(co)
sc.camera = co


def azimuth(i):
    v = doc["views"][i]
    return math.atan2(v["look_at"][1] - v["pos"][1], v["look_at"][0] - v["pos"][0])


def spread_by_azimuth(cands, k):
    """Pick k views whose look directions are as far apart as possible (farthest-point in angle)."""
    if k <= 0 or not cands:
        return []
    if k >= len(cands):
        return list(cands)
    chosen = [cands[0]]
    while len(chosen) < k:
        best, best_gap = None, -1.0
        for i in cands:
            if i in chosen:
                continue
            gap = min(
                abs(math.atan2(math.sin(azimuth(i) - azimuth(j)), math.cos(azimuth(i) - azimuth(j)))) for j in chosen
            )
            if gap > best_gap:
                best, best_gap = i, gap
        if best is None:
            break
        chosen.append(best)
    return chosen


def stratified(cands, k):
    """Pick k views spread by azimuth, proportionally across view kind."""
    by_kind = {}
    for i in cands:
        by_kind.setdefault(doc["views"][i].get("kind", "interior"), []).append(i)
    out = []
    for kind, members in sorted(by_kind.items()):
        share = max(1, round(k * len(members) / (len(cands) or 1))) if k > 0 else 0
        out += spread_by_azimuth(members, min(share, len(members)))
    out = out[:k]
    if len(out) < k:
        out += spread_by_azimuth([i for i in cands if i not in set(out)], k - len(out))
    return out


n_total = len(doc["views"])
gt_idx = sorted(stratified(list(range(n_total)), int(os.environ.get("N_PROJ_VIEWS", "0") or 0)))
log(f"GT views: {gt_idx} at {SAMPLES} spp {RW}x{RH}")
for i in gt_idx:
    v = doc["views"][i]
    fp = os.path.join(GT_DIR, f"gt_v{i:02d}.png")
    co.matrix_world = look_at(v["pos"], v["look_at"])
    sc.render.filepath = fp
    bpy.ops.render.render(write_still=True)
    sig = [[round(c, 5) for c in v["pos"]], [round(c, 5) for c in v["look_at"]], SAMPLES, RW, RH]
    open(fp + ".pose", "w").write(json.dumps(sig))

man = dict(scene=SCENE_NAME, gt_views=gt_idx, proj_views=gt_idx, samples=SAMPLES, res=[RW, RH], room=doc["room"])
man.update({k: doc[k] for k in ("origin", "n_interior_actual")}, n_views=n_total, orbit_radius=doc["orbit_radius"])
json.dump(man, open(os.path.join(OUT, "prep_manifest.json"), "w"), indent=1)
log(f"PREP_OK {SCENE_NAME} views={n_total} gt={len(gt_idx)} total={time.time() - T0:.1f}s")
