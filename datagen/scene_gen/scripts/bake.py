#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""bake.py -- per-object Cycles bake with a two-phase emission swap."""

import bpy, sys, os, math, json, time
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scene_gen.blender_util import cycles_gpu, show_all_meshes

try:
    import cv2  # optional: cosmetic denoise only
except Exception:
    cv2 = None

argv = sys.argv[sys.argv.index("--") + 1 :]
OUT, SAMP, MAXRES = argv[0], int(argv[1]), int(argv[2])
TEXDIR = os.path.join(OUT, "tex")
BAKE_UV = "uv_bake"
MAXF, REVERT_MAXF = 12000, 400000  # decimate above MAXF faces; revert a lossy decimation up to REVERT_MAXF
DARK_GATE, COV_GATE = 8.0, 0.02  # a texture darker / emptier than this goes to the next fallback
os.makedirs(TEXDIR, exist_ok=True)
T0 = time.time()
ANCHOR = json.load(open(os.environ["ANCHOR_JSON"]))
LOOK, KEEP_RADIUS = tuple(float(v) for v in ANCHOR["look"]), float(ANCHOR.get("keep_radius", 8.0))


def log(*a):
    print(f"[bake {time.time() - T0:7.1f}s]", *a, flush=True)


sc = bpy.context.scene
vl = bpy.context.view_layer
show_all_meshes()


def world_center(o):
    """World-space bbox center of the evaluated object (the authored bbox if it evaluates empty)."""
    oe, has = o.evaluated_get(bpy.context.evaluated_depsgraph_get()), False
    try:
        has = len(oe.to_mesh().vertices) > 0
        oe.to_mesh_clear()
    except Exception:
        pass
    pts = [o.matrix_world @ Vector(c) for c in (oe.bound_box if has else o.bound_box)]
    mn = Vector([min(p[i] for p in pts) for i in range(3)])
    mx = Vector([max(p[i] for p in pts) for i in range(3)])
    return (mn + mx) / 2


def render_hidden_by_collection(o):
    """True if any ancestor collection is render-hidden."""
    parent = {child.name: coll for coll in bpy.data.collections for child in coll.children}
    for cur in o.users_collection:
        seen = set()
        while cur is not None and cur.name not in seen:
            seen.add(cur.name)
            if cur.hide_render:
                return True
            cur = parent.get(cur.name)
    return False


keep = []
for o in list(bpy.data.objects):
    if o.type != "MESH" or len(o.data.polygons) == 0 or "placeholder" in o.name.lower():
        continue
    c = world_center(o)
    if ((c.x - LOOK[0]) ** 2 + (c.y - LOOK[1]) ** 2) ** 0.5 > KEEP_RADIUS or c.z > LOOK[2] + 3.0:
        continue
    if not render_hidden_by_collection(o):
        keep.append(o)
log(f"keeping {len(keep)} meshes")

sc.render.engine = "CYCLES"
sc.cycles.use_denoising = True
sc.cycles.max_bounces = 12
sc.cycles.diffuse_bounces = 8
cycles_gpu(sc, log)
sc.cycles.samples = SAMP
sc.render.bake.use_selected_to_active = False
sc.render.bake.use_clear = True
sc.view_settings.view_transform = "AgX"
sc.view_settings.look = "None"


def surface_area(o):
    """True world-space surface area in m^2."""
    mw, mesh, total = o.matrix_world, o.data, 0.0
    for p in mesh.polygons:
        vs = [mw @ mesh.vertices[i].co for i in p.vertices]
        for j in range(1, len(vs) - 1):
            total += (vs[j] - vs[0]).cross(vs[j + 1] - vs[0]).length * 0.5
    return total


def pick_res(area):
    """Texture resolution from world area, targeting a roughly constant texel density."""
    e = int(round(math.log2(max(1024.0 * math.sqrt(max(area, 1e-4)), 1.0))))
    return int(min(max(2**e, 256), MAXRES))


def unwrap(o, margin):
    """Create the bake-target UV atlas without destroying the object's authored UVs."""
    src = next((ol.name for ol in o.data.uv_layers if ol.active_render), None)
    if src is None and len(o.data.uv_layers):
        src = o.data.uv_layers[0].name
    for ol in list(o.data.uv_layers):
        if ol.name != src and ol.name.startswith(BAKE_UV):
            o.data.uv_layers.remove(ol)
    while len(o.data.uv_layers) >= 8:  # Blender's UV-layer limit
        spare = next((x for x in o.data.uv_layers if x.name != src), None)
        if spare is None:
            break
        o.data.uv_layers.remove(spare)

    ul = o.data.uv_layers.new(name=BAKE_UV)
    o.data.uv_layers.active = ul
    ul.active_render = True
    for x in bpy.data.objects:
        try:
            x.select_set(False)
        except Exception:
            pass
    o.select_set(True)
    vl.objects.active = o
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=margin)
    bpy.ops.object.mode_set(mode="OBJECT")
    return src


def pin_source_uvs(o, src_uv):
    """Point every UV consumer in the material at `src_uv` instead of the active (atlas) layer."""
    n_img = n_coord = n_texnodes = 0
    if not src_uv:
        return n_img, n_coord, n_texnodes
    for mat in o.data.materials:
        if mat is None or not mat.use_nodes or not mat.node_tree:
            continue
        nt = mat.node_tree
        for n in list(nt.nodes):
            if n.type == "TEX_IMAGE":
                n_texnodes += 1
                vec = n.inputs.get("Vector")
                if vec is None or vec.is_linked:
                    continue
                uvn = nt.nodes.new("ShaderNodeUVMap")
                uvn.uv_map = src_uv
                uvn.location = (n.location.x - 220, n.location.y)
                nt.links.new(uvn.outputs["UV"], vec)
                n_img += 1
            elif n.type == "TEX_COORD":
                out = n.outputs.get("UV")
                if out is None or not out.is_linked:
                    continue
                uvn = nt.nodes.new("ShaderNodeUVMap")
                uvn.uv_map = src_uv
                uvn.location = (n.location.x, n.location.y - 160)
                for lk in list(out.links):
                    tgt = lk.to_socket
                    nt.links.remove(lk)
                    nt.links.new(uvn.outputs["UV"], tgt)
                    n_coord += 1
            elif n.type == "UVMAP" and n.uv_map == "":
                n.uv_map = src_uv
                n_coord += 1
    return n_img, n_coord, n_texnodes


def specular_score(o):
    """How view-dependent the object looks, in [0, 1]: max(transmission, metallic * (1 - roughness)). Diagnostic."""
    best = 0.0
    for mat in o.data.materials or []:
        if not mat or not mat.use_nodes or not mat.node_tree:
            continue
        for node in mat.node_tree.nodes:
            if node.type != "BSDF_PRINCIPLED":
                continue

            def val(name, default):
                s = node.inputs.get(name)
                if s is None or s.is_linked:
                    return default
                try:
                    return float(s.default_value)
                except Exception:
                    return default

            best = max(best, val("Transmission", 0.0), val("Metallic", 0.0) * (1.0 - val("Roughness", 0.5)))
    return min(1.0, max(0.0, best))


def attach_target(o, img):
    """Add the bake-target image node to every material (creating one if absent)."""
    if not o.data.materials:
        m = bpy.data.materials.new(f"m_{o.name}")
        m.use_nodes = True
        o.data.materials.append(m)
    for mat in o.data.materials:
        if mat is None:
            continue
        if not mat.use_nodes:
            mat.use_nodes = True
        n = mat.node_tree.nodes.new("ShaderNodeTexImage")
        n.image = img
        n.select = True
        mat.node_tree.nodes.active = n


def bake_to(o, img, tp, mode):
    """Bake `o` in `mode` (COMBINED: lit; DIFFUSE: colour only), save to tp; return (coverage, mean)."""
    sc.cycles.bake_type = mode
    sc.render.bake.use_pass_direct = sc.render.bake.use_pass_indirect = mode == "COMBINED"
    sc.render.bake.use_pass_color = True
    with bpy.context.temp_override(active_object=o, object=o, selected_objects=[o], selected_editable_objects=[o]):
        bpy.ops.object.bake(type=sc.cycles.bake_type)
    img.save_render(filepath=tp, scene=sc)
    return tex_stats(tp)


def tex_stats(path):
    """(coverage, mean-of-covered) for an on-disk texture."""
    img = bpy.data.images.load(path)
    w, h = img.size
    rgb = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)[..., :3]
    bpy.data.images.remove(img)
    lit = rgb.sum(-1) > 0.01
    return float(lit.mean()), float(rgb[lit].mean() * 255) if lit.any() else 0.0


def flat_basecolor(o):
    """Material base color as an RGB tuple, for the last-resort fallback."""
    for mat in o.data.materials or []:
        if not mat or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                c = node.inputs["Base Color"].default_value
                return (c[0], c[1], c[2])
    return (0.5, 0.5, 0.5)


log("=== PHASE A: bake all objects with ORIGINAL materials ===")
records, baked = [], []
for oi, o in enumerate(keep):
    t_obj = time.time()
    area_true = surface_area(o)
    nf = len(o.data.polygons)
    if nf > MAXF:
        backup = o.data.copy()
        d = o.modifiers.new("dec", "DECIMATE")
        d.ratio = MAXF / nf
        with bpy.context.temp_override(object=o, active_object=o):
            bpy.ops.object.modifier_apply(modifier="dec")
        if surface_area(o) < 0.95 * area_true and nf <= REVERT_MAXF:  # decimation ate surface: keep all faces
            old = o.data
            o.data = backup
            bpy.data.meshes.remove(old)
        else:
            bpy.data.meshes.remove(backup)
    o.modifiers.new("tri", "TRIANGULATE")
    with bpy.context.temp_override(object=o, active_object=o):
        bpy.ops.object.modifier_apply(modifier="tri")

    area = surface_area(o)
    res = pick_res(area)
    sc.render.bake.margin = max(4, res // 128)
    src_uv = unwrap(o, 0.02)
    n_pin_img, n_pin_coord, n_texnodes = pin_source_uvs(o, src_uv)
    img = bpy.data.images.new(f"t{oi}", res, res, float_buffer=True)
    attach_target(o, img)

    # fallbacks for an empty/black bake: (a) re-unwrap with a wider margin, (b) DIFFUSE colour only
    # (glass/transmissive), (c) a flat 8x8 of the base colour
    tp = os.path.join(TEXDIR, f"obj{oi:03d}.png")
    route, cov, mean = "COMBINED", 0.0, 0.0
    try:
        cov, mean = bake_to(o, img, tp, "COMBINED")
        failed = False
    except Exception as e:
        failed = True
        log(f"  obj{oi:03d} COMBINED failed: {e}")
    if failed or cov < COV_GATE or mean < DARK_GATE:
        try:
            unwrap(o, 0.05)
            cov, mean = bake_to(o, img, tp, "COMBINED")
            route = "COMBINED+reunwrap"
        except Exception as e:
            log(f"  obj{oi:03d} re-unwrap failed: {e}")
    if cov < COV_GATE or mean < DARK_GATE:
        try:
            cov, mean = bake_to(o, img, tp, "DIFFUSE")
            route = "DIFFUSE"
        except Exception as e:
            log(f"  obj{oi:03d} DIFFUSE failed: {e}")
    if cov < COV_GATE or mean < DARK_GATE:
        rgb = flat_basecolor(o)
        flat = bpy.data.images.new(f"f{oi}", 8, 8)
        flat.pixels = [v for _ in range(8 * 8) for v in (rgb[0], rgb[1], rgb[2], 1.0)]
        flat.filepath_raw = tp
        flat.file_format = "PNG"
        flat.save()
        bpy.data.images.remove(flat)
        cov, mean = tex_stats(tp)
        route = "FLAT"

    if cv2 is not None and route.startswith("COMBINED"):
        try:
            cv2.imwrite(tp, cv2.fastNlMeansDenoisingColored(cv2.imread(tp), None, 7, 7, 7, 21))
        except Exception:
            pass

    rec = dict(idx=oi, name=o.name, faces=len(o.data.polygons), specular=round(specular_score(o), 3), src_uv=src_uv)
    rec.update(uvs_pinned=n_pin_img + n_pin_coord, pin_img=n_pin_img, pin_coord=n_pin_coord)
    rec.update(n_tex_image_nodes=n_texnodes, area_m2=round(area, 3), res=res, route=route, coverage=round(cov, 4))
    rec.update(mean=round(mean, 1), secs=round(time.time() - t_obj, 1), tex=f"tex/obj{oi:03d}.png")
    hf = None  # high-frequency detail of the atlas (Laplacian std): a flat texture on a big surface is suspect
    if cv2 is not None:
        try:
            hf = float(
                cv2.Laplacian(cv2.cvtColor(cv2.imread(tp), cv2.COLOR_BGR2GRAY).astype("float32"), cv2.CV_32F).std()
            )
        except Exception:
            pass
    rec["hf"] = None if hf is None else round(hf, 3)
    records.append(rec)
    baked.append((o, oi, tp))
    log(f"  obj{oi:03d} '{o.name[:38]}' res={res} area={area:.2f}m2 route={route} cov={cov:.3f} mean={mean:.1f}")
bake_secs = time.time() - T0
log(f"=== PHASE A done: {len(baked)}/{len(keep)} objects in {bake_secs:.1f}s ===")

log("=== PHASE B: swap to unlit emission + export ===")
for o, oi, tp in baked:
    disp = bpy.data.images.load(tp)
    disp.colorspace_settings.name = "sRGB"
    um = bpy.data.materials.new(f"u{oi:03d}")
    um.use_nodes = True
    nt = um.node_tree
    nt.nodes.clear()
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = disp
    uvn = nt.nodes.new("ShaderNodeUVMap")
    uvn.uv_map = "uv"
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    on = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(uvn.outputs["UV"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    ec = bsdf.inputs["Emission Color"] if "Emission Color" in bsdf.inputs else bsdf.inputs["Emission"]
    nt.links.new(tex.outputs["Color"], ec)
    if "Emission Strength" in bsdf.inputs:
        bsdf.inputs["Emission Strength"].default_value = 1.0
    nt.links.new(bsdf.outputs["BSDF"], on.inputs["Surface"])
    o.data.materials.clear()
    o.data.materials.append(um)
    for p in o.data.polygons:
        p.material_index = 0

# Deleting an un-baked parent (e.g. a window frame) moves its baked children (the panes) to the origin;
# re-apply their saved world transforms unparented.
saved_world = {o: o.matrix_world.copy() for o, _, _ in baked}
bk = set(saved_world)
for o in list(bpy.data.objects):
    if o.type == "MESH" and o not in bk:
        try:
            bpy.data.objects.remove(o, do_unlink=True)
        except Exception:
            pass
bpy.context.evaluated_depsgraph_get()
for o, mw in saved_world.items():
    try:
        if o.parent is not None:
            o.parent = None
        o.matrix_world = mw
    except Exception:
        pass
bpy.context.view_layer.update()
log(f"restored world transforms on {len(saved_world)} baked objects")

n_fixed = 0
for o in list(bpy.data.objects):
    if o.type != "MESH" or not o.data.polygons:
        continue
    try:
        bpy.ops.object.select_all(action="DESELECT")
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode="OBJECT")
        n_fixed += 1
    except Exception as e:
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass
        log(f"normals_make_consistent failed on {o.name}: {e}")
bpy.context.view_layer.update()
log(f"recalculated consistent outward normals on {n_fixed} objects")

try:
    opts = dict(export_materials=True, export_uv=True, path_mode="COPY", up_axis="Z", forward_axis="Y")
    bpy.ops.wm.obj_export(filepath=os.path.join(OUT, "scan.obj"), **opts)
except Exception as e:
    log("obj export err", e)

empties = [r for r in records if r["coverage"] < COV_GATE or r["mean"] < DARK_GATE]
manifest = dict(version="1.0", samples=SAMP, maxres=MAXRES, n_keep=len(keep), n_baked=len(baked))
manifest.update(bake_secs=round(bake_secs, 1), total_secs=round(time.time() - T0, 1), empty_textures=empties)
tot_a = sum(r.get("area_m2", 0.0) for r in records) or 1.0
spec = [r for r in records if r.get("specular", 0.0) >= 0.3]
manifest.update(objects=records, specular_area_m2=round(sum(r.get("area_m2", 0.0) for r in spec), 3))
manifest["specular_area_fraction"] = round(sum(r.get("area_m2", 0.0) for r in spec) / tot_a, 4)
spec_objs = [{"name": r["name"][:48], "specular": r["specular"], "area_m2": r.get("area_m2")} for r in spec]
manifest["specular_objects"] = sorted(spec_objs, key=lambda r: -(r["area_m2"] or 0))[:12]
with open(os.path.join(OUT, "bake_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
log(f"DONE in {time.time() - T0:.1f}s -> {OUT} ({len(empties)} empty textures)")
