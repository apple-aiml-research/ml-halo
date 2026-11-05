#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""MuJoCo XML helpers for the render: inject an EpisodeSpec's props into a scene."""

from __future__ import annotations

import os

from render.assets import extract_member, resolve_asset, usdz_mesh
from motion_gen.episode_spec import AssetPlacement, EpisodeSpec

_OBJ_CACHE_DIR = "/tmp/mj_obj_cache"


def set_obj_cache_dir(path: str) -> None:
    """Directory the prop OBJ/MTL/PNG files are extracted to."""
    global _OBJ_CACHE_DIR
    _OBJ_CACHE_DIR = path
    os.makedirs(path, exist_ok=True)


def _usdz_to_obj(usdz_path: str, out_dir: str) -> str | None:
    """``<out_dir>/<stem>.obj`` (+ .mtl and diffuse .png) of a USDZ prop; reused if already there."""
    stem = os.path.splitext(os.path.basename(usdz_path))[0]
    obj_path = os.path.join(out_dir, f"{stem}.obj")
    if os.path.isfile(obj_path):
        return obj_path
    mesh = usdz_mesh(usdz_path)
    if mesh is None:
        return None
    diffuse = mesh["tex"].get("diffuse")
    with open(os.path.join(out_dir, f"{stem}.mtl"), "w") as f:
        f.write(f"newmtl mat_{stem}\n")
        f.write("Ka 1 1 1\nKd 1 1 1\nKs 0 0 0\nillum 1\n")
        if diffuse is not None:
            extract_member(usdz_path, diffuse, os.path.join(out_dir, stem + ".png"))
            f.write(f"map_Kd {stem}.png\n")
    with open(obj_path, "w") as f:
        f.write(f"mtllib {stem}.mtl\nusemtl mat_{stem}\n")
        for v in mesh["verts"]:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        faces = mesh["faces"] + 1
        if mesh["uvs"] is not None:
            for uv in mesh["uvs"].reshape(-1, 2):
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
            for fi, (v0, v1, v2) in enumerate(faces):
                f.write(f"f {v0}/{fi * 3 + 1} {v1}/{fi * 3 + 2} {v2}/{fi * 3 + 3}\n")
        else:
            for v0, v1, v2 in faces:
                f.write(f"f {v0} {v1} {v2}\n")
    return obj_path


def _placement_to_xml(p: AssetPlacement, idx: int) -> tuple[str, str]:
    """``(asset_block, body_block)`` XML for one placement: textured, emissive (unlit room) mesh."""
    asset_path = resolve_asset(p.asset_path)
    obj = _usdz_to_obj(asset_path, _OBJ_CACHE_DIR)
    if obj is None:
        raise RuntimeError(f"[episode_objects] could not read the mesh of placement {idx} ({asset_path!r})")
    mesh_name, tex_name, mat_name = (f"mesh_{idx:02d}", f"tex_{idx:02d}", f"mat_{idx:02d}")
    tex = os.path.splitext(obj)[0] + ".png"
    asset = f'    <mesh name="{mesh_name}" file="{obj}"/>\n'
    if os.path.isfile(tex):
        asset += f'    <texture name="{tex_name}" type="2d" file="{tex}"/>\n'
        asset += (
            f'    <material name="{mat_name}" texture="{tex_name}" specular="0.05" shininess="0.05" emission="1.0"/>\n'
        )
        geom = f'<geom type="mesh" mesh="{mesh_name}" material="{mat_name}"/>'
    else:
        geom = f'<geom type="mesh" mesh="{mesh_name}" rgba="0.850 0.850 0.850 1.000"/>'
    px, py, pz = p.pos_xyz
    qw, qx, qy, qz = p.quat_wxyz
    body = (
        f'    <body name="prop_{idx:02d}" pos="{px:.4f} {py:.4f} {pz:.4f}" '
        f'quat="{qw:.4f} {qx:.4f} {qy:.4f} {qz:.4f}">\n'
        f"      {geom}\n"
        f"    </body>\n"
    )
    return asset, body


def add_objects_to_xml(scene_xml: str, spec: EpisodeSpec) -> str:
    """Insert the spec's placements (one per distinct asset) into ``<asset>`` and before ``</worldbody>``."""
    assets, bodies, seen = [], [], set()
    for i, p in enumerate(spec.placements):
        if p.asset_path in seen:
            continue
        a, b = _placement_to_xml(p, idx=i + 1)
        assets.append(a)
        bodies.append(b)
        seen.add(p.asset_path)
    if assets:
        scene_xml = scene_xml.replace("</asset>", "".join(assets) + "  </asset>")
    return scene_xml.replace("</worldbody>", "".join(bodies) + "  </worldbody>")
