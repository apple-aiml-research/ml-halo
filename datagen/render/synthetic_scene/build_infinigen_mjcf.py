#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Assemble a robot+camera-ready MuJoCo scene XML for an infinigen baked room."""

from __future__ import annotations

import logging
import os
import re

from render.synthetic_scene.obj_to_mjcf import geometry_assets_and_geoms, split_obj

log = logging.getLogger("build_infinigen_mjcf")

_G1_ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "robots", "g1"))
_THOR_ROBOT_MJCF = os.path.join(_G1_ASSETS, "mjcf", "g1_29dof_v2_thor.xml")
_THOR_MESHES = os.path.join(_G1_ASSETS, "meshes")

_EGO_PLACEHOLDER = '<camera name="ego_camera" pos="0.054040 0.036023 0.433784" xyaxes="1 0 0 0 0 1" fovy="43.17"/>'
# oak1 as a 140-deg-HFOV pinhole (vfov 119.6 at 16:10), the source of the cv2.fisheye warp; the MJCF's
# focalpixel encodes the (narrower) fisheye K.
_OAK_RE = re.compile(
    r'<camera\s+name="oak1"\s+'
    r'(pos="[^"]*"(?:\s+(?:euler|xyaxes|quat)="[^"]*")?)'
    r"(?:[^/]|\s)*?/>",
    re.DOTALL,
)


def make_robot_with_cameras(out_path: str, base_mjcf: str = _THOR_ROBOT_MJCF, meshes_dir: str = _THOR_MESHES) -> str:
    """Copy of the THOR G1 MJCF without its floor/skybox/light, absolute mesh paths, torso ego_camera, pinhole oak1."""
    xml = open(base_mjcf).read()
    xml = re.sub(r'\s*<geom\s+name="floor"[^>]*?/>', "", xml)
    xml = re.sub(r'\s*<pair\b[^>]*\bgeom[12]="floor"[^>]*?/>', "", xml)
    xml = re.sub(r'\s*<texture\s+type="skybox"[^>]*?/>', "", xml)
    xml = re.sub(r'\s*<texture\b[^>]*\bname="groundplane"[^>]*?/>', "", xml)
    xml = re.sub(r'\s*<material\s+name="groundplane"[^>]*?/>', "", xml)
    xml = re.sub(r'\s*<light\b[^>]*\bdirectional="true"[^>]*?/>', "", xml)

    def _abs_mesh(m: re.Match) -> str:
        fn = m.group(1)
        return m.group(0) if fn.startswith("/") else f'file="{os.path.join(meshes_dir, fn)}"'

    xml = re.sub(r'file="([^"]+\.(?:STL|stl|obj|OBJ))"', _abs_mesh, xml)
    xml = re.sub(r'(<compiler[^>]*?)\s*meshdir="[^"]*"', r"\1", xml)
    anchor = '<body name="torso_link">'
    xml = xml.replace(anchor, anchor + "\n            " + _EGO_PLACEHOLDER, 1)
    xml = _OAK_RE.sub(lambda m: f'<camera name="oak1" {m.group(1)} fovy="119.6"/>', xml)
    with open(out_path, "w") as f:
        f.write(xml)
    return out_path


def build_scene_with_robot(scene_dir: str, workdir: str, *, offwidth: int = 1600, offheight: int = 1000) -> str:
    """Write ``<workdir>/scene_with_robot.xml`` (unlit baked room + THOR G1 + cameras); returns its path."""
    scene_dir = os.path.expanduser(scene_dir)
    obj_path = os.path.join(scene_dir, "asset", "scan.obj")
    if not os.path.isfile(obj_path):
        raise FileNotFoundError(obj_path)
    os.makedirs(workdir, exist_ok=True)
    parts = split_obj(obj_path, os.path.join(workdir, "_mjparts"))
    assets, geoms = geometry_assets_and_geoms(parts)
    robot_mjcf = make_robot_with_cameras(os.path.join(workdir, "g1_29dof_ego.xml"))
    scene_xml = f"""<mujoco model="infinigen_scene_with_robot">
  <compiler meshdir="/" texturedir="/" angle="radian" autolimits="true"/>
  <include file="{os.path.basename(robot_mjcf)}"/>
  <visual>
    <headlight ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/>
    <global offwidth="{offwidth}" offheight="{offheight}"/>
    <quality offsamples="4"/>
    <map znear="0.005" zfar="5.0"/>
  </visual>
  <asset>
{assets}
  </asset>
  <worldbody>
{geoms}
  </worldbody>
</mujoco>
"""
    out = os.path.join(workdir, "scene_with_robot.xml")
    with open(out, "w") as f:
        f.write(scene_xml)
    log.info("wrote %s (%d geometry parts + G1 robot + torso cameras)", out, len(parts))
    return out
