#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Resolve asset specs (checkpoints, configs, policies, scenes) to local paths.

Asset specs are paths: ``~`` and ``$VARS`` are expanded; absolute paths are used as-is, relative
paths are taken relative to the repository root.

A ``<...>`` spec is a placeholder in a mission YAML and is an error until it is replaced.
"""
from __future__ import annotations

import os
from pathlib import Path

# system/halo_system/lib/assets.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_asset(spec, required: str | None = None) -> str:
    """Resolve an asset spec to a local path. An empty / None spec is returned unchanged, or,
    when ``required`` names the asset (e.g. "generator checkpoint"), is an error."""
    if not spec:
        if required:
            raise SystemExit(f"no {required} given: pass a path")
        return spec
    spec = str(spec)
    if spec.startswith("<") and spec.endswith(">"):
        raise SystemExit(
            f"'{spec}' is a placeholder: replace it in the mission YAML with your own file (or pass "
            f"it on the command line). See 'Models' in system/README.md.")
    p = Path(os.path.expandvars(os.path.expanduser(spec)))
    if p.is_absolute():
        return str(p)
    in_repo = REPO_ROOT / p
    if in_repo.exists():
        return str(in_repo)
    if p.exists():
        return str(p.resolve())
    return str(in_repo)


def load_scene_model(scene_xml: str):
    """MuJoCo model of a scene XML for the Python side (viewer).

    Room scans can contain flat, zero-volume meshes. The C++ sim tracker's MuJoCo accepts them as they
    are and rejects the `inertia="shell"` attribute; newer Python MuJoCo refuses them without it. So
    when the plain load fails on a mesh volume, the scene is reloaded with shell inertia on every mesh.
    """
    import mujoco
    try:
        return mujoco.MjModel.from_xml_path(scene_xml)
    except ValueError as exc:
        if "mesh volume is too small" not in str(exc):
            raise
    spec = mujoco.MjSpec.from_file(scene_xml)
    for mesh in spec.meshes:
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    return spec.compile()
