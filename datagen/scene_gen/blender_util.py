#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""blender_util.py -- scene setup shared by the Blender stages (prep_and_gt.py, bake.py)."""

import bpy


def show_all_meshes():
    """Un-exclude every collection and un-hide every mesh; collection.hide_render is left as authored."""

    def enable(lc):
        try:
            lc.exclude = False
            lc.hide_viewport = False
        except Exception:
            pass
        for c in lc.children:
            enable(c)

    enable(bpy.context.view_layer.layer_collection)
    for o in bpy.data.objects:
        if o.type == "MESH":  # meshes only; native lights stay as authored
            o.hide_render = False
            try:
                o.hide_set(False)
            except Exception:
                pass
            o.hide_viewport = False


def cycles_gpu(sc, log):
    """Render with Cycles on every GPU (OptiX, else CUDA, else Metal); CPU if none."""
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for dt in ("OPTIX", "CUDA", "METAL"):
            try:
                prefs.compute_device_type = dt
                break
            except Exception:
                pass
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True
        sc.cycles.device = "GPU"
    except Exception as e:
        log(f"GPU skip: {e}")
