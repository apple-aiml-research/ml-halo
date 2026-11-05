#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""infinigen_style.py -- scene-style wrapper around Infinigen's generate_indoors.

Patches Infinigen in-process before running generate_indoors: sofa / rug constraints (NO_SOFA,
RUG_TARGET), smaller rugs, lamp brightness (LIGHT_GAIN), matte wood, the floor family (FLOOR_STYLE),
an even rug palette, and for PURE_COLOR=1 scenes saturated plaster walls with no accent wall.
"""

import os
import runpy
from collections import defaultdict

LIGHT_GAIN = float(os.environ.get("LIGHT_GAIN") or "2.0")


def floor_palette(style):
    """The floor material palette of one floor style."""
    from infinigen.assets.materials import ceramic, fabric, wood

    palettes = {
        "wood": [(wood.CompositeWoodTile, 1.0), (wood.CrossedWoodTile, 1.0), (wood.HardwoodFloor, 2.0)]
        + [(wood.HexagonWoodTile, 1.0), (wood.SquareWoodTile, 1.0), (wood.StaggeredWoodTile, 1.0)],
        "mat": [(fabric.Rug, 3.0), (fabric.CoarseKnitFabric, 1.0), (fabric.FineKnitFabric, 1.0)],
        "stone": [(ceramic.Marble, 2.0), (ceramic.MarbleRegular, 1.0), (ceramic.Concrete, 1.0)],
        "tile": [(ceramic.Tile, 1.0)],
    }
    if style not in palettes:
        raise SystemExit(f"[style] unknown FLOOR_STYLE={style!r}; expected one of {sorted(palettes)}")
    return palettes[style]


def apply_matte_wood():
    """Matte wood (floors and furniture): replace the roughness / clear-coat draws inside wood.py."""
    from numpy.random import uniform

    from infinigen.assets.materials.wood import wood as W

    rules = {
        (0.0, 0.4): lambda: uniform(0.55, 0.75),  # roughness base
        (0.0, 0.8): lambda: uniform(0.00, 0.20),  # roughness ramp top
        (0, 1.4): 0.0,  # clear coat
    }
    orig = W.uniform

    def patched(*a, **k):
        if not k and a in rules:
            r = rules[a]
            return r() if callable(r) else r
        return orig(*a, **k)

    W.uniform = patched


def patch_home_constraints():
    """No sofa (NO_SOFA != 0) and a rug-count target (RUG_TARGET) in home_furniture_constraints."""
    from infinigen.assets.objects import elements, seating
    from infinigen.core.constraints import constraint_language as cl
    from infinigen.core.tags import Semantics
    from infinigen_examples.constraints import home
    from infinigen_examples.constraints import util as cu

    no_sofa = (os.environ.get("NO_SOFA", "1") or "1").strip() != "0"
    rt = (os.environ.get("RUG_TARGET") or "").strip()
    rug_target = int(rt) if rt.isdigit() else None
    orig = home.home_furniture_constraints

    def patched(*a, **k):
        prob = orig(*a, **k)
        cons, st = dict(prob.constraints), dict(prob.score_terms)
        rooms = cl.scene()[{Semantics.Room, -Semantics.Object}]
        obj = cl.scene()[{Semantics.Object, -Semantics.Room}]
        if no_sofa:
            sofas = obj[Semantics.Furniture].related_to(rooms, cu.on_floor)[seating.SofaFactory]
            st.pop("sofa", None)  # stop maximising sofa volume
            cons["no_sofa_v6"] = sofas.count().in_range(0, 0)
        if rug_target is not None:
            rugs = obj[elements.RugFactory].related_to(rooms, cu.on_floor)
            cons["v6_rug_cap"] = rugs.count().in_range(0, 2)
            st["v6_rug_target"] = rugs.count().sub(rug_target).abs().minimize(weight=8)
        return cl.Problem(constraints=cons, score_terms=st)

    home.home_furniture_constraints = patched
    print(f"[style] home constraints: no_sofa={no_sofa} rug_target={rug_target}", flush=True)


def after_init(cls, fn):
    """Wrap cls.__init__ so fn(self) runs after it (best effort: a failing fn leaves the stock object)."""
    orig_init = cls.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        try:
            fn(self)
        except Exception:
            pass

    cls.__init__ = init


def apply_small_rugs():
    """Shrink rugs (~0.55x) so 2-3 fit a small room."""
    from infinigen.assets.objects.elements import rug as RUG

    def shrink(self):
        self.width = float(self.width) * 0.55
        self.length = float(self.length) * 0.55

    after_init(RUG.RugFactory, shrink)


def apply_brightness(gain):
    """Scale the drawn Wattage of every PointLampFactory bulb (ceiling + floor + desk lamps)."""
    from infinigen.assets.lighting import indoor_lights as IL

    def brighten(self):
        self.params["Wattage"] = float(self.params["Wattage"]) * gain

    after_init(IL.PointLampFactory, brighten)
    print(f"[style] brightness: PointLampFactory Wattage x{gain}", flush=True)


def apply_floor_style():
    """Use the FLOOR_STYLE palette (wood/tile/stone/mat) for every room's floor."""
    from infinigen.assets.composition import material_assignments as MA
    from infinigen.core.constraints.example_solver.room import decorate

    style = (os.environ.get("FLOOR_STYLE") or "wood").strip().lower()
    pal = floor_palette(style)
    MA.floor = pal
    MA.office_floor = pal
    decorate.room_floor_fns = defaultdict(lambda: pal)
    print(f"[style] floor style={style}", flush=True)


def apply_rug_variety():
    """Equal weights over the rug palette (stock: ~80% of rugs are fabric.Rug or ArtRug)."""
    from infinigen.assets.composition import material_assignments as MA

    pal = list(getattr(MA, "rug_fabric", []) or [])
    if pal:
        MA.rug_fabric = [(cls, 1.0) for cls, _ in pal]


def apply_no_accent_walls():
    """Pure-colour scenes: no 'alternative' accent wall."""
    from infinigen.core.constraints.example_solver.room import decorate

    try:
        alt = decorate.room_wall_alternative_fns
        for k in list(alt.keys()):
            alt[k] = ("weighted_choice", (1, "none"))
    except Exception as e:
        print(f"[style] WARN could not disable accent walls: {type(e).__name__}: {e}", flush=True)


def apply_saturated_walls():
    """Pure-colour scenes: coloured plaster walls and ceilings, saturation lifted to ~0.80-0.92."""
    from infinigen.assets.composition import material_assignments as MA
    from infinigen.assets.materials.ceramic import plaster as P
    from infinigen.core.constraints.example_solver.room import decorate

    MA.wall = list(MA.wall_plaster)
    MA.ceiling = list(MA.wall_plaster)
    decorate.room_wall_fns = defaultdict(lambda: MA.wall, dict(decorate.room_wall_fns))
    decorate.room_ceiling_fns = defaultdict(lambda: MA.ceiling, dict(decorate.room_ceiling_fns))

    orig_apply, orig_gen = P.Plaster.apply, P.Plaster.generate

    def sat_apply(self, obj, selection=None, plaster_colored=None, **kw):
        return orig_apply(self, obj, selection=selection, plaster_colored=True, **kw)

    def sat_generate(self, plaster_colored=None, **kw):
        return orig_gen(self, plaster_colored=True, **kw)

    P.Plaster.apply = sat_apply
    P.Plaster.generate = sat_generate
    P.Plaster.__call__ = sat_generate
    orig_hsv = P.hsv2rgba

    def vivid_hsv(hsv, *args):
        h, s, v = (hsv, args[0], args[1]) if args else (hsv[0], hsv[1], hsv[2])
        if s is not None and s < 0.6:
            s = min(0.92, 0.60 + s * 0.65)  # 0.3->0.795, 0.5->0.925
        return orig_hsv(h, s, v)

    P.hsv2rgba = vivid_hsv


def apply_all():
    patch_home_constraints()
    apply_small_rugs()
    apply_brightness(LIGHT_GAIN)
    apply_matte_wood()
    apply_floor_style()
    apply_rug_variety()
    if (os.environ.get("PURE_COLOR", "") or "").strip().lower() in ("1", "true", "yes"):
        apply_saturated_walls()
        apply_no_accent_walls()
        print("[style] PURE_COLOR=1: saturated feature walls", flush=True)


if __name__ == "__main__":
    apply_all()
    runpy.run_module("infinigen_examples.generate_indoors", run_name="__main__")
