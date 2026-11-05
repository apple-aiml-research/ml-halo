#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""make_manifest.py -- build the single-room scene list (spec/scenes_single.json).

Every scene is a pure function of its index (room type, size, windows, furniture density, sofa, floor
family, rugs, light gain, plain vs decorated walls); scenes are packed into shards longest-first by a
predicted pod time.
"""

import argparse
import hashlib
import json
import math
import os
import random

CAMPAIGN = "scenegen-v7-diverse"  # the RNG salt of every scene draw: renaming it re-rolls every scene
SEED_BASE = 700_000
BALANCE_TOLERANCE = 1.25  # max shard cost must be within this factor of the mean

ROOMS = [  # name, floor-plan key, probability, W range, D range (m)
    ("LivingRoom", "living-room", 0.45, (6.5, 9.5), (4.5, 6.5)),
    ("Office", "office", 0.30, (5.5, 8.0), (3.5, 5.5)),
    ("Bedroom", "bedroom", 0.25, (5.5, 8.0), (4.0, 6.0)),
]
COMMON = [
    "compose_indoors.lights_off_enabled=False",  # never ship an unlit scene
    "compose_indoors.lights_off_chance=0.0",
    "compose_indoors.skirting_floor_chance=1.0",  # finished-interior baseboards
    "compose_indoors.skirting_ceiling_chance=1.0",
]


def density_bindings(large, medium, small, weight):
    return [
        "compose_indoors.floating_objs_enabled=False",
        f"compose_indoors.solve_steps_large={large}",
        f"compose_indoors.solve_steps_medium={medium}",
        f"compose_indoors.solve_steps_small={small}",
        f"solve_objects.addition_weight_scalar={weight}",
    ]


DENSITY = {  # name: (probability, gin bindings)
    "sparse": (0.30, density_bindings(60, 30, 8, 1.0)),
    "medium": (0.40, density_bindings(100, 55, 14, 1.5)),
    "dense": (0.30, density_bindings(120, 70, 16, 1.8)),
}
MAX_ASPECT = 2.0
WINDOW_FRACTION = 0.45  # window length as a fraction of its wall
WINDOW_JITTER = 0.12
WINDOW_MIN_LEN = 1.2
WINDOW_MAX_LEN = 4.0
PANORAMIC_PROB = 0.35
LIGHT_GAIN_RANGE = (0.9, 2.2)


def rng_for(idx):
    """Deterministic per-scene RNG, independent of iteration order or Python's hash seed."""
    h = hashlib.sha256(f"{CAMPAIGN}::{idx}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def weighted(rng, items, weights):
    r = rng.random() * sum(weights)
    acc = 0.0
    for item, w in zip(items, weights):
        acc += w
        if r < acc:
            return item
    return items[-1]


def windows(rng, W, D):
    """1-3 windows on distinct walls of the WxD box (origin at 0,0)."""
    perimeter = 2 * (W + D)
    n = 1 + int(perimeter > 18.0) + int(perimeter > 26.0 and rng.random() < 0.5)
    walls = ["S", "E", "N", "W"]
    rng.shuffle(walls)
    out = []
    for wall in walls[:n]:
        span = W if wall in ("S", "N") else D
        length = span * (WINDOW_FRACTION + rng.uniform(-WINDOW_JITTER, WINDOW_JITTER))
        length = max(WINDOW_MIN_LEN, min(WINDOW_MAX_LEN, length))
        length = min(length, max(0.6, span - 0.6))
        c = rng.uniform(0.30, 0.70) * span
        c = max(length / 2 + 0.3, min(span - length / 2 - 0.3, c))
        a, b = c - length / 2, c + length / 2
        pa, pb = {
            "S": ((round(a, 3), 0.0), (round(b, 3), 0.0)),
            "N": ((round(a, 3), round(D, 3)), (round(b, 3), round(D, 3))),
            "W": ((0.0, round(a, 3)), (0.0, round(b, 3))),
            "E": ((round(W, 3), round(a, 3)), (round(W, 3), round(b, 3))),
        }[wall]
        out.append({"wall": wall, "a": pa, "b": pb, "length": round(length, 3), "panoramic": False})
    if out and rng.random() < PANORAMIC_PROB:
        out[rng.randrange(len(out))]["panoramic"] = True
    return out


def scene_spec(idx, seed_base=SEED_BASE):
    """The full spec of scene `idx`. Pure function of (idx, seed_base)."""
    rng = rng_for(idx)
    room_name = weighted(rng, [r[0] for r in ROOMS], [r[2] for r in ROOMS])
    _, room_key, _, w_range, d_range = next(r for r in ROOMS if r[0] == room_name)
    rng.random()  # the tier draw (a single tier, B0, is left); keeps the RNG stream of scenes_single.json
    density = weighted(rng, list(DENSITY), [p for p, _ in DENSITY.values()])
    floor_style = weighted(rng, ["wood", "tile", "stone", "mat"], [0.45, 0.20, 0.20, 0.15])
    rug_target = weighted(rng, [0, 1, 2], [0.20, 0.45, 0.35])
    light_gain = round(rng.uniform(*LIGHT_GAIN_RANGE), 2)
    W = rng.uniform(*w_range)
    D = rng.uniform(*d_range)
    if D > W:
        W, D = D, W
    D = max(D, W / MAX_ASPECT)
    W, D = round(W, 1), round(D, 1)
    sofa = bool(hashlib.sha256(f"{CAMPAIGN}::sofa::{idx}".encode()).digest()[0] & 1)
    spec = dict(idx=idx, name=f"scene_{idx:06d}", seed=seed_base + idx, room=room_name, room_key=room_key, tier="B0")
    spec.update(W=W, D=D, area=round(W * D, 2), aspect=round(W / D, 3), gin=["singleroom.gin"])
    spec.update(pextra=COMMON + DENSITY[density][1], windows=windows(rng, W, D), floor_style=floor_style)
    spec.update(pure_color=idx % 2 == 0, density=density, has_sofa=sofa, rug_target=rug_target, light_gain=light_gain)
    return spec


def floorplan(spec):
    """Infinigen predefined floor plan: one WxD box room plus its windows (shapely expressions)."""
    W, D = spec["W"], spec["D"]
    fp = {"rooms": {f"{spec['room_key']}_0/0": {"shape": f"shapely.box(0,0,{W},{D})"}}, "windows": {}}
    for i, w in enumerate(spec["windows"]):
        (x0, y0), (x1, y1) = w["a"], w["b"]
        entry = {"shape": f"shapely.LineString([({x0},{y0}),({x1},{y1})])"}
        if w["panoramic"]:
            entry["is_panoramic"] = 1
        fp["windows"]["window" if i == 0 else f"window.{i:03d}"] = entry
    import shapely  # noqa: F401  fail here, not 40 min into a pod: every shape must eval

    for objs in fp.values():
        for info in objs.values():
            eval(info["shape"])
    return fp


def anchor_for(spec):
    """Room-centre anchor with a keep-radius covering the whole room."""
    W, D = spec["W"], spec["D"]
    cx, cy = W / 2.0, D / 2.0
    pos = [round(cx, 3), round(max(0.6, cy - min(2.5, D / 2 - 0.6)), 3), 1.45]
    look = [round(cx, 3), round(cy + 0.5, 3), 0.70]
    keep_radius = round(0.5 * math.hypot(W, D) + 1.5, 3)
    return dict(name=spec["name"], pos=pos, look=look, lens=35.0, sensor=36.0, keep_radius=keep_radius)


def gen_minutes(area):
    """Predicted Infinigen time (min): sub-linear in area, x1.5 for the cloud CPUs."""
    return 20.0 * (area / 40.0) ** 0.6 * 1.5


def gpu_minutes(area, n_views, gt_frames, n_projected=3):
    """Predicted minutes of the GPU stages: prep, GT frames, bake, projection, usdc, per-view costs."""
    bake = 41.0 * (area / 20.0) ** 0.45 * 1.0
    return (
        3.0 + gt_frames * 4.75 + bake + n_projected * 1.0 + 0.5
        + n_views * (4.4 / 60.0) + 2.0 + n_views * (0.5 / 60.0) + 5.0
    )  # fmt: skip


def build(n, n_shards, n_views, gt_frames, seed_base=SEED_BASE):
    """Scene entries, then longest-first packing onto the currently-lightest shard."""
    entries = []
    for idx in range(n):
        spec = scene_spec(idx, seed_base=seed_base)
        g, p = gen_minutes(spec["area"]), gpu_minutes(spec["area"], n_views, gt_frames)
        entries.append(
            {**spec, "floorplan": floorplan(spec), "anchor": anchor_for(spec), "pred_gen_min": round(g, 2),
             "pred_gpu_min": round(p, 2), "pred_min": round(max(p, g / 3) + 0.0, 2)}
        )  # fmt: skip
    load = [0.0] * n_shards
    members = [[] for _ in range(n_shards)]
    for e in sorted(entries, key=lambda e: (-e["pred_min"], e["idx"])):
        s = min(range(n_shards), key=lambda i: (load[i], i))
        load[s] += e["pred_min"]
        members[s].append(e["idx"])
        e["shard"] = s
    return entries, load, members


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1200, help="number of scenes")
    ap.add_argument("--shards", type=int, default=300, help="number of shards (one job each)")
    ap.add_argument("--views", type=int, default=64, help="viewpoints per scene (cost model input)")
    ap.add_argument("--gt-frames", type=int, default=4, help="ground-truth frames per scene (cost model input)")
    ap.add_argument("--out", default=None, help="write manifest JSON here")
    ap.add_argument("--seed-base", type=int, default=SEED_BASE, help="infinigen seed = seed_base + idx")
    a = ap.parse_args()

    entries, load, members = build(a.n, a.shards, a.views, a.gt_frames, seed_base=a.seed_base)
    mean, mx = sum(load) / len(load), max(load)
    print(f"{a.n} scenes in {a.shards} shards; shard cost max/mean {mx / mean:.3f}, longest {mx / 60:.1f} h")
    if mx / mean >= BALANCE_TOLERANCE:
        raise SystemExit(f"FAIL: shard imbalance max/mean={mx / mean:.3f} >= {BALANCE_TOLERANCE}; raise --shards")
    doc = dict(campaign=CAMPAIGN, version="1.0", n_scenes=a.n, n_shards=a.shards, n_views=a.views)
    doc.update(gt_frames=a.gt_frames, seed_base=a.seed_base, shard_pred_min=[round(x, 1) for x in load])
    doc.update(shard_members=members, scenes=entries)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(doc, f, separators=(",", ":"))
        print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
