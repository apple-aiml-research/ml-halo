#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""make_dualroom_manifest.py -- build a dual-room scene list: two rooms joined by a door-less opening."""

import argparse, hashlib, json, math, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_manifest import COMMON, density_bindings

SEM = {"living-room": "LivingRoom", "office": "Office", "bedroom": "Bedroom"}
TYPES = ["living-room", "office", "bedroom"]
V6C = COMMON + ["RoomConstants.global_params.wall_thickness=0.04"]  # 4 cm per wall, 8 cm at the shared one
DENS = {
    "sparse": density_bindings(60, 30, 8, 1.0),
    "medium": density_bindings(100, 55, 14, 1.5),
    "dense": density_bindings(140, 80, 20, 1.8),
}
# layout id: (room A polygon, room B polygon, door-less opening segment, window (x0, x1, y))
TEMPLATES = {
    1: ([(0, 3.5), (6, 3.5), (6, 7), (0, 7)], [(0.3, 0), (2.6, 0), (2.8, 3.5), (0, 3.5)], [(0.8, 3.5), (2.0, 3.5)], (1.5, 4.0, 7)),
    2: ([(0, 3.5), (5, 3.5), (5, 7), (0, 7)], [(1, 0), (3.2, 0), (2.9, 3.5), (0.6, 3.5)], [(1.2, 3.5), (2.4, 3.5)], (1.5, 3.5, 7)),
    3: ([(0, 3), (7, 3), (7, 6.5), (0, 6.5)], [(0, 0), (1.6, 0), (1.6, 3), (0, 3)], [(0.2, 3), (1.4, 3)], (2.5, 5.0, 6.5)),
    4: ([(0, 1.2), (5, 1.2), (6.5, -0.3), (5, -1.8), (0, -1.8)], [(0, -3), (5, -3), (5, -1.8), (0, -1.8)],
        [(1, -1.8), (2.2, -1.8)], (1.0, 3.5, 1.2)),
    5: ([(0, 5), (5, 3), (5, 8), (0, 8)], [(0, 0), (5, 0), (5, 3), (0, 5)], [(3.4, 3.64), (4.5, 3.2)], (1.0, 4.0, 8)),
}  # fmt: skip


def polystr(pts):
    return "shapely.Polygon([" + ",".join(f"({round(x, 2)},{round(y, 2)})" for x, y in pts) + "])"


def segstr(p, q):
    return f"shapely.LineString([({round(p[0], 2)},{round(p[1], 2)}),({round(q[0], 2)},{round(q[1], 2)})])"


def build(n, shards, layouts, out, seed_base=640_000):
    """Write the list: layouts cycled, each scaled x0.85-1.25, room types / style drawn from one seeded RNG."""
    rng = random.Random(20260904)
    scenes = []
    for idx in range(n):
        lid = layouts[idx % len(layouts)]
        A, B, op, win = TEMPLATES[lid]
        s = rng.uniform(0.85, 1.25)
        A, B, op = ([(x * s, y * s) for x, y in pts] for pts in (A, B, op))
        win = (win[0] * s, win[1] * s, win[2] * s)
        ta, tb = rng.choice(TYPES), rng.choice(TYPES)
        allx, ally = [p[0] for p in A + B], [p[1] for p in A + B]
        W, D = round(max(allx) - min(allx), 2), round(max(ally) - min(ally), 2)
        cxA, cyA = round(sum(p[0] for p in A) / len(A), 2), round(sum(p[1] for p in A) / len(A), 2)
        dens = rng.choice(["sparse", "medium", "medium", "dense"])
        fp = {
            "rooms": {f"{ta}_0/0": {"shape": polystr(A)}, f"{tb}_0/1": {"shape": polystr(B)}},
            "opens": {"open_0": {"shape": segstr(op[0], op[1])}},
            "windows": {"window": {"shape": segstr((win[0], win[2]), (win[1], win[2]))}},
        }
        fs = rng.choices(["wood", "tile", "stone", "mat"], [0.45, 0.2, 0.2, 0.15])[0]
        rug = rng.choices([0, 1, 2], [0.2, 0.45, 0.35])[0]
        sofa = bool(hashlib.sha256(f"dr::sofa::{idx}".encode()).digest()[0] & 1)
        e = dict(idx=idx, name=f"scene_{idx:06d}", seed=seed_base + idx, room=SEM[ta], room_key=ta, tier="DR", W=W, D=D)
        e.update(
            area=round(W * D, 2), aspect=round(W / max(D, 0.1), 3), gin=["fast_solve.gin"], pextra=V6C + DENS[dens]
        )
        e.update(windows=[], floorplan=fp, floor_style=fs, pure_color=(idx % 2 == 0), has_sofa=sofa, rug_target=rug)
        pos = [cxA, round(max(0.8, cyA - 1.5), 2), 1.45]
        keep_radius = round(0.5 * math.hypot(W, D) + 2.0, 3)
        anchor = dict(name=e["name"], pos=pos, look=[cxA, cyA, 0.7], lens=35.0, sensor=36.0, keep_radius=keep_radius)
        e.update(density=dens, layout=f"L{lid}", anchor=anchor, pred_gen_min=20.0, pred_gpu_min=60.0, pred_min=60.0)
        scenes.append(e)
    members = [[] for _ in range(shards)]
    for e in scenes:
        members[e["idx"] % shards].append(e["idx"])
    doc = dict(campaign="scenegen-v6-dual", version="1.0", n_scenes=n, n_shards=shards, n_views=0, gt_frames=0)
    doc.update(seed_base=seed_base, shard_pred_min=[0] * shards, shard_members=members, scenes=scenes)
    json.dump(doc, open(out, "w"), separators=(",", ":"))
    print("WROTE", out, "n", n, "shards", shards, "layouts", layouts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layouts", default="1,2,3", help="comma list of dual-room layout ids to cycle through")
    ap.add_argument("--n", type=int, default=100, help="number of scenes")
    ap.add_argument("--shards", type=int, default=50, help="number of shards (one job each)")
    ap.add_argument("--seed-base", type=int, default=640_000, help="infinigen seed = seed_base + idx")
    ap.add_argument("--out", required=True, help="write manifest JSON here")
    a = ap.parse_args()
    build(a.n, a.shards, [int(x) for x in a.layouts.split(",")], a.out, seed_base=a.seed_base)
