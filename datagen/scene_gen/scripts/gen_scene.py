#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""gen_scene.py -- generate one Infinigen scene (coarse task -> scene.blend) from a manifest entry."""

import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# manifest field -> style env var read by infinigen_style.py (an env var already set wins)
STYLE_ENV = (
    ("floor_style", "FLOOR_STYLE", str),
    ("pure_color", "PURE_COLOR", lambda v: "1" if v else "0"),
    ("has_sofa", "NO_SOFA", lambda v: "0" if v else "1"),
    ("light_gain", "LIGHT_GAIN", str),
    ("rug_target", "RUG_TARGET", str),
)


def resolve_cwd(infinigen_root, python):
    """A directory containing `infinigen_examples/` (gin's relative includes resolve from there)."""
    if os.path.isdir(os.path.join(infinigen_root, "infinigen_examples")):
        return infinigen_root
    probe = "import infinigen_examples,os;print(os.path.dirname(os.path.dirname(infinigen_examples.__file__)))"
    cand = subprocess.run([python, "-c", probe], capture_output=True, text=True).stdout.strip()
    if cand and os.path.isdir(os.path.join(cand, "infinigen_examples")):
        return cand
    raise SystemExit(f"gen_scene: no infinigen_examples/ under {infinigen_root!r} or {cand!r}")


def build_cmd(entry, python, out_dir, fp_json):
    """The Infinigen command: infinigen_style.py wrapping generate_indoors, with the entry's gin + bindings."""
    bindings = ["compose_indoors.terrain_enabled=False"]
    if os.environ.get("NO_FLOOR_PLAN", "0") != "1":
        bindings.append(f'Solver.floor_plan="{fp_json}"')
    if os.environ.get("NO_ROOM_RESTRICT", "0") != "1":  # dual-room: furnish every room
        bindings.append(f'restrict_solving.restrict_parent_rooms=["{entry["room"]}"]')
    bindings += list(entry.get("pextra", []))
    gin = list(entry["gin"])
    return (
        [python, os.path.join(HERE, "infinigen_style.py")]
        + ["--seed", str(entry["seed"]), "--task", "coarse", "--output_folder", out_dir]
        + (["-g"] + gin if gin else [])
        + ["-p"]
        + bindings
    )


def room_tally(out_dir):
    """Room types actually solved, from solve_state.json. A single-room scene must yield one."""
    p = os.path.join(out_dir, "solve_state.json")
    if not os.path.isfile(p):
        return None
    try:
        d = json.load(open(p))
        objs = d.get("objs", d)
        pat = re.compile(
            r"(living-room|bedroom|kitchen|bathroom|dining-room|hallway|closet"
            r"|garage|balcony|utility|staircase|open-office|meeting-room"
            r"|break-room|factory-office|office|warehouse|restroom)"
        )
        tally = {}
        for k in objs:
            m = pat.search(k)
            if m:
                tally[m.group(1)] = tally.get(m.group(1), 0) + 1
        return {"n_objs": len(objs), "rooms": tally}
    except Exception as e:
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="scene list JSON (spec/scenes_{single,dual}.json)")
    ap.add_argument("--idx", type=int, help="scene index in the manifest (this or --name)")
    ap.add_argument("--name", help="scene name in the manifest, e.g. scene_000042 (this or --idx)")
    ap.add_argument("--out-root", required=True, help="scene dir is <out-root>/<name>")
    ap.add_argument("--infinigen-root", default=os.environ.get("INFINIGEN_ROOT", "/opt/infinigen_src"))
    ap.add_argument("--python", default=os.environ.get("IPY", sys.executable), help="Python with Infinigen")
    ap.add_argument("--timeout-s", type=int, default=int(os.environ.get("GEN_TIMEOUT_S", "18000")))
    a = ap.parse_args()

    entry = next((e for e in json.load(open(a.manifest))["scenes"] if e["idx"] == a.idx or e["name"] == a.name), None)
    if entry is None:
        raise SystemExit(f"gen_scene: no manifest entry for idx={a.idx} name={a.name}")
    out_dir = os.path.join(a.out_root, entry["name"])
    os.makedirs(out_dir, exist_ok=True)
    fp_json = os.path.join(out_dir, "floorplan.json")
    json.dump(entry["floorplan"], open(fp_json, "w"), indent=1)
    for key, var, fmt in STYLE_ENV:
        if entry.get(key) is not None and not os.environ.get(var):
            os.environ[var] = fmt(entry[key])

    cmd = build_cmd(entry, a.python, out_dir, fp_json)
    logp = os.path.join(out_dir, "gen.log")
    t0 = time.time()
    status, rc = "ok", 0
    with open(logp, "w") as lf:
        lf.write("CMD " + " ".join(repr(c) if " " in c else c for c in cmd) + "\n")
        lf.flush()
        try:
            cwd = resolve_cwd(a.infinigen_root, a.python)
            rc = subprocess.call(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT, timeout=a.timeout_s)
            if rc != 0:
                status = "failed"
        except subprocess.TimeoutExpired:
            status, rc = "timeout", -9
        except Exception as e:
            status, rc = f"error:{e}", -1

    blend = os.path.join(out_dir, "scene.blend")
    have = os.path.isfile(blend)
    if status == "ok" and not have:
        status = "no_blend"
    rec = {k: entry[k] for k in ("idx", "name", "seed", "room", "tier", "W", "D")}
    rec.update(status=status, rc=rc, gen_s=round(time.time() - t0, 1), pred_gen_min=entry.get("pred_gen_min"))
    rec.update(blend=blend if have else None, blend_mb=round(os.path.getsize(blend) / 1e6, 1) if have else 0.0)
    rec.update(tally=room_tally(out_dir), log=logp, out_dir=out_dir)
    json.dump(rec, open(os.path.join(out_dir, "gen_result.json"), "w"), indent=1)
    print(f"{'GEN_OK' if status == 'ok' else 'GEN_FAIL'} {entry['name']} status={status} {rec['gen_s']}s", flush=True)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
