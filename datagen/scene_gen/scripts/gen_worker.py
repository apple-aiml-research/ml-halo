#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""gen_worker.py -- background generation pool that keeps the GPU fed.

Runs gen_scene.py for this shard's scenes, a few at a time, and drops <queue>/<name>.ready (or .failed)
for the bake loop in pod/run_shard.sh. New scenes wait while free memory / disk is under a floor or
GEN_MAX_READY scenes are already waiting.
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def _read(path):
    with open(path) as f:
        return f.read().strip()


def mem_available_gb():
    """GB the pod may still use: the cgroup limit (v2, then v1), since /proc/meminfo reports the host."""
    try:
        raw = _read("/sys/fs/cgroup/memory.max")
        if raw != "max":
            return (int(raw) - int(_read("/sys/fs/cgroup/memory.current"))) / 1e9
    except Exception:
        pass
    try:
        limit = int(_read("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
        if limit < (1 << 62):
            return (limit - int(_read("/sys/fs/cgroup/memory/memory.usage_in_bytes"))) / 1e9
    except Exception:
        pass
    try:
        for ln in open("/proc/meminfo"):
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1e6
    except Exception:
        pass
    return float("inf")


def disk_free_gb(path):
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize / 1e9
    except Exception:
        return float("inf")


def auto_workers():
    """clamp(nproc // 6, 1, 4), leaving cores for the bake."""
    try:
        n = len(os.sched_getaffinity(0))
    except Exception:
        n = os.cpu_count() or 4
    return max(1, min(4, n // 6))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="scene list JSON (spec/scenes_{single,dual}.json)")
    ap.add_argument("--scenes", required=True, help="comma-separated scene indices, in consume order")
    ap.add_argument("--out-root", required=True, help="scene dirs are written as <out-root>/<name>")
    ap.add_argument("--queue", required=True, help="dir of .ready/.failed markers the bake loop polls")
    ap.add_argument("--workers", type=int, default=0, help="0 = auto from nproc")
    a = ap.parse_args()
    mem_floor = float(os.environ.get("GEN_MEM_FLOOR_GB", "60"))
    disk_floor = float(os.environ.get("GEN_DISK_FLOOR_GB", "40"))
    max_ready = int(os.environ.get("GEN_MAX_READY", "3"))
    timeout_s = os.environ.get("GEN_TIMEOUT_S", "18000")
    python = os.environ.get("IPY", sys.executable)

    workers = a.workers or auto_workers()
    os.makedirs(a.queue, exist_ok=True)
    os.makedirs(a.out_root, exist_ok=True)
    by_idx = {e["idx"]: e for e in json.load(open(a.manifest))["scenes"]}
    pending = [i for i in (int(x) for x in a.scenes.split(",") if x.strip()) if i in by_idx]
    print(f"[gw] workers={workers} scenes={len(pending)} mem_floor={mem_floor}GB disk_floor={disk_floor}GB", flush=True)

    def mark(name, kind, rec):
        with open(os.path.join(a.queue, f"{name}.{kind}"), "w") as f:
            json.dump(rec, f)

    ready_count = lambda: len([f for f in os.listdir(a.queue) if f.endswith(".ready")])
    running = {}  # idx -> (Popen, start_time)
    while pending or running:
        for idx, (proc, t0) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            name = by_idx[idx]["name"]
            ok = rc == 0 and os.path.isfile(os.path.join(a.out_root, name, "scene.blend"))
            try:
                rec = json.load(open(os.path.join(a.out_root, name, "gen_result.json")))
            except Exception:
                rec = {"idx": idx, "name": name}
            rec.update(rc=rc, wall_s=round(time.time() - t0, 1))
            mark(name, "ready" if ok else "failed", rec)
            print(f"[gw] {'READY' if ok else 'FAILED'} {name} rc={rc} {rec['wall_s']}s", flush=True)
            del running[idx]

        while pending and len(running) < workers and ready_count() < max_ready:
            mem, disk = mem_available_gb(), disk_free_gb(a.out_root)
            if mem < mem_floor or disk < disk_floor:  # the bake is the priority consumer
                print(f"[gw] hold: MemAvailable {mem:.0f}GB, free disk {disk:.0f}GB", flush=True)
                break
            idx = pending.pop(0)
            name = by_idx[idx]["name"]
            if os.path.isfile(os.path.join(a.out_root, name, "scene.blend")):
                mark(name, "ready", {"idx": idx, "name": name, "resumed": True})
                continue
            cmd = [python, os.path.join(HERE, "gen_scene.py"), "--manifest", a.manifest, "--idx", str(idx)]
            cmd += ["--out-root", a.out_root, "--timeout-s", timeout_s]
            lf = open(os.path.join(a.out_root, f"{name}.worker.log"), "w")
            running[idx] = (subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT), time.time())
            print(f"[gw] START {name} (running={len(running)}/{workers} mem={mem:.0f}GB disk={disk:.0f}GB)", flush=True)
        if pending or running:
            time.sleep(10.0)
    print("[gw] pool drained", flush=True)


if __name__ == "__main__":
    main()
