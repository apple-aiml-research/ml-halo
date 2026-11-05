#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""run_real.py — launcher for the REAL robot (does NOT start the tracker).

Reads the same mission YAML as ``run_sim.py`` and launches on THIS box the generator
(``nodes/motiongen_zmq.py``, owns the real camera), the orchestrator (``nodes/orchestrator_zmq.py``) and,
with ``--web``, the web viewer. The C++ ``motiontrack_real_zmq`` tracker runs on the robot host
and is started separately, so the YAML's tracker keys (``scene_xml``, ``start``, ``policy_path``,
``robot_config``) are ignored. Nothing is saved: the viewer's live plan and the halo_serve files go to a
temporary folder that is deleted on exit.

    # 1. on the robot host: motiontrack_real_zmq --config ... --policy ... --net eth0
    # 2. export HALO_COMM_REGISTRY=<robot-host>:5560
    # 3. on this box:
    python system/halo_system/scripts/run_real.py \\
        --task_config system/halo_system/scripts/configs/<task>.yaml [--web]

Two boxes: run ``--motiongen`` on one box and ``--orchestrator`` on the other (each launched by hand).
CLI flags override the YAML.
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_SYS = _REPO / "system"
GEN_NODE = str(_SYS / "halo_system" / "nodes" / "motiongen_zmq.py")
MISSION_NODE = str(_SYS / "halo_system" / "nodes" / "orchestrator_zmq.py")
VIEWER_MODULE = "halo_system.visualization.server"

# Import this checkout's packages.
_PATHS = (str(_REPO / "system" / "comm" / "python"), str(_REPO / "system" / "serve" / "python"),
          str(_SYS), str(_REPO), str(_REPO / "orchestrator"))
for _p in reversed(_PATHS):  # sys.path order == PYTHONPATH order
    if _p not in sys.path:
        sys.path.insert(0, _p)

from halo_serve.serve_env import serve_env, service_env

# The robot-host tracker's embedded registry.
_DEFAULT_REGISTRY = "192.168.123.164:5560"


def _free_ports(ports) -> None:
    """Kill stale processes on this box's node ports (best-effort, via lsof).

    Tracker ports (5555-5558) are not touched: the tracker runs on the robot host."""
    for p in ports:
        try:
            out = subprocess.run(["lsof", "-nP", f"-tiTCP:{p}"],
                                 capture_output=True, text=True, timeout=5).stdout.split()
        except Exception:  # noqa: BLE001 — lsof missing / no perms: skip
            continue
        for pid in out:
            try:
                os.kill(int(pid), signal.SIGKILL)
                print(f"[real] freed stale process on port {p} (PID {pid})", flush=True)
            except Exception:  # noqa: BLE001
                pass


def _registry() -> str:
    return os.environ.get("HALO_COMM_REGISTRY") or _DEFAULT_REGISTRY


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task_config", required=True,
                   help="Task YAML. Same file the sim launcher reads; scene/start/policy/"
                        "robot_config are IGNORED (those configure the tracker).")
    p.add_argument("--duration", type=float, default=0.0,
                   help="Generator runtime seconds (0 = run until Ctrl-C). YAML `duration` used if set "
                        "and this is 0.")
    # camera (the real generator owns it)
    p.add_argument("--oak_camera_index", type=int, default=0,
                   help="OAK/DepthAI camera socket index (default 0).")
    # Per-node mode, e.g.:
    #   generator box:  --motiongen --mission_host <orchestrator box IP>
    #   orchestrator box:    --orchestrator --web --obs_host <generator box IP>
    p.add_argument("--motiongen", action="store_true",
                   help="Run the MOTION GENERATOR on this box (owns the camera + GPU).")
    p.add_argument("--orchestrator", dest="orchestrator_node", action="store_true",
                   help="Run the MISSION ORCHESTRATOR on this box (needs network access to its LLM).")
    # The viewer runs on the orchestrator's box: the operator inbox it writes is a local file.
    p.add_argument("--web", action="store_true",
                   help="Also launch the web viewer on this mission config, on the box that runs "
                        "the orchestrator, and open it in the browser (like run_sim).")
    p.add_argument("--web_port", type=int, default=8090, help="Web viewer HTTP port for --web.")
    p.add_argument("--no_open", action="store_true",
                   help="With --web, start the web viewer but do NOT open a browser.")
    p.add_argument("--mission_host", type=str, default="auto",
                   help="Host running the ORCHESTRATOR (where g1/mission is bound). Default 'auto' = "
                        "resolve from the registry, which is correct whether the orchestrator is on this "
                        "box or another one. Pass an explicit IP to bypass discovery, or "
                        "'127.0.0.1' to force local.")
    p.add_argument("--obs_host", type=str, default="auto",
                   help="Host running the GENERATOR (where g1/obs + g1/gen_status are bound). Set "
                        "this on the ORCHESTRATOR box when the generator is on another machine. 'auto' "
                        "resolves it from the registry -- with HALO_COMM_REGISTRY exported on both "
                        "boxes, --obs_host auto + --mission_host auto need no IPs at all.")
    p.add_argument("--mission_port", type=int, default=5554, help="g1/mission (orchestrator->generator).")
    p.add_argument("--obs_port", type=int, default=5559, help="g1/obs (generator->orchestrator/viewer).")
    p.add_argument("--genstatus_port", type=int, default=5561, help="g1/gen_status (generator->orchestrator).")
    # tracker ports (robot host)
    p.add_argument("--state_port", type=int, default=5555)
    p.add_argument("--plan_port", type=int, default=5556)
    p.add_argument("--trim_port", type=int, default=5557)
    p.add_argument("--state_pub_port", type=int, default=5558)
    return p


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    args = build_parser().parse_args(argv)

    import yaml
    mcfg = yaml.safe_load(Path(args.task_config).expanduser().read_text()) or {}

    subs = mcfg.get("subtasks") or {}
    if not subs:
        print(f"[ERROR] {args.task_config} has no `subtasks:` block — nothing to run.")
        return 2
    sub_names = list(subs.keys())
    subtask_prompts = ";".join(
        f"{k}={v['prompt']}" for k, v in subs.items() if v.get("prompt"))
    initial = mcfg.get("initial_subtask") or sub_names[0]
    if initial not in subs:
        print(f"[ERROR] initial_subtask {initial!r} not in subtasks {sub_names}.")
        return 2
    # One generator for every subtask. Specs are resolved inside the generator (resolve_asset), so
    # an orchestrator-only box needs neither file.
    checkpoint, gen_config = mcfg.get("checkpoint"), mcfg.get("gen_config")
    if not gen_config or not checkpoint:
        print("[ERROR] task YAML needs `checkpoint` and `gen_config` (the generator).")
        return 2
    checkpoint, gen_config = str(checkpoint), str(gen_config)

    mission = mcfg.get("instruction") or ""
    pl = mcfg.get("orchestrator") or {}
    # No node flag (or both) = single box.
    roles = ("gen" if args.motiongen and not args.orchestrator_node
             else "orchestrator" if args.orchestrator_node and not args.motiongen else "both")
    if args.motiongen or args.orchestrator_node:
        if (roles == "gen" and str(args.mission_host).lower() == "auto"
                and not os.environ.get("HALO_COMM_REGISTRY")):
            print("[nodes] WARNING: generator only, but --mission_host is 'auto' and "
                  "HALO_COMM_REGISTRY is unset -> it will look for the orchestrator on 127.0.0.1. Pass "
                  "--mission_host <orchestrator box IP>.", flush=True)
        print("[nodes] this box runs: "
              + ", ".join(n for n, on in (("motiongen", args.motiongen),
                                          ("orchestrator", args.orchestrator_node),
                                          ("web viewer", args.web and args.orchestrator_node)) if on),
              flush=True)
        if args.web and not args.orchestrator_node:
            print("[nodes] NOTE: --web starts the web viewer only on the box that runs --orchestrator "
                  "(it writes the orchestrator's operator inbox); ignored here.", flush=True)
    if roles == "orchestrator" and bool(pl.get("enabled", True)):
        print("[nodes] ERROR: the local orchestrator model (halo_serve) is served by the generator's "
              "engine over a local socket -- run --motiongen and --orchestrator on the same box.",
              flush=True)
        return 2
    # Single-task YAMLs set `agentic: false` + `orchestrator.enabled: false` (fixed generator prompt).
    agentic = bool(mcfg.get("agentic", True))
    orchestrator_enabled = bool(pl.get("enabled", True)) and roles in ("both", "orchestrator")
    gen_enabled = roles in ("both", "gen")
    gen_prompt = "" if agentic else (mcfg.get("prompt") or subs[initial].get("prompt") or "")
    # Skill enum as in run_sim: the subtask keys, in registry order.
    from halo_orchestrator import registry as _skill_registry
    choices = _skill_registry.enum_for_run(loaded=sub_names)
    duration = args.duration if args.duration > 0 else float(mcfg.get("duration") or 0.0)

    # Scratch folder for the live plan and halo_serve files; deleted on exit (nothing is saved).
    run_dir = Path(tempfile.mkdtemp(prefix="halo_real_"))

    gen = [sys.executable, GEN_NODE,
           "--config", gen_config,
           "--checkpoint", checkpoint,
           "--prompt", gen_prompt,
           "--obs_port", str(args.obs_port),
           "--mission_port", str(args.mission_port),
           "--genstatus_port", str(args.genstatus_port),
           "--state_port", str(args.state_port),
           "--plan_port", str(args.plan_port),
           "--trim_port", str(args.trim_port),
           "--state_pub_port", str(args.state_pub_port),
           "--output_dir", str(run_dir),
           "--mission_host", str(args.mission_host),
           "--duration", str(duration)]
    _gi = mcfg.get("gen_interval", 0.4)
    if _gi is not None:
        gen += ["--gen_interval", str(float(_gi))]
    _raq = ",".join(k for k, v in subs.items()
                    if v.get("rebase_robot_when_quiet"))
    if _raq:
        gen += ["--robot_anchor_when_quiet", _raq]
    if subtask_prompts:
        # Warm up every prompt: one conditioning shape per prompt length (as run_sim.py).
        gen += ["--warmup_prompts", subtask_prompts]
    if agentic:
        gen.append("--agentic")
    if not bool(pl.get("enabled", True)):
        # No orchestrator: subscribing would wait in discovery or attach to another run's orchestrator.
        # Keyed on the mission, not the node flags (a split generator box still needs g1/mission).
        gen.append("--no_mission")
    # Camera: the OAK.
    gen.append("--real")
    gen += ["--oak_camera_index", str(args.oak_camera_index)]

    plan = [sys.executable, MISSION_NODE,
            "--host", str(args.obs_host),
            # g1/state comes from the tracker on the robot host
            "--state_host", _registry().rsplit(":", 1)[0],
            "--obs_port", str(args.obs_port),
            "--mission_port", str(args.mission_port),
            "--genstatus_port", str(args.genstatus_port),
            "--subtasks", ",".join(choices)]
    if subtask_prompts:
        plan += ["--subtask_prompts", subtask_prompts]
    if mission:
        plan += ["--instruction", mission]
    if pl.get("model"):
        plan += ["--model", str(pl["model"])]
    if pl.get("quiet_travel_m") is not None:            # 0 = joint-only quiet
        plan += ["--quiet_travel_m", str(float(pl["quiet_travel_m"]))]

    env = os.environ.copy()
    cur_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join(list(_PATHS) + ([cur_pp] if cur_pp else []))
    env["PYTHONUNBUFFERED"] = "1"
    # Same offline HF defaults as run_sim.py (the mini has no internet; models are pre-cached).
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # The nodes find the tracker through this registry (default: the NX).
    env.setdefault("HALO_COMM_REGISTRY", _registry())
    # Generator acceleration: same `halo_serve:` handling as run_sim.py.
    env.update(serve_env(mcfg.get("halo_serve"), run_dir))

    _mode = "multi-task (agentic + orchestrator)" if agentic else "single-task (fixed prompt)"
    print(f"[real] task={mcfg.get('name', args.task_config)}  subtasks={sub_names}  "
          f"initial={initial}")
    print(f"[real] mode            = {_mode}")
    print(f"[real] orchestrator         = {'on' if orchestrator_enabled else 'OFF'}")
    if not agentic:
        print(f"[real] fixed prompt    = {gen_prompt!r}")
    print(f"[real] orchestrator choices = {choices}")
    print(f"[real] checkpoint      = {checkpoint}")
    print("[real] NOTE: the tracker is NOT started here — run motiontrack_real_zmq on the robot "
          "host and set HALO_COMM_REGISTRY so the generator resolves g1/* to it.")

    # Viewer: orchestrator channels are local (127.0.0.1), g1/state via --registry, g1/obs via --obs_host.
    viewer_cmd = None
    if args.web and orchestrator_enabled:
        _obs = ("127.0.0.1" if gen_enabled or str(args.obs_host).lower() == "auto"
                else str(args.obs_host))
        viewer_cmd = [sys.executable, "-u", "-m", VIEWER_MODULE,
                      "--port", str(args.web_port),
                      "--task_config", str(Path(args.task_config).expanduser().resolve()),
                      "--live_host", "127.0.0.1",
                      "--live_port", str(args.state_pub_port),     # g1/state PUB on the robot host
                      "--obs_host", _obs, "--obs_port", str(args.obs_port),
                      "--mission_port", str(args.mission_port),
                      "--live_run_dir", str(run_dir),  # generator writes live_plan.npz here (live plan overlay)
                      "--no_scene",          # real robot: no sim scene in the viewport
                      "--registry", _registry()]

    if gen_enabled:   # stop on a placeholder model path before anything starts
        from halo_system.lib.assets import resolve_asset
        resolve_asset(checkpoint), resolve_asset(gen_config)
    _free_ports((args.mission_port, args.obs_port, args.genstatus_port)
                + ((args.web_port,) if viewer_cmd else ()))

    procs = []
    viewer = None
    try:
        # Orchestrator first (ZMQ reconnects anyway; this keeps the logs in order).
        if orchestrator_enabled:
            env.update(service_env(plan, run_dir))
            print(f"\n[real] launching orchestrator:\n  {' '.join(plan)}\n", flush=True)
            procs.append(subprocess.Popen(plan, env=env, cwd=str(_REPO)))
            time.sleep(2.0)
        else:
            if roles == "gen":
                print(f"\n[real] orchestrator NOT launched (--motiongen only) — expecting it on "
                      f"{args.mission_host}.\n", flush=True)
            else:
                print("\n[real] orchestrator DISABLED (single-task) — running the generator's fixed "
                      "prompt only.\n", flush=True)
        if gen_enabled:
            print(f"\n[real] launching generator:\n  {' '.join(gen)}\n", flush=True)
            procs.append(subprocess.Popen(gen, env=env, cwd=str(_REPO)))
        else:
            print("\n[real] generator NOT launched (--orchestrator only) — expecting it on "
                  f"{args.obs_host}.\n", flush=True)
        if viewer_cmd:
            print(f"\n[real] launching web viewer:\n  {' '.join(viewer_cmd)}\n", flush=True)
            # Not in `procs`: a failing viewer must not stop the robot.
            viewer = subprocess.Popen(viewer_cmd, env=env, cwd=str(_REPO))
            _url = f"http://127.0.0.1:{args.web_port}/"
            if not args.no_open:
                # Wait for the port, or the browser races the server.
                import socket as _s
                for _ in range(60):
                    if viewer.poll() is not None:
                        print("[viewer] exited during startup -- not opening a browser", flush=True)
                        break
                    with _s.socket() as _c:
                        _c.settimeout(0.3)
                        if _c.connect_ex(("127.0.0.1", args.web_port)) == 0:
                            import webbrowser
                            webbrowser.open(_url)
                            break
                    time.sleep(0.25)
            print(f"[viewer] {_url}", flush=True)
        if not procs:
            print(f"[real] nothing to launch for roles={roles} "
                  f"(orchestrator.enabled={pl.get('enabled', True)}).", flush=True)
            return 2
        # If one child exits, tear the others down.
        while True:
            for pr in procs:
                rc = pr.poll()
                if rc is not None:
                    print(f"\n[real] a child exited (rc={rc}) — stopping the other.", flush=True)
                    raise KeyboardInterrupt
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[real] stopping…", flush=True)
    finally:
        if viewer is not None and viewer.poll() is None:
            viewer.terminate()
        for pr in procs:
            if pr.poll() is None:
                pr.terminate()
        for pr in procs:
            try:
                pr.wait(timeout=6.0)
            except subprocess.TimeoutExpired:
                pr.kill()
        shutil.rmtree(run_dir, ignore_errors=True)
        print("[real] done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
