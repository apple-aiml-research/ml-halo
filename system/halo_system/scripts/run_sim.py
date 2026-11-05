#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Sim launcher: C++ MuJoCo tracker + multi-model generator + mission orchestrator over halo_comm.

Runs the full split deployment as separate OS processes:
  * the C++ tracker (``system/build/<plat>/motiontrack_mujoco_zmq``: policy + physics + ego render),
  * the generator (``nodes/motiongen_zmq.py --mujoco``): one model per subtask,
    swapped when the orchestrator changes the subtask on ``g1/mission``,
  * the mission orchestrator (``nodes/orchestrator_zmq.py``), optional,
  * the web viewer (``python -m halo_system.visualization.server``) with ``--web``.

The whole task comes from ONE YAML file:

    python system/halo_system/scripts/run_sim.py \\
        --task_config system/halo_system/scripts/configs/<task>.yaml

SINGLE-TASK vs MULTI-TASK is declared IN the task YAML:
  * multi-task (default): ``agentic: true`` + ``orchestrator.enabled: true`` -> the generator holds for
    the orchestrator, which switches subtasks over ``g1/mission``.
  * single-task: ``agentic: false`` + ``orchestrator.enabled: false`` -> no orchestrator runs and the
    generator runs a fixed prompt (top-level ``prompt:`` or the initial subtask's ``prompt``).

TASK YAML keys (each maps to a launcher flag; an explicit CLI flag still overrides):
  models:      checkpoint + gen_config (the generator, one for every subtask),
               policy_path + robot_config (the tracker); see "Tasks" in system/README.md
  scene/run:   scene_xml, start.{pos,yaw}, duration, gen_interval
  objects:     hide_bodies (hidden in the sim at startup)
  orchestrator:     orchestrator.{enabled, model, quiet_travel_m}
  serving:     halo_serve (bool, from the task YAML)
  subtasks:    <name>.{prompt, rebase_robot_when_quiet}

Asset specs (checkpoint, gen_config, policy_path, scene_xml, robot_config) may be ``~``/env-var
paths, absolute paths or paths relative to the repository root.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Repo-anchored so the launcher runs from any CWD (<root>/system/halo_system/scripts/run_sim.py).
REPO_ROOT = Path(__file__).resolve().parents[3]

for _p in (str(REPO_ROOT / "system" / "comm" / "python"), str(REPO_ROOT / "system" / "serve" / "python"),
           str(REPO_ROOT / "system"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.append(str(REPO_ROOT / "orchestrator"))

import halo_comm                                     # noqa: E402
from halo_system.lib.assets import resolve_asset     # noqa: E402

GEN_NODE = str(REPO_ROOT / "system/halo_system/nodes/motiongen_zmq.py")
MISSION_NODE = str(REPO_ROOT / "system/halo_system/nodes/orchestrator_zmq.py")

DEFAULT_SCENE = "assets/scenes/bedroom/bedroom.xml"

# Set file the C++ tracker polls; written once at startup to hide hide_bodies.
DEFAULT_SET_FILE = "/tmp/halo_set_objects.txt"
DEFAULT_OBJ_FILE = "/tmp/halo_objects.txt"      # the viewer reads the same default


def default_tracker_bin() -> str:
    """First existing C++ tracker build (mac, then linux), else the mac path as a placeholder."""
    for plat in ("mac", "linux"):
        cand = REPO_ROOT / "system" / "build" / plat / "motiontrack_mujoco_zmq"
        if cand.exists():
            return str(cand)
    return str(REPO_ROOT / "system/build/mac/motiontrack_mujoco_zmq")


def make_pythonpath_env() -> dict:
    """Copy of the env with this checkout's packages prepended to PYTHONPATH."""
    env = dict(os.environ)
    extra = ":".join(str(REPO_ROOT / d) for d in ("system/comm/python", "system/serve/python", "system", "orchestrator", ""))
    env["PYTHONPATH"] = extra + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    # The VLM encoder checks the HF hub at load and crashes offline; encoders are pre-cached, so
    # default to offline (export HF_HUB_OFFLINE=0 to override).
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    return env


def _hide_bodies(set_file: str, bodies: list[str]) -> None:
    """Hide scene bodies in ONE set-file write (the tracker acts on each new token once, so
    separate writes inside one poll window would drop all but the last), then wait for the poll."""
    tmp = f"{set_file}.tmp"
    with open(tmp, "w") as f:
        f.write(f"{time.time():.4f}\n" + "".join(f"hide {b}\n" for b in bodies))
    os.replace(tmp, set_file)
    time.sleep(0.3)
    print(f"   [scene] hidden: {', '.join(bodies)}")


def wait_for_tracker(host, state_port, plan_port, trim_port, state_pub_port,
                     timeout_s, is_alive) -> bool:
    """Poll the tracker's get_state (via the generator's TrackerClient) until it is populated;
    ``is_alive()`` returning False aborts early."""
    from halo_system.lib.zmq_helper import TrackerClient
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not is_alive():
            return False
        cli = None
        try:
            cli = TrackerClient(host, state_port, plan_port, trim_port,
                                state_pub_port, rcv_ms=1000, snd_ms=1000)
            view = cli.get_state()
            if np.asarray(view.base_pos).size >= 3:
                return True
        except Exception:  # noqa: BLE001 — not up yet
            pass
        finally:
            if cli is not None:
                try:
                    cli.close()
                except Exception:  # noqa: BLE001
                    pass
        time.sleep(1.0)
    return False


def terminate_process(proc: subprocess.Popen, name: str, grace_s: float = 10.0) -> None:
    """SIGINT → wait → SIGTERM → wait → SIGKILL a child process."""
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.terminate()
    try:
        proc.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        pass
    print(f"[WARN] {name} did not exit; killing.")
    proc.kill()
    proc.wait()


# Extra generator/orchestrator args from the task YAML, appended to the subprocess argvs.
#   agentic: generator subscribes g1/mission + holds for the orchestrator.  orchestrator: launch the orchestrator.
_EXTRA = {"agentic": True, "orchestrator": True, "name": None, "hide_bodies": None,
          "robot_anchor_when_quiet": None, "halo_serve": None}
_ORCHESTRATOR_EXTRA = {"subtasks": None, "subtask_prompts": None, "quiet_travel_m": None}


def _image_groups(robot_config: str) -> list[dict]:
    """The robot config's image observation groups (non-empty = a depth policy)."""
    import yaml
    with open(robot_config) as f:
        groups = (yaml.safe_load(f).get("observation") or {}).get("groups") or {}
    return [g for g in groups.values() if isinstance(g, dict) and g.get("layout") == "image"]


TRACKER_WARMUP_S = 180.0   # max wait for the tracker to serve get_state


def build_tracker_cmd(a) -> list[str]:
    """The C++ tracker subprocess argv."""
    # duration 0 = unlimited -> a horizon no session reaches. The tracker's clock starts at launch,
    # the generator's after model warm-up (budget 1800 s).
    tracker_duration = (1.0e6 if float(a.duration) == 0.0
                        else a.duration + TRACKER_WARMUP_S + 60.0 + 1800.0)
    cmd = [
        a.tracker_bin,
        "--config", a.robot_config,
        "--policy", a.policy_path,
        "--bind_ip", a.bind_ip,
        "--state_port", str(a.state_port),
        "--plan_port", str(a.plan_port),
        "--trim_port", str(a.trim_port),
        "--state_pub_port", str(a.state_pub_port),
        "--duration", str(tracker_duration),
        "--init_yaw", str(a.init_yaw),
        "--init_pos", *[str(x) for x in a.init_pos],
    ]
    if a.scene_xml:
        cmd += ["--scene_xml", a.scene_xml]
    cmd += ["--ego_render",
            "--ego_camera_name", a.ego_camera,
            "--ego_render_width", str(a.ego_width),
            "--ego_render_height", str(a.ego_height)]
    if _image_groups(a.robot_config):
        # Depth policy: the tracker renders the torso depth camera and feeds it to the policy.
        cmd += ["--depth_render"]
    return cmd


def build_generator_cmd(a, run_dir: Path) -> list[str]:
    """The generator argv; image + state both come from the tracker's get_state."""
    cmd = [
        sys.executable, GEN_NODE,
        "--mujoco",
        "--config", a.config,
        "--checkpoint", a.checkpoint,
        "--prompt", a.prompt,
        "--state_port", str(a.state_port),
        "--plan_port", str(a.plan_port),
        "--trim_port", str(a.trim_port),
        "--state_pub_port", str(a.state_pub_port),
        "--gen_interval", str(a.gen_interval),
        "--duration", str(a.duration),
        "--output_dir", str(run_dir),
        "--mission_port", str(a.mission_port),
        "--obs_port", str(a.obs_port),
        "--genstatus_port", str(a.genstatus_port),
    ]
    if _EXTRA["robot_anchor_when_quiet"]:
        cmd += ["--robot_anchor_when_quiet", _EXTRA["robot_anchor_when_quiet"]]
    if _ORCHESTRATOR_EXTRA["subtask_prompts"]:
        # Warm up every prompt: one conditioning shape per prompt length.
        cmd += ["--warmup_prompts", _ORCHESTRATOR_EXTRA["subtask_prompts"]]
    if _EXTRA["agentic"]:
        cmd += ["--agentic"]
    return cmd


def build_orchestrator_cmd(a) -> list[str]:
    """The orchestrator argv; it SUBs g1/obs and PUBs the chosen subtask on g1/mission."""
    cmd = [
        sys.executable, MISSION_NODE,
        "--instruction", a.mission,
        "--host", a.tracker_host,
        "--obs_port", str(a.obs_port),
        "--mission_port", str(a.mission_port),
        "--plan_port", str(a.orchestrator_plan_port),
    ]
    if a.orchestrator_model:
        cmd += ["--model", a.orchestrator_model]
    if _ORCHESTRATOR_EXTRA["subtasks"]:
        cmd += ["--subtasks", _ORCHESTRATOR_EXTRA["subtasks"]]
    if _ORCHESTRATOR_EXTRA["subtask_prompts"]:
        cmd += ["--subtask_prompts", _ORCHESTRATOR_EXTRA["subtask_prompts"]]
    if _ORCHESTRATOR_EXTRA["quiet_travel_m"] is not None:
        cmd += ["--quiet_travel_m", str(_ORCHESTRATOR_EXTRA["quiet_travel_m"])]
    return cmd


def gen_env() -> dict:
    env = make_pythonpath_env()
    # Control files (always on; cheap when unread):
    #   HALO_SET_FILE   -> tracker polls it; run_session writes hide_bodies there once at startup.
    #   HALO_OBJ_FILE   -> tracker writes the free bodies' live poses (~10 Hz) for the web viewer.
    env.setdefault("HALO_SET_FILE", DEFAULT_SET_FILE)
    env.setdefault("HALO_OBJ_FILE", DEFAULT_OBJ_FILE)
    if sys.platform.startswith("linux"):
        # A tracker built against a staged onnxruntime has no RPATH -> put it on LD_LIBRARY_PATH.
        for _c in sorted((REPO_ROOT / "system/build/third_party").glob("onnxruntime-*/lib")):
            env["LD_LIBRARY_PATH"] = f"{_c}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
            break
        # cgl is macOS-only; it kills MuJoCo at GL init on Linux.
        if env.get("MUJOCO_GL") == "cgl":
            print("[run_sim] MUJOCO_GL=cgl is macOS-only — using egl on Linux", flush=True)
            env["MUJOCO_GL"] = "egl"
    return env


def run_session(a, run_dir: Path) -> str | None:
    """Launch the tracker, wait for get_state, start the orchestrator (optional), run the generator
    to completion, then tear everything down. All child output goes to this terminal.
    Returns an error message if the tracker never came up, else None.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    tracker_cmd = build_tracker_cmd(a)
    gen_cmd = build_generator_cmd(a, run_dir)
    env = gen_env()

    # Bring up the discovery registry if HALO_COMM_REGISTRY names this machine (idempotent).
    halo_comm.init()

    print(f"   [tracker cmd] {' '.join(tracker_cmd)}")
    print(f"   [generator cmd] {' '.join(gen_cmd)}")

    tracker = subprocess.Popen(tracker_cmd, env=env, cwd=str(REPO_ROOT))
    orchestrator = None
    try:
        print("   [tracker] starting")
        _t_ready = time.time()
        if not wait_for_tracker(a.tracker_host, a.state_port, a.plan_port,
                                a.trim_port, a.state_pub_port, TRACKER_WARMUP_S,
                                lambda: tracker.poll() is None):
            if tracker.poll() is not None:
                return f"tracker exited during warmup (rc={tracker.returncode})"
            return f"tracker not ready within {TRACKER_WARMUP_S}s"
        print(f"   [timing] tracker warmup->ready: {time.time() - _t_ready:.1f}s", flush=True)
        print("   [tracker] ready; launching generator")
        if _EXTRA["hide_bodies"]:
            _hide_bodies(env["HALO_SET_FILE"], _EXTRA["hide_bodies"])
        # Orchestrator starts first so it is subscribed to g1/obs before the first frame.
        if a.orchestrator:
            orchestrator_cmd = build_orchestrator_cmd(a)
            # The local orchestrator model: both processes share the generator's serving endpoint.
            from halo_serve.serve_env import service_env
            env.update(service_env(orchestrator_cmd, run_dir))
            print(f"   [orchestrator cmd] {' '.join(orchestrator_cmd)}")
            print("   [orchestrator] streaming to console")
            orchestrator = subprocess.Popen(orchestrator_cmd, env=env,
                                       cwd=str(REPO_ROOT))
        _t_gen = time.time()
        subprocess.run(gen_cmd, env=env, cwd=str(REPO_ROOT))
        print(f"   [timing] generator subprocess (model-load + {a.duration:.0f}s gen + "
              f"shutdown): {time.time() - _t_gen:.1f}s", flush=True)
    finally:
        if orchestrator is not None:
            terminate_process(orchestrator, "orchestrator")
        terminate_process(tracker, "tracker")
    return None


def camera_bundle() -> dict:
    """The evaluation FPV camera preset as ONE dict.

    The OAK-1 fisheye training view: the tracker renders `oak1` as a wide 1600x1000 pinhole and
    warps it to the calibrated 1280x800 fisheye (automatic for any camera named "oak*"). Never
    pass these fields piecemeal: a mismatched render silently gives the generator
    out-of-distribution input.
    """
    return dict(ego_camera="oak1", ego_width=1600, ego_height=1000)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # tracker
    p.add_argument("--config", type=str, default=None,
                   help="Generator config yaml (task YAML: gen_config).")
    p.add_argument("--robot_config", type=str, default=None,
                   help="Tracker yaml (required; task YAML: robot_config).")
    p.add_argument("--policy_path", type=str, default=None,
                   help="Tracker onnx policy (required; task YAML: policy_path).")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Generator checkpoint (required; task YAML: checkpoint).")
    p.add_argument("--scene_xml", type=str, default=None,
                   help=f"Scene XML for the tracker (default {DEFAULT_SCENE}).")
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--init_yaw", type=float, default=45.0)
    p.add_argument("--init_pos", type=float, nargs=3, default=[-0.1, -0.1, 0.83])
    p.add_argument("--tracker_host", type=str, default="127.0.0.1")
    p.add_argument("--bind_ip", type=str, default="0.0.0.0")
    p.add_argument("--state_port", type=int, default=5555)
    p.add_argument("--plan_port", type=int, default=5556)
    p.add_argument("--trim_port", type=int, default=5557)
    p.add_argument("--state_pub_port", type=int, default=5558)
    # mission orchestrator
    p.add_argument("--orchestrator", action="store_true",
                   help="Launch the mission orchestrator (orchestrator_zmq.py): it watches g1/obs and "
                        "publishes the next subtask on g1/mission.")
    p.add_argument("--instruction", dest="mission", type=str, default="complete the task safely",
                   help="Operator instruction the orchestrator decomposes into subtasks.")
    p.add_argument("--orchestrator_model", type=str, default=None,
                   help="Local orchestrator model (repo id or path), co-served with the generator "
                        "through halo_serve (default mlx-community/Qwen3.6-35B-A3B-4bit).")
    p.add_argument("--obs_port", type=int, default=5559,
                   help="g1/obs PUB port shared by generator + orchestrator.")
    p.add_argument("--genstatus_port", type=int, default=5561,
                   help="g1/gen_status PUB port (generator -> orchestrator).")
    p.add_argument("--orchestrator_plan_port", type=int, default=5564,
                   help="g1/plan PUB port (orchestrator -> viewer). Shift it to run two simulations at once.")
    p.add_argument("--mission_port", type=int, default=5554,
                   help="g1/mission PUB/SUB port shared by orchestrator + generator "
                        "(5560 is reserved by the halo_comm discovery registry).")
    p.add_argument("--duration", type=float, default=15.0,
                   help="Generator runtime in seconds (0 = unlimited). Default 15 s.")
    p.add_argument("--gen_interval", type=float, default=0.0,
                   help="Min seconds between plans (0 = replan as fast as the model allows).")
    p.add_argument("--output_dir", type=str, default="eval_closedloop_sim_multi_output",
                   help="Run dir (halo_serve stats, the viewer's live_plan.npz).")
    return p


def _base_main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.policy_path:
        print("[ERROR] provide --policy_path (or policy_path: in the task YAML)",
              file=sys.stderr)
        return 2
    if not args.config:
        print("[ERROR] provide --config (or gen_config: in the task YAML)", file=sys.stderr)
        return 2
    if not args.checkpoint:
        print("[ERROR] provide --checkpoint (or checkpoint: in the task YAML)", file=sys.stderr)
        return 2
    if not args.robot_config:
        print("[ERROR] provide --robot_config (or robot_config: in the task YAML)",
              file=sys.stderr)
        return 2

    args.tracker_bin = default_tracker_bin()
    for attr in ("config", "robot_config", "policy_path", "checkpoint"):
        setattr(args, attr, resolve_asset(getattr(args, attr)))

    if args.gen_interval >= 5.0:
        print("[ERROR] gen_interval must be < 5s (plan is only 5s)")
        return 2

    vars(args).update(camera_bundle())

    if not Path(args.tracker_bin).exists():
        print(f"[ERROR] C++ tracker not found: {args.tracker_bin}\n"
              "        Build it with: bash system/cpp/scripts/build_motiontrack_mujoco_zmq.sh")
        return 2

    if not args.scene_xml:
        print(f"[INFO] No --scene_xml; using the default scene: {DEFAULT_SCENE}")
    args.scene_xml = resolve_asset(str(args.scene_xml or DEFAULT_SCENE))
    print(f"[INFO] Tracker: {args.tracker_bin}")
    print(f"[INFO] Checkpoint: {args.checkpoint}")
    print(f"[INFO] Scene: {args.scene_xml}")

    out_dir = Path(args.output_dir).resolve()
    print(f"\n=== Launching multi-process sim -> {out_dir} ===")
    error = run_session(args, out_dir)
    if error:
        print(f"[ERROR] {error}")
        return 1
    print("\n[done]")
    return 0


def _inject(rest, flag, *vals):
    """Append `flag vals...` to rest only if the user didn't already pass `flag` (CLI wins)."""
    if flag not in rest:
        rest += [flag, *[str(v) for v in vals]]


def _flag(rest, flag, default):
    """The value after `flag` in rest, else default."""
    if flag in rest and rest.index(flag) + 1 < len(rest):
        return rest[rest.index(flag) + 1]
    return default


def _apply_task_config(path, rest):
    """Load a task YAML and translate it into launcher argv + generator/orchestrator extras.
    Mutates `rest`, `_EXTRA`, `_ORCHESTRATOR_EXTRA`."""
    import yaml
    from halo_orchestrator import registry as _skill_registry
    mcfg = yaml.safe_load(Path(path).expanduser().read_text()) or {}
    _EXTRA["name"] = mcfg.get("name") or Path(path).stem
    # Resolved into the generator env in main() once the run dir is known.
    _EXTRA["halo_serve"] = mcfg.get("halo_serve")
    agentic = bool(mcfg.get("agentic", True))
    _EXTRA["agentic"] = agentic
    # Empty instruction -> the orchestrator waits for the operator to type one in the web viewer.
    _instr = mcfg.get("instruction") or ""
    _inject(rest, "--instruction", _instr)
    if mcfg.get("scene_xml"):
        _inject(rest, "--scene_xml", str(mcfg["scene_xml"]))
    start = mcfg.get("start", {}) or {}
    if start.get("pos"):
        _inject(rest, "--init_pos", *start["pos"])
    if start.get("yaw") is not None:
        _inject(rest, "--init_yaw", start["yaw"])
    if mcfg.get("policy_path"):
        _inject(rest, "--policy_path", str(mcfg["policy_path"]))
    if mcfg.get("robot_config"):
        _inject(rest, "--robot_config", str(mcfg["robot_config"]))
    if mcfg.get("gen_config"):
        _inject(rest, "--config", str(mcfg["gen_config"]))
    if mcfg.get("checkpoint"):
        _inject(rest, "--checkpoint", str(mcfg["checkpoint"]))
    if mcfg.get("duration") is not None:
        _inject(rest, "--duration", mcfg["duration"])
    else:
        # A mission runs until done / Ctrl-C; `duration:` caps it.
        _inject(rest, "--duration", 3600)
    if mcfg.get("gen_interval") is not None:
        _inject(rest, "--gen_interval", mcfg["gen_interval"])
    _EXTRA["robot_anchor_when_quiet"] = ",".join(
        k for k, v in (mcfg.get("subtasks") or {}).items()
        if isinstance(v, dict) and v.get("rebase_robot_when_quiet")) or None

    # Body-name list (YAML list or comma string).
    def _names(v):
        if v is None:
            return []
        return [str(x).strip() for x in (v if isinstance(v, list) else str(v).split(","))
                if str(x).strip()]
    # Bodies hidden from the start (objects the task's skill does not use).
    _hide = _names(mcfg.get("hide_bodies"))
    if _hide:
        _EXTRA["hide_bodies"] = _hide
        print(f"[task_config] hidden: {','.join(_hide)}")
    pl = dict(mcfg.get("orchestrator", {}) or {})
    _EXTRA["orchestrator"] = bool(pl.get("enabled", True))
    if pl.get("model"):
        _inject(rest, "--orchestrator_model", pl["model"])
    if pl.get("quiet_travel_m") is not None:
        _ORCHESTRATOR_EXTRA["quiet_travel_m"] = float(pl["quiet_travel_m"])
    subs = mcfg.get("subtasks") or {}
    # Orchestrator choices: the loaded `subtasks:` in registry order (else every registry skill).
    loaded = list(subs.keys())
    resolved = _skill_registry.enum_for_run(loaded=loaded)
    _ORCHESTRATOR_EXTRA["subtasks"] = ",".join(resolved)
    print(f"[task_config] orchestrator choices -> {resolved}")
    _ORCHESTRATOR_EXTRA["subtask_prompts"] = ";".join(
        f"{k}={v['prompt']}" for k, v in subs.items() if v.get("prompt"))
    init_sub = mcfg.get("initial_subtask") or next(iter(subs), None)
    # Agentic: empty prompt (hold for the orchestrator). Single-task: top-level or initial subtask prompt.
    if agentic:
        _inject(rest, "--prompt", "")
    else:
        _direct = (mcfg.get("prompt")
                   or (subs.get(init_sub, {}).get("prompt") if init_sub else "")
                   or "")
        _inject(rest, "--prompt", _direct)
        print(f"[task_config] single-task (agentic off): prompt={_direct!r}")
    print(f"[task_config] {mcfg.get('name', path)}: subtasks={list(subs)} initial={init_sub}")


def _free_ports(ports):
    """Best-effort (lsof): kill processes orphaned on the run's ports by a hard-killed prior run."""
    for p in ports:
        try:
            out = subprocess.run(["lsof", "-nP", f"-tiTCP:{p}"],
                                 capture_output=True, text=True, timeout=5).stdout.split()
        except Exception:  # noqa: BLE001 — lsof missing / no perms: skip
            continue
        for pid in out:
            try:
                os.kill(int(pid), signal.SIGKILL)
                print(f"[run_sim] freed stale process on port {p} (PID {pid})", flush=True)
            except Exception:  # noqa: BLE001
                pass


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if any(a in ("-h", "--help") for a in argv):
        print("task flags (in addition to the launcher flags below):\n"
              "  --task_config <yaml>     whole task from a config file (scene/start/models/\n"
              "                           per-subtask prompts/orchestrator)\n"
              "  --web [--web_port 8090]  also launch the web viewer on this task config +\n"
              "                           these ports, and open it in a browser (--no_open to skip)\n")
        # Stop before any side effect (run folder).
        return _base_main(["--help"])

    # Peel off task-level flags; allow_abbrev=False so a prefix never matches a longer flag.
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--task_config", default=None)
    pre.add_argument("--web", action="store_true",
                     help="Also launch the web viewer on this task config and open it in the "
                          "browser.")
    pre.add_argument("--web_port", type=int, default=8090, help="Web viewer HTTP port for --web.")
    pre.add_argument("--no_open", action="store_true",
                     help="With --web, start the web viewer but do NOT open a browser.")
    ns, rest = pre.parse_known_args(argv)

    if ns.task_config:
        _apply_task_config(ns.task_config, rest)
    # Stop on a placeholder model path before any process is started or any port freed.
    for _f in ("--checkpoint", "--config", "--policy_path"):
        if _flag(rest, _f, None) is not None:
            resolve_asset(_flag(rest, _f, None))

    os.environ["PYTHONUNBUFFERED"] = "1"       # stream subprocess logs live

    _orchestrator_on = _EXTRA["orchestrator"]
    if _orchestrator_on and "--orchestrator" not in rest:
        rest = ["--orchestrator"] + rest

    # Unique timestamped run dir unless --output_dir is pinned.
    _run_dir = _flag(rest, "--output_dir", None)
    if _run_dir is None:
        _run_name = _EXTRA["name"] or "run"
        _stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time() % 1) * 1e6):06d}"
        _run_dir = f"eval_closedloop_sim_multi_output/{_run_name}_{_stamp}"
        _i = 1
        while Path(_run_dir).exists():
            _run_dir = f"eval_closedloop_sim_multi_output/{_run_name}_{_stamp}_{_i}"
            _i += 1
        rest += ["--output_dir", _run_dir]

    # halo_serve overlay from the task YAML goes into this env, which gen_env() copies.
    from halo_serve.serve_env import serve_env as _serve_env
    os.environ.update(_serve_env(_EXTRA["halo_serve"], _run_dir))

    _mode = ("multi-task (agentic + orchestrator)" if _EXTRA["agentic"]
             else "single-task (fixed prompt)")
    print(f"[run_sim] mode          = {_mode}")
    print(f"[run_sim] generator      = {GEN_NODE}")
    print(f"[run_sim] orchestrator        = {'on' if _orchestrator_on else 'OFF'}")
    print(f"[run_sim] orchestrator choices= {_ORCHESTRATOR_EXTRA['subtasks'] or '(all registry skills)'}")
    print(f"[run_sim] run dir         = {_run_dir}")
    # Free only THIS run's ports so a parallel sim on shifted ports survives.
    pa, _ = build_parser().parse_known_args(rest)
    _mine = [pa.mission_port, pa.state_port, pa.plan_port, pa.trim_port, pa.state_pub_port,
             pa.obs_port, pa.genstatus_port, pa.orchestrator_plan_port, 5553]   # 5553 = g1/depth
    _reg = os.environ.get("HALO_COMM_REGISTRY", "")
    _mine.append(int(_reg.rsplit(":", 1)[1]) if ":" in _reg else 5560)
    _free_ports(tuple(sorted(set(_mine))))
    print(f"[run_sim] ports          = {sorted(set(_mine))}", flush=True)

    # --web: started here, the only layer that knows both the task config and the resolved ports.
    viewer = None
    if ns.web:
        viewer_cmd = [sys.executable, "-u", "-m", "halo_system.visualization.server",
                      "--port", str(ns.web_port),
                      "--live_port", str(pa.state_pub_port),      # g1/state
                      "--obs_port", str(pa.obs_port),
                      "--mission_port", str(pa.mission_port),
                      "--plan_port", str(pa.orchestrator_plan_port)]
        if ns.task_config:
            viewer_cmd += ["--task_config", ns.task_config]
        viewer_cmd += ["--live_run_dir", str(Path(_run_dir).resolve())]  # live plan overlay (live_plan.npz)
        url = f"http://127.0.0.1:{ns.web_port}/"
        print(f"[visualization] {' '.join(viewer_cmd)}", flush=True)
        viewer = subprocess.Popen(viewer_cmd, cwd=str(REPO_ROOT), env=make_pythonpath_env())
        if not ns.no_open:
            # Wait for the port to accept before opening, or the browser races the server.
            import socket as _s
            for _ in range(60):
                if viewer.poll() is not None:
                    print("[visualization] exited during startup — not opening a browser",
                          flush=True)
                    break
                with _s.socket() as _c:
                    _c.settimeout(0.3)
                    if _c.connect_ex(("127.0.0.1", ns.web_port)) == 0:
                        import webbrowser
                        webbrowser.open(url)
                        break
                time.sleep(0.25)
        print(f"[visualization] {url}", flush=True)
    try:
        return _base_main(rest)
    finally:
        if viewer is not None:
            terminate_process(viewer, "web viewer")


if __name__ == "__main__":
    raise SystemExit(main())
