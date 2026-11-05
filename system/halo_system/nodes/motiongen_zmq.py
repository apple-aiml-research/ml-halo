#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""motiongen_zmq.py — the motion GENERATOR node (System-1), sim and real robot.

Captures an ego image + the robot state, runs the diffusion motion model and streams 5 s motion
plans to a tracker (``g1/send_plan``), conditioned on the tracker's remaining motion
(``g1/get_trim``). One model (``--checkpoint`` + ``--config``) serves every subtask: the mission
orchestrator picks the subtask on ``g1/mission`` and the node switches the prompt. It republishes the
ego frame on ``g1/obs`` and per-plan execution feedback on ``g1/gen_status``. Without an orchestrator
it runs one fixed prompt for the whole run.

Modes:
  * ``--mujoco`` (default): image + state come together from the MuJoCo tracker's ``get_state``
    (its ego render).
  * ``--real``: this node owns the OAK camera (``--oak_camera_index``, default 0).
    State comes from the tracker's ``g1/state`` topic with a REQ fallback.

The mission launchers (``scripts/run_sim.py``, ``scripts/run_real.py``) start this node with the
right flags. By hand:

  # sim, against a running motiontrack_mujoco_zmq on this box
  python system/halo_system/nodes/motiongen_zmq.py --mujoco \\
      --config <gen_config.yaml> --checkpoint <ckpt.pt> --prompt "walk to man"

  # real robot, OAK camera; tracker + orchestrator resolved from the halo_comm registry
  python system/halo_system/nodes/motiongen_zmq.py --real --oak_camera_index 0 \\
      --config <gen_config.yaml> --checkpoint <ckpt.pt> --prompt "" --agentic
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# Repo root (parents[3]); put in-repo halo_system / halo_comm / halo_orchestrator on sys.path so the
# node runs by path without an install.
_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "orchestrator", _REPO / "system" / "comm" / "python",
           _REPO / "system" / "serve" / "python", _REPO / "system"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_OBS_QUALITY = 80             # g1/obs JPEG quality
_OBS_HZ = 10.0                # sim: keep g1/obs fresh at this rate between plans
_RTC_BETA = 5.0               # RTC guidance weight
# A plan is "quiet" for --robot_anchor_when_quiet at/below this joint activity and pelvis XY travel.
_QUIET_ANCHOR_ACTIVITY, _QUIET_ANCHOR_TRAVEL_M = 0.06, 0.3
# The hold path runs no inference: re-installing a robot-anchored reference every tick (10-22x/s)
# makes it chase the robot's own balance corrections (a shake while standing).
_HOLD_REFRESH_S = 0.5
# Hold plan length (2 s at 30 fps) / ease-back time to the startup posture.
_HOLD_FRAMES, _HOLD_RETURN_S = 60, 1.0
# An automatic hold after these skills FREEZES instead of easing back (the seated pose is kept).
_HOLD_RETURN_SKIP = {"sit"}
# Standing up from 'sit': lean the upper body forward (hips flexed by the same angle so the thighs
# stay on the seat), then ease to standing over _HOLD_RETURN_SIT_S.
_SIT_LEAN_DEG, _SIT_LEAN_S, _HOLD_RETURN_SIT_S = 45.0, 1.0, 2.0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Motion generator node (sim or real robot, multi-model, halo_comm)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--mujoco", dest="real", action="store_false",
                      help="(default) Sim: image + state from the MuJoCo tracker's get_state.")
    mode.add_argument("--real", dest="real", action="store_true",
                      help="Real robot: this node opens the OAK camera; state from the g1/state topic.")
    p.set_defaults(real=False)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Generator checkpoint (one model for every subtask).")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--prompt", type=str, required=True,
                   help="Initial prompt. Pass \"\" in --agentic mode (the node "
                        "holds standing until the orchestrator sends the first subtask + prompt).")
    p.add_argument("--warmup_prompts", type=str, default="",
                   help="'subtask=prompt;...' -- every generator prompt the orchestrator can send. A "
                        "model conditioned on every text token has one conditioning shape per "
                        "prompt length, so the warm-up runs each of these prompts.")
    p.add_argument("--state_port", type=int, default=5555)
    p.add_argument("--plan_port", type=int, default=5556)
    p.add_argument("--trim_port", type=int, default=5557)
    p.add_argument("--state_pub_port", type=int, default=5558,
                   help="Tracker's g1/state topic port. 0 disables the SUB (--real then "
                        "polls state over REQ on state_port).")
    p.add_argument("--oak_camera_index", type=int, default=0,
                   help="(--real) Luxonis OAK / DepthAI camera socket index (0 = CAM_A on OAK-1). "
                        "Requires `pip install depthai`.")
    p.add_argument("--gen_interval", type=float, default=0.0,
                   help="Seconds between motion generations (must be < 5s; 0 = back-to-back)")
    p.add_argument("--duration", type=float, default=0.0,
                   help="Total runtime in seconds (0 = run forever until Ctrl-C).")
    p.add_argument("--output_dir", type=str, default="generator_output",
                   help="Run dir: the generator overwrites live_plan.npz here (the viewer's LIVE "
                        "plan overlay).")
    p.add_argument("--robot_anchor_when_quiet", type=str, default="",
                   help="Comma list of subtasks that switch a plan to the ROBOT anchor once it is "
                        "quiet (plans after #1 continue from the reference). YAML: subtasks.<name>."
                        "rebase_robot_when_quiet: true.")
    p.add_argument("--obs_port", type=int, default=5559,
                   help="g1/obs PUB port (default 5559).")
    p.add_argument("--mission_port", type=int, default=5554,
                   help="g1/mission SUB port. MUST equal the orchestrator's --mission_port (default "
                        "5554; 5560 is the discovery registry).")
    p.add_argument("--mission_host", type=str, default="auto",
                   help="Host to SUBSCRIBE g1/mission on: the ORCHESTRATOR's host (it BINDS the port). "
                        "'auto' (default): the tracker host with --mujoco, the "
                        "registry with --real (the real tracker publishes no "
                        "g1/mission).")
    p.add_argument("--agentic", action="store_true",
                   help="Orchestrator-driven mode: HOLD a standing pose until the orchestrator sends a REAL "
                        "subtask on g1/mission, then swap model/prompt per mission.")
    p.add_argument("--no_mission", dest="mission_subscribe", action="store_false",
                   help="Disable the g1/mission subscription (fixed --prompt + --checkpoint for the "
                        "whole run). Default: subscribe with --agentic under --mujoco, always "
                        "under --real.")
    p.set_defaults(mission_subscribe=None)
    p.add_argument("--genstatus_port", type=int, default=5561,
                   help="g1/gen_status PUB port. MUST equal the orchestrator's --genstatus_port "
                        "(default 5561).")
    return p


_LIVE_PLAN_WRITER = None


def _write_async(fn, *args, **kwargs):
    """Run the live-plan file write off the observation -> plan path.

    One worker keeps writes ordered; main() drains it before exiting.
    """
    global _LIVE_PLAN_WRITER
    if _LIVE_PLAN_WRITER is None:
        from concurrent.futures import ThreadPoolExecutor
        _LIVE_PLAN_WRITER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="live-plan-writer")
    _LIVE_PLAN_WRITER.submit(fn, *args, **kwargs)


def _write_live_plan(path, plan, t0, idx):
    """Atomically overwrite <run>/live_plan.npz: the newest plan [T,36] (plan frame), its t0 row
    [stamp, pos(3), quat WXYZ(4)] and the plan counter. The viewer's LIVE plan overlay reads it."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:          # a file object: np.savez on a path would append .npz
        np.savez(fh, plan=plan, t0=t0, idx=np.int64(idx))
    os.replace(tmp, path)


def _run(argv=None) -> int:
    # Heavy imports, after the sys.path bootstrap.
    import threading
    from halo_system.lib.zmq_helper import TrackerClient, ObsPublisher
    from halo_system.lib.model_loader import load_generator_model
    from halo_system.lib.assets import resolve_asset
    from halo_system.lib.inference import (
        motion_plan_to_qpos, generate_motion_plan, tracker_qpos_to_model_chunk,
        joint_activity, sit_lean_pose, ramp_plan)

    args = build_parser().parse_args(argv)
    camera_mode = bool(args.real)                # real robot: this node owns the camera
    src = "oak" if camera_mode else "tracker"

    import torch
    import torchvision.transforms.functional as TF

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Tracker host: localhost (--mujoco) or the registry (--real).
    tracker_host = None if camera_mode else "127.0.0.1"

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Image source: {src}")
    print(f"[INFO] Prompt: \"{args.prompt}\"")
    print(f"[INFO] Gen interval: {args.gen_interval}s")
    print(f"[INFO] Tracker: {tracker_host or 'resolved from registry'} "
          f"(ports {args.state_port}/{args.plan_port}/{args.trim_port}/{args.state_pub_port})")
    print(f"[INFO] RTC: on for plan_counter > 0 (beta={_RTC_BETA})")

    if args.gen_interval >= 5.0:
        print("[ERROR] gen_interval must be < 5s (plan is only 5s)")
        return 1
    _quiet_anchor_subs = {x.strip() for x in (args.robot_anchor_when_quiet or "").split(",")
                          if x.strip()}
    _t_model_load = time.time()
    _ckpt_local = resolve_asset(str(args.checkpoint), required="generator checkpoint (--checkpoint)")
    motion_model, image_transform, config = load_generator_model(
        resolve_asset(str(args.config)), _ckpt_local, device)
    use_inpaint = getattr(motion_model, "inpaint_prev_chunk", False)
    prev_chunk_frames = int(getattr(motion_model, "prev_chunk_frames", 15))
    active_subtask = None
    print(f"[gen] loaded {args.checkpoint} (inpaint={use_inpaint} P={prev_chunk_frames}, "
          f"steps={getattr(config.model, 'num_inference_steps', '?')}, "
          f"cfg={Path(str(args.config)).name}) in {time.time() - _t_model_load:.1f}s", flush=True)
    # Warm the model with one throwaway plan per prompt (first predict is cold), using the same call
    # shape as the real loop on a 1280x800 frame. Runs before the tracker / camera wait.
    _t_w = time.time()
    _raw = np.zeros((800, 1280, 3), dtype=np.uint8)
    _q = np.zeros((1, 36))
    _q[0, 3] = 1.0                      # standing root, identity quat
    _all_prompts = []
    for _kv in (args.warmup_prompts or "").split(";"):
        _pr = _kv.split("=", 1)[-1].strip()
        if _pr and _pr not in _all_prompts:
            _all_prompts.append(_pr)
    _rgb = image_transform(TF.to_tensor(_raw)).unsqueeze(0).to(device)
    # Prefix length fed to the model: P frames (inpaint) or the full horizon (RTC guidance).
    prefix_len = prev_chunk_frames if use_inpaint else int(config.model.trajectory_length)
    _prev = tracker_qpos_to_model_chunk(np.repeat(_q, prefix_len, axis=0), prefix_len)
    # A NON-EMPTY prompt: agentic missions start with args.prompt == "".
    _prompts = [args.prompt or "walk to man"]
    _prompts += [_pr for _pr in _all_prompts if _pr not in _prompts]
    for _pr in _prompts:
        try:
            generate_motion_plan(motion_model, _rgb, _pr, device,
                                 prev_chunk=_prev, inference_delay=4, beta=_RTC_BETA)
        except Exception as e:                              # noqa: BLE001
            print(f"[warmup] {_pr!r} failed: {type(e).__name__}: {e}", flush=True)
    # Under halo_serve every warm-up predict is an engine job, so the cold first plans would
    # land in <run>/halo_serve_stats.json and skew its latency/miss numbers. Zero them here.
    if getattr(motion_model, "_halo_serve_attached", None):
        try:
            from halo_serve.runtime.engine import global_engine
            global_engine().reset_stats()
        except Exception as e:                                  # noqa: BLE001 -- telemetry only
            print(f"[warmup] serving stats reset failed: {type(e).__name__}: {e}", flush=True)
    print(f"[warmup] {len(_prompts)} prompt(s) warmed in {time.time() - _t_w:.1f}s", flush=True)
    # Now the co-served orchestrator model, with the motion model loaded and warm.
    if os.environ.get("HALO_SERVE_ENDPOINT"):
        try:
            from halo_serve.deploy import preload_orchestrator
            preload_orchestrator()
        except Exception as e:                                  # noqa: BLE001
            print(f"[warmup] orchestrator preload failed: {type(e).__name__}: {e}", flush=True)
    if use_inpaint:
        print(f"[INFO] Model uses prev_chunk inpainting (P={prev_chunk_frames}); "
              f"prefix fed on every plan, inference-time RTC guidance disabled.")
    print("[INFO] Model loaded")

    camera = None
    if camera_mode:
        from halo_system.lib.camera import OAKCameraStream
        # OAK default 1280x800 (16:10) = the sim's oak render and the training FPV aspect.
        camera = OAKCameraStream(args.oak_camera_index, 1280, 800)
        print(f"[INFO] Using OAK (DepthAI) camera, socket index {args.oak_camera_index}")
        camera.start()
        print("[INFO] Waiting for camera...")
        for _ in range(100):
            if camera.get_frame() is not None:
                break
            time.sleep(0.1)
        if camera.get_frame() is None:
            print("[ERROR] No camera frames received")
            camera.stop()
            return 1
        print("[INFO] Camera ready")
    # g1/obs is published from the main loop with --real; with the tracker source it rides each
    # get_state (+ a refresher).

    def _loop_frame():
        """The frame the main loop publishes on g1/obs (the camera's newest; None without one)."""
        return None if camera is None else camera.get_frame()

    tracker = TrackerClient(
        tracker_host, args.state_port, args.plan_port,
        args.trim_port, args.state_pub_port)

    # g1/state SUB callback: the primary state path for the camera
    # sources (GIL-safe single-ref assign).
    latest_state = {"v": None}
    use_sub = bool(args.state_pub_port and args.state_pub_port > 0)
    if use_sub:
        tracker.subscribe_state(
            on_message=lambda view: latest_state.__setitem__("v", view))
    print("[INFO] Connected to tracker")

    # g1/obs: viewers + orchestrator subscribe here instead of calling get_state (an extra render).
    # JPEG with --real (raw frames flood the robot's Wi-Fi).
    obs_pub = ObsPublisher(tracker.node, args.obs_port, jpeg=camera_mode, quality=_OBS_QUALITY)
    # The tracker REQ socket and the PUB socket are not thread-safe, and the generation thread,
    # the main loop and (sim) the obs refresher all use them, so both get a lock. `_obs_last` =
    # when g1/obs was last published.
    _obs_last = [0.0]
    _rpc_lock, _pub_lock = threading.Lock(), threading.Lock()
    _get_state_raw = tracker.get_state

    def _get_state_locked():
        with _rpc_lock:
            return _get_state_raw()
    tracker.get_state = _get_state_locked
    _publish_raw = obs_pub.publish

    def _publish_locked(*a, **k):
        with _pub_lock:
            _obs_last[0] = time.time()
            return _publish_raw(*a, **k)
    obs_pub.publish = _publish_locked

    # g1/gen_status: per-plan settle feedback for the orchestrator.
    from halo_comm import motion_pb2 as _mpb          # lazy
    gen_status_pub = tracker.node.create_publisher(
        _mpb.GenStatus, "g1/gen_status", args.genstatus_port)
    print(f"[gen_status] publishing g1/gen_status :{args.genstatus_port}", flush=True)

    # g1/mission: the orchestrator BINDS it on its own box, so subscribe on the orchestrator's host, which on
    # the robot is not the tracker host ('auto' -> registry).
    active_prompt = args.prompt          # initial default until the orchestrator speaks
    last_mission_tick = -1               # the orchestrator republishes each decision; dedup on tick
    got_real_subtask = False             # agentic: hold until a REAL (non-hold) subtask arrives
    holding = False                      # orchestrator sent a 'hold'/'stand' (or no task yet)
    mission_client = None
    _subscribe = (args.mission_subscribe if args.mission_subscribe is not None
                  else (camera_mode or args.agentic))
    if _subscribe:
        from halo_system.lib.zmq_helper import MissionClient
        if str(args.mission_host).lower() != "auto":
            _mhost = args.mission_host
        else:
            _mhost = tracker_host
        mission_client = MissionClient(node=tracker.node, host=_mhost,
                                       mission_port=args.mission_port)
        mission_client.subscribe_mission()
        print(f"[INFO] subscribed g1/mission {_mhost or 'auto'}:{args.mission_port} "
              f"(initial prompt: \"{active_prompt}\")")
    if args.agentic and mission_client is None:
        print("[mission] WARNING: --agentic but g1/mission is not subscribed -> the node will HOLD "
              "standing forever (safer than moving on its own initiative). Fix the subscription "
              "or drop --agentic.")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    live_plan_path = output_dir / "live_plan.npz"
    live_plan_path.unlink(missing_ok=True)       # never show a previous run's plan
    print(f"[INFO] Output dir: {output_dir}")

    plan_counter = 0
    generating = False
    gen_thread = None
    last_gen_time = -args.gen_interval
    last_gen_subtask = None   # subtask of the last generation; a switch generates at once
    start_time = time.time()
    camera_failed = threading.Event()
    rtc_inference_delay = 4
    # `hold` bookkeeping. The hold plan is derived from the TRACKER's own current reference
    # (get_trim -> freeze frame 0 -> pad), so the generator needs no standing pose of its own.
    holds_sent = 0
    hold_warned = False
    _last_hold_send = [0.0]   # wall time of the last hold install; see _send_standing_hold
    # Hold-return: `_startup_pose` = first reference frame (captured at tracker_ready);
    # `_hold_t0` = start of the current hold episode (the ramp is a function of wall time).
    _startup_pose = [None]
    # The robot's REAL standing pelvis height at startup (world z, from get_state). The trim
    # carries pelvis POSITION relative to the robot, so its z column cannot restore the height.
    _startup_world_z = [None]
    _hold_t0 = [0.0]
    _sit_lean = [None]   # (hold episode start, leaned-forward seated pose) for sit stand-up
    # Pose mode of the current hold: 'stand' = always ease back to the standing pose (skill
    # `stand`), 'freeze' = keep the current pose (an explicit operator `hold`), 'auto' = the
    # orchestrator's own automatic pause (freeze after _HOLD_RETURN_SKIP skills, else ease back).
    _hold_mode = ["auto"]
    _held = [False]      # the "holding until a task" notice was printed

    def _poll_state():
        """Freshest tracker state as RobotStateView (g1/state topic, else a REQ), or None."""
        if use_sub:
            v = latest_state["v"]
            if v is not None:
                return v
        try:
            return tracker.get_state()
        except Exception:  # noqa: BLE001 — timeout / transient
            return None

    def _capture_observation():
        """(rgb_tensor, state).

        Camera sources: the newest camera frame and state=None (the caller polls it). tracker:
        one get_state REQ -> the image rides the same RPC as the state, so they are from one
        frame."""
        if camera_mode:
            frame = camera.get_frame()
            if frame is None:
                return None, None
            return image_transform(TF.to_tensor(frame)).unsqueeze(0).to(device), None
        try:
            fresh = tracker.get_state()
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] get_state() failed: {exc}")
            return None, None
        if fresh.image is not None:
            raw = fresh.image
            obs_pub.publish(raw, tick=fresh.tick)   # forward the held frame on g1/obs
            rgb_tensor = image_transform(TF.to_tensor(raw)).unsqueeze(0).to(device)
            return rgb_tensor, fresh
        return None, fresh

    def _q_yaw(q):
        """Yaw (rad) of an XYZW quaternion — the plan/qpos order send_plan uses."""
        x, y, z, w = (float(v) for v in q)
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _q_z(yaw):
        """Pure-Z XYZW quaternion."""
        return np.array([0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)])

    def _q_mul(a, b):
        """Hamilton product of two XYZW quaternions."""
        x1, y1, z1, w1 = (float(v) for v in a)
        x2, y2, z2, w2 = (float(v) for v in b)
        return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                         w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])

    def _upright_like(q_now, q_start):
        """CURRENT yaw + STARTUP pitch/roll.

        A hold must not TURN the robot (that would be locomotion), but it must un-LEAN it: pitch
        and roll are posture, and a robot that ends a crawl pitched forward should not stand back
        up still pitched forward.
        """
        resid = _q_mul(_q_z(-_q_yaw(q_start)), np.asarray(q_start, float))   # startup, yaw removed
        out = _q_mul(_q_z(_q_yaw(q_now)), resid)                             # + the current yaw
        n = float(np.linalg.norm(out))
        return out / n if n > 1e-9 else np.asarray(q_now, float)

    def _send_standing_hold():
        """Install a robot-anchored HOLD plan.

        Frame 0 of ``get_trim()`` (the current reference, plan order) is frozen and padded to
        _HOLD_FRAMES, or eased back to the startup posture (_HOLD_RETURN_S). Holds are counted in
        ``holds_sent``, not ``plan_counter``.
        """
        nonlocal holds_sent, hold_warned
        # Rate-limited by _HOLD_REFRESH_S (see its comment).
        now = time.time()
        if holds_sent and (now - _last_hold_send[0]) < _HOLD_REFRESH_S:
            return
        _last_hold_send[0] = now
        try:
            trim = tracker.get_trim()
        except Exception as exc:  # noqa: BLE001 — a hold must never kill the loop
            if not hold_warned:
                hold_warned = True
                print(f"[mission] WARNING: hold get_trim() failed ({type(exc).__name__}: {exc}); "
                      f"the tracker will hold its last reference frame on its own", flush=True)
            return
        ref = np.asarray(trim.plan, dtype=np.float32)
        if len(ref) == 0:
            # Nothing to freeze; the tracker's own end-of-motion clamp already holds the last frame.
            if not hold_warned:
                hold_warned = True
                print("[mission] hold: tracker reported no remaining reference; it holds its "
                      "current pose on its own", flush=True)
            return
        in_fps = trim.input_fps or 30
        N = _HOLD_FRAMES
        ref0 = ref[0].astype(np.float64)
        tgt, R = _startup_pose[0], _HOLD_RETURN_S
        if active_subtask == "sit":
            R = _HOLD_RETURN_SIT_S                  # stand up from the chair more slowly
        # FREEZE after a posture skill: see _HOLD_RETURN_SKIP. active_subtask still names the last
        # real skill while holding.
        if _hold_mode[0] == "freeze" or (_hold_mode[0] == "auto"
                                         and active_subtask in _HOLD_RETURN_SKIP):
            tgt = None
        if tgt is None:
            plan = np.tile(ref0[None, :], (N, 1))       # freeze where the robot is
        else:
            # Ease back: joints, pitch/roll and pelvis height only; x/y and yaw stay current.
            T = ref0.copy()
            # Raise the (relative) height reference by the real pelvis deficit; re-measured each
            # refresh so it shrinks as the robot rises.
            _cur_z = None
            try:
                _cur_z = float(np.asarray((_poll_state() or tracker.get_state()).base_pos)[2])
            except Exception:  # noqa: BLE001 -- no height -> joints/lean only
                _cur_z = None
            _crouched = float(np.abs(ref0[7:36] - tgt[7:36]).max()) > 0.3
            if _startup_world_z[0] is not None and _cur_z is not None:
                if camera_mode:
                    # Real base z is an estimate: raise only (a downward step would accumulate
                    # under drift), only when the joints say crouched, capped at 0.45 m.
                    if _crouched:
                        T[2] = ref0[2] + float(np.clip(_startup_world_z[0] - _cur_z, 0.0, 0.45))
                else:
                    T[2] = ref0[2] + (_startup_world_z[0] - _cur_z)
            T[7:36] = tgt[7:36]                         # joints -> the startup posture
            T[3:7] = _upright_like(ref0[3:7], tgt[3:7])  # un-LEAN (pitch/roll) but keep the yaw
            # The first hold episode is not armed by the mission branch.
            if _hold_t0[0] == 0.0:
                _hold_t0[0] = now
            _sit_up = active_subtask == "sit"
            knots = []
            if _sit_up:
                # Stand up from the chair: lean forward, then ease to standing over R. The lean
                # pose is built once per hold episode.
                if _sit_lean[0] is None or _sit_lean[0][0] != _hold_t0[0]:
                    _sit_lean[0] = (_hold_t0[0], sit_lean_pose(ref0, _SIT_LEAN_DEG))
                knots.append((_hold_t0[0] + _SIT_LEAN_S, _sit_lean[0][1]))
            # Frame 0 = current reference (no step); close the gap over the remaining ramp time.
            knots.append((_hold_t0[0] + (_SIT_LEAN_S if _sit_up else 0.0) + R, T))
            plan = ramp_plan(ref0, knots, now, N, in_fps)
        plan = plan.astype(np.float32)
        try:
            success = tracker.send_plan(plan, input_fps=in_fps, align_yaw=False,
                                              auto_start_offset=False, align_anchor="robot")
        except Exception as exc:  # noqa: BLE001
            if not hold_warned:
                hold_warned = True
                print(f"[mission] WARNING: hold send_plan failed ({type(exc).__name__}: {exc}); "
                      f"the tracker will hold its last reference frame", flush=True)
            return
        holds_sent += 1
        if not success and not hold_warned:
            hold_warned = True
            print("[mission] WARNING: hold plan rejected by the tracker; it will hold its last "
                  "reference frame", flush=True)

    def _consume_mission():
        """Pick up the freshest mission decision (if any) right before generating a plan. The
        orchestrator publishes EVERY decision (per-decision tick), so dedup on the tick, then swap
        the prompt only on an actual subtask/prompt change. A 'hold'/'stand'/'' subtask
        -> stand."""
        nonlocal active_prompt, last_mission_tick, holding, got_real_subtask, active_subtask
        if mission_client is not None:
            cmd = mission_client.recv_mission(timeout_ms=0)
            if cmd is not None and int(cmd.tick) != last_mission_tick:
                last_mission_tick = int(cmd.tick)
                if cmd.subtask in ("hold", "stand", ""):
                    _rsn = cmd.reason
                    _prev_mode = _hold_mode[0]
                    _hold_mode[0] = ("stand" if cmd.subtask == "stand"
                                     else "freeze" if _rsn.startswith("operator hold") else "auto")
                    if holding and _hold_mode[0] != _prev_mode:
                        _last_hold_send[0] = 0.0     # hold <-> stand while holding: re-send at once
                        _hold_t0[0] = time.time()    # and restart the ease-back ramp from here
                    if not holding:
                        _last_hold_send[0] = 0.0     # entering hold -> install at once
                        _hold_t0[0] = time.time()    # and restart the return-to-startup ramp
                        _hr = _rsn.strip()
                        print("[mission] HOLD — standing"
                              + (f" ({_hr})" if _hr
                                 else " (agent needs input, or no task yet)"),
                              flush=True)
                    holding = True
                else:
                    got_real_subtask = True        # a genuine task -> we may act
                    holding = False
                    if cmd.subtask and cmd.subtask != active_subtask:
                        active_subtask = cmd.subtask
                        if cmd.prompt:
                            active_prompt = cmd.prompt
                        print(f"[mission] subtask {cmd.subtask_id} ({cmd.subtask}) "
                              f"done={cmd.done} -> prompt: \"{active_prompt}\"")
                    elif cmd.prompt and cmd.prompt != active_prompt:
                        active_prompt = cmd.prompt
    def _hold_tick():
        """Agentic: send an explicit hold until the orchestrator sends a real subtask (no model runs)."""
        if not _held[0]:
            print("[mission] holding standing pose until the orchestrator sends a task…", flush=True)
            _held[0] = True
        _send_standing_hold()

    def _generate_and_send():
        nonlocal plan_counter, generating, rtc_inference_delay
        _consume_mission()
        try:
            t0 = time.time()

            if camera_mode:
                # g1/obs comes from the main loop. Holds even if the g1/mission subscribe failed.
                if args.agentic and (not got_real_subtask or holding):
                    _hold_tick()
                    return
                rgb_tensor, fresh_state = _capture_observation()
                if rgb_tensor is None:
                    print("[ERROR] No camera frame available — exiting. Check "
                          "--oak_camera_index / the OAK connection.")
                    camera_failed.set()
                    return
            else:
                # The capture publishes g1/obs, so the orchestrator sees frames while we hold.
                rgb_tensor, fresh_state = _capture_observation()
                if rgb_tensor is None or fresh_state is None:
                    print("[WARN] No image or state available; skipping plan.")
                    return
                if args.agentic and (not got_real_subtask or holding):
                    _hold_tick()
                    return
            t_obs = time.time()

            if fresh_state is None:
                fresh_state = _poll_state()
            if fresh_state is None or np.asarray(fresh_state.joint_pos).size == 0:
                print("[WARN] No state available yet (tracker state has no joints); "
                      "skipping plan.")
                return
            t0_pos = np.asarray(fresh_state.base_pos, dtype=np.float64).copy()
            t0_quat_wxyz = np.asarray(fresh_state.imu_quat_wxyz, dtype=np.float64).copy()
            t_state = t0_time = time.time()

            # Prefix from the tracker's remaining motion: inpaint models on every plan (first P
            # frames); RTC-guidance models from plan #2 (full horizon).
            rtc_prev = None
            build_prefix = use_inpaint or plan_counter > 0
            if build_prefix:
                trim = tracker.get_trim()
                tracker_qpos = trim.plan
                if len(tracker_qpos) > 0:
                    rtc_prev = tracker_qpos_to_model_chunk(tracker_qpos, prefix_len)
                    print(f"[INFO] {'inpaint' if use_inpaint else 'RTC'}: using "
                          f"{len(tracker_qpos)} trimmed frames as prev_chunk (P={prefix_len})")

            joints, rel_pos, rel_quat = generate_motion_plan(
                motion_model, rgb_tensor, active_prompt, device,
                prev_chunk=rtc_prev, inference_delay=rtc_inference_delay, beta=_RTC_BETA)
            t_infer = time.time()
            rtc_inference_delay = max(1, int((t_infer - t_state) * 30))

            qpos_data = motion_plan_to_qpos(joints, rel_pos, rel_quat)
            plan_counter += 1

            # joint_activity = mean frame-to-frame ‖Δjoints‖.
            _activity = joint_activity(qpos_data)
            # Planned pelvis XY travel: a slow sidestep has low joint activity but still moves the
            # robot, so the orchestrator must not call it quiet.
            _travel = float(np.linalg.norm(qpos_data[-1, :2] - qpos_data[0, :2]))
            # One status line per plan (subtask, checkpoint, prompt in force, moving/holding).
            if holding:
                _what = "HOLDING (orchestrator: hold/stand or no task yet)"
            elif _activity <= _QUIET_ANCHOR_ACTIVITY:
                _what = f"quiet (activity {_activity:.4f} <= settle {_QUIET_ANCHOR_ACTIVITY:g})"
            else:
                _what = f"MOVING (base_travel {_travel:.2f} m)"
            print(f"[gen] plan#{plan_counter}  subtask={active_subtask or '(none)'}"
                  f"  prompt=\"{active_prompt}\"  {_what}", flush=True)
            try:
                gen_status_pub.publish(_mpb.GenStatus(
                    joint_activity=float(_activity), subtask=str(active_subtask or ""),
                    plan_travel=_travel))
            except Exception:  # noqa: BLE001 — best-effort feedback
                pass

            # Plan #1 is robot-anchored; later plans continue from the reference.
            plan_anchor = "robot" if plan_counter == 1 else "reference"
            # --robot_anchor_when_quiet: re-sync a quiet plan to where the robot is (e.g. seated).
            if plan_anchor == "reference" and active_subtask in _quiet_anchor_subs:
                if _activity <= _QUIET_ANCHOR_ACTIVITY and _travel <= _QUIET_ANCHOR_TRAVEL_M:
                    plan_anchor = "robot"
                    print(f"[anchor] plan#{plan_counter} '{active_subtask}' quiet (activity "
                          f"{_activity:.3f}, travel {_travel:.2f} m) -> robot anchor", flush=True)

            # Newest plan for the viewer's LIVE plan overlay (one overwritten file, off the hot path).
            _write_async(_write_live_plan, live_plan_path, np.array(qpos_data, copy=True),
                            np.concatenate([[t0_time], t0_pos, t0_quat_wxyz]), plan_counter)

            # Yaw alignment is server-side; continuation chunks self-slice via auto_start_offset.
            # Plan #1 installs from frame 0 (slicing would skip into the motion).
            auto_start_offset = rtc_prev is not None and plan_counter > 1
            try:
                success = tracker.send_plan(
                    qpos_data, input_fps=30,
                    align_yaw=True, auto_start_offset=auto_start_offset,
                    align_anchor=plan_anchor)
                if success:
                    print(f"[INFO] Plan #{plan_counter} sent ({len(qpos_data)} frames, "
                          f"{len(qpos_data) / 30:.1f}s, auto_start_offset={auto_start_offset}, "
                          f"anchor={plan_anchor}) | "
                          f"obs={t_obs - t0:.2f}s state={t_state - t_obs:.2f}s "
                          f"infer={t_infer - t_state:.2f}s total={time.time() - t0:.2f}s")
                else:
                    print(f"[WARN] Plan #{plan_counter} rejected by tracker")
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Plan #{plan_counter} send failed: {exc}")
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] Generation failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            generating = False

    def _cleanup():
        if camera is not None:
            camera.stop()
        obs_pub.close()
        if gen_thread is not None and gen_thread.is_alive():
            gen_thread.join(timeout=6.0)
        try:
            tracker.close()
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] tracker.close() raised: {exc}")

    # Wait for tracker_ready. The sim sets it at startup, so this typically passes on the first
    # poll; the real tracker sets it once the operator presses B on the robot remote.
    print("\n[INFO] Waiting for tracker_ready signal...")
    if camera_mode:
        print("[INFO] On the robot remote: Start → default pose, A → stand up, "
              "then press B to start.")
    print()
    wait_start = time.time()
    while True:
        if args.duration > 0 and (time.time() - start_time) > args.duration:
            print("[INFO] Duration expired before tracker ready. Exiting.")
            _cleanup()
            return 0
        try:
            state = tracker.get_state()
            if state is not None and state.tracker_ready:
                break
        except Exception:  # noqa: BLE001 — transient
            pass
        elapsed = time.time() - wait_start
        if int(elapsed) % 10 == 0 and int(elapsed) > 0:
            print(f"[INFO] Still waiting for tracker_ready"
                  + (" — press B to start" if camera_mode else "")
                  + f"... ({elapsed:.0f}s)", flush=True)
        time.sleep(0.1)

    print("[INFO] Tracker ready! Starting generation loop. Press Ctrl+C to stop.\n")
    # Startup pose for hold-return, sampled before anything moves: frame 0 of the first trim.
    try:
        _t0 = np.asarray(tracker.get_trim().plan, dtype=np.float64)
        if len(_t0):
            _startup_pose[0] = _t0[0].copy()
            try:
                _bp0 = np.asarray(tracker.get_state().base_pos, dtype=np.float64)
                _z0 = float(_bp0[2])
                # Real robot without odometry reports a fixed base_pos (0, 0, z): no height
                # restore is possible.
                no_odom = camera_mode and _bp0.size >= 2 and _bp0[0] == 0.0 and _bp0[1] == 0.0
                if no_odom:
                    print(f"[mission] no odometry (base_pos is the tracker's fixed fallback "
                          f"(0, 0, {_z0:.3f})) -> hold restores joints/lean but NOT pelvis "
                          f"height", flush=True)
                # Trust only a plausible standing height (the sim's is always taken).
                elif not camera_mode or 0.4 <= _z0 <= 1.2:
                    _startup_world_z[0] = _z0
                    print(f"[mission] startup standing height (world) = {_z0:.3f} m",
                          flush=True)
                else:
                    print(f"[mission] startup base z {_z0:.3f} m is not a standing height -> "
                          f"hold will restore joints/lean but not pelvis height", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[mission] WARNING: no startup world height ({type(exc).__name__}); "
                      f"hold will restore joints/lean but not pelvis height", flush=True)
            print(f"[mission] startup pose captured for hold-return: pelvis z="
                  f"{_startup_pose[0][2]:.3f} m (a hold will ease back to this over "
                  f"{_HOLD_RETURN_S:.1f}s)", flush=True)
        else:
            print("[mission] hold-return DISABLED: tracker had no reference to sample at "
                  "startup; holds will freeze the current pose instead", flush=True)
    except Exception as exc:  # noqa: BLE001 — never block the run on this
        print(f"[mission] hold-return DISABLED: could not sample the startup pose "
              f"({type(exc).__name__}: {exc}); holds will freeze the current pose",
              flush=True)
    # tracker source: get_state (which carries the image) runs once per plan; refresh between.
    if src == "tracker":
        def _obs_refresher():
            period = 1.0 / _OBS_HZ
            while True:
                time.sleep(min(0.02, period / 4))
                if time.time() - _obs_last[0] < period:      # a plan's capture just published one
                    continue
                try:
                    st = tracker.get_state()
                except Exception:  # noqa: BLE001 -- tracker busy/gone: back off
                    time.sleep(0.5)
                    continue
                if st.image is not None:
                    try:
                        obs_pub.publish(st.image, tick=st.tick)
                    except Exception:  # noqa: BLE001 -- socket closed at shutdown
                        return
                else:
                    _obs_last[0] = time.time()
        threading.Thread(target=_obs_refresher, daemon=True, name="obs_refresh").start()
        print(f"[obs] g1/obs kept fresh at {_OBS_HZ:g} Hz between plans", flush=True)
    obs_period = 1.0 / _OBS_HZ
    try:
        while args.duration == 0 or (time.time() - start_time) < args.duration:
            now = time.time()
            if camera_failed.is_set():
                print("[INFO] Stopping generator loop due to camera failure.")
                break

            # g1/obs for the camera source, decoupled from the plan cadence.
            if camera_mode and (now - _obs_last[0]) >= obs_period:
                frame = _loop_frame()
                if frame is not None:
                    obs_pub.publish(frame)

            # A subtask switch generates at once.
            if not generating and (active_subtask != last_gen_subtask or
                                   (now - last_gen_time) >= args.gen_interval):
                generating = True
                last_gen_time = now
                last_gen_subtask = active_subtask
                gen_thread = threading.Thread(target=_generate_and_send, daemon=True)
                gen_thread.start()

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")
    finally:
        print("[INFO] Cleaning up...")
        _cleanup()
        print(f"[INFO] Generated {plan_counter} plans. Done.")
    return 0


def main(argv=None) -> int:
    try:
        rc = _run(argv)
    finally:
        # Drain the queued live-plan write: os._exit below skips atexit handlers.
        if _LIVE_PLAN_WRITER is not None:
            _LIVE_PLAN_WRITER.shutdown(wait=True)
    sys.stdout.flush()
    sys.stderr.flush()
    # os._exit: the g1/state SUB daemon thread otherwise wedges normal interpreter exit in the zmq
    # Context.term().
    os._exit(rc or 0)


if __name__ == "__main__":
    raise SystemExit(main())
