#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""VLM mission-orchestrator node: reads the ego image (g1/obs), decides the next subtask, publishes g1/mission.

Run (sim): PYTHONPATH=system/comm/python:system/serve/python:system:orchestrator:. python system/halo_system/nodes/orchestrator_zmq.py --instruction "..."
"""
from __future__ import annotations

import argparse
import binascii
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from halo_system.lib.zmq_helper import (MissionClient, OrchestratorActivityPublisher,
                                        RobotStateView)
from halo_orchestrator import VLMInterface, MissionOrchestrator
from halo_orchestrator.plan import unauthored
from halo_orchestrator import registry as _registry
from halo_orchestrator.deliberation import PlanRef, StaticDeliberator
from halo_orchestrator.history import MissionHistory
from halo_orchestrator.context import FrameRing
from halo_comm import state_pb2
from halo_comm import motion_pb2

# Skills whose successful end state is not standing (exempt from the standing guard).
ENDS_LOW_SKILLS = frozenset({"sit"})

# Skills that may succeed with no base motion (exempt from the must-have-moved guard).
NEVER_MOVES_SKILLS = frozenset({"hold", "stand"})
# Agentic step switching.
COMPLETE_AFTER_S = 8.0         # accept a complete_step blocked only by "still moving" after this long
MIN_STEP_FAIL_S = 30.0         # the retry cap may fail a step only after it has run this long
MAX_RETRIES = 10               # per-step retry_step cap, then fail+advance
INSERT_COOLDOWN_S = 4.0        # min seconds between insert_step interjections
SWITCH_ACTIVITY_THRESH = 0.06  # joint activity (rad) at/below which a done step switches
HOLD_ACTIVITY_THRESH = 0.06    # a 'not done' holds only at/below this joint activity (rad)
# Walk-like skills that switch the moment the agent says done (trailing * = prefix). sidewalk is
# not one: its done must wait for quiet joints, or a doorway view closes it before the door.
DIRECT_SWITCH_SKILLS = ("go_to*", "follow*")
SWITCH_HOLD_S = 1.0            # pre-switch standing hold (= the generator's hold ease-back)
SWITCH_STEADY_SPEED = 0.3      # peak joint speed (rad/s) that counts as steady in that hold
SWITCH_HOLD_MAX_S = 3.0        # longest the pre-switch hold waits for steadiness
VLM_TIMEOUT_LONG_S = 180.0     # T1 plan-authoring budget (cold model + guided JSON is slow)
GATE_TIMEOUT_S = 240.0         # deliberation gate releases after this (above VLM_TIMEOUT_LONG_S)


def _mint_run_id() -> str:
    """`<UTC compact>-<6 hex>` run id, minted once per orchestrator process."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{binascii.hexlify(os.urandom(3)).decode('ascii')}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VLM mission-orchestrator node (halo_comm)")
    p.add_argument("--instruction", dest="mission", metavar="TEXT", type=str, default="",
                   help="operator instruction the orchestrator decomposes into subtasks. EMPTY (default) "
                        "=> the orchestrator WAITS: it holds the robot and sends NO task until an operator "
                        "types an instruction in the web viewer input box.")
    p.add_argument("--model", type=str, default=None,
                   help="Local orchestrator model, a repo id or path, served in-process through "
                        "halo_serve (default mlx-community/Qwen3.6-35B-A3B-4bit).")
    p.add_argument("--host", type=str, default="127.0.0.1",
                   help="host publishing g1/obs + g1/gen_status (i.e. the GENERATOR's "
                        "host). 'auto' resolves it from the halo_comm registry, so a "
                        "split-host run needs only HALO_COMM_REGISTRY.")
    p.add_argument("--obs_port", type=int, default=5559)
    p.add_argument("--mission_port", type=int, default=5554)
    # Orchestrator plan topic (orchestrator -> viewer).
    p.add_argument("--plan_port", type=int, default=5564,
                   help="g1/plan PUB port — the checkpoint checklist ON THE WIRE. This is the "
                        "live viewer's only plan source.")
    p.add_argument("--subtasks", type=str, default=None,
                   help="Comma list restricting the orchestrator's choices to the loaded subtasks, "
                        "e.g. 'crawl_under,sit'. Default: all registry skills.")
    p.add_argument("--subtask_prompts", type=str, default=None,
                   help="Per-subtask generator-prompt overrides 'sub=prompt;sub=prompt' "
                        "(';' separator — prompts contain spaces/commas).")
    p.add_argument("--state_host", type=str, default=None,
                   help="Host of g1/state (the TRACKER). Default: --host. On the real robot the tracker "
                        "runs on the NX, not on the generator box, so run_real passes the NX IP.")
    p.add_argument("--genstatus_port", type=int, default=5561,
                   help="g1/gen_status topic port — the generator publishes the latest plan's "
                        "joint activity, subtask and planned travel; the orchestrator uses these "
                        "to tell whether the motion has settled.")
    p.add_argument("--quiet_travel_m", type=float, default=0.3,
                   help="A plan only counts as QUIET (switch / hold) if its planned "
                        "pelvis XY travel is also below this (m). A slow sidestep has joint activity "
                        "under 0.06 while still moving ~0.9 m (operator, 2026-10-09). 0 = joints only.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    print(f"[mission] config: mission={args.mission!r}", flush=True)

    subtasks = ([s.strip() for s in args.subtasks.split(",") if s.strip()]
                if args.subtasks else None)
    prompt_map = {}
    if args.subtask_prompts:
        for pair in args.subtask_prompts.split(";"):
            k, _, v = pair.partition("=")
            if k.strip() and v.strip():
                prompt_map[k.strip()] = v.strip()

    # Registry and run enum are resolved once at startup, never per tick.
    _reg = _registry.load()
    run_enum = _registry.enum_for_run(loaded=subtasks, reg=_reg)
    print(f"[mission] skill enum for this run: {run_enum}", flush=True)

    def _prompt_for(name: str) -> str:
        """Generator prompt for `name` from the registry, else the mission text."""
        return _registry.prompt_of(_reg, name) if name in _registry.names(_reg) else args.mission

    vlm = VLMInterface(model=args.model, timeout_s_long=VLM_TIMEOUT_LONG_S)
    orch = MissionOrchestrator(vlm, mission=args.mission, subtasks=run_enum)
    orch.step_prompts = dict(prompt_map)   # step titles = the generator prompt (see _title)
    print(f"[mission] orchestrator model={vlm.model} (local, halo_serve)", flush=True)
    print(f"[mission] mission=\"{args.mission}\"  choices={orch.subtasks}  "
          f"obs:{args.host}:{args.obs_port} -> mission:{args.host}:{args.mission_port}", flush=True)

    mc = MissionClient(host=args.host, obs_port=args.obs_port, mission_port=args.mission_port)
    # Bind + announce g1/mission now: the generator resolves it before publishing, so a lazy
    # bind would deadlock a split-host run.
    mc.ensure_mission_pub()
    print(f"[mission] g1/mission bound on :{args.mission_port} (announced to the registry)",
          flush=True)
    mc.subscribe_obs()
    # The g1/plan topic shares the MissionClient node.
    activity = OrchestratorActivityPublisher(node=mc.node, plan_port=args.plan_port)
    print(f"[mission] plan topic: g1/plan :{args.plan_port}", flush=True)
    # g1/state: robot pose for the prompt (freshest only).
    state_sub = mc.node.create_subscription(
        state_pb2.RobotState, "g1/state", args.state_host or args.host, 5558, conflate=True)
    # g1/gen_status: latest plan's joint activity, subtask and planned travel (freshest only).
    # Registry-resolved like g1/obs (--host auto).
    _gs_h, _gs_p = mc.resolve_host("g1/gen_status", args.genstatus_port)
    genstatus_sub = mc.node.create_subscription(
        motion_pb2.GenStatus, "g1/gen_status", _gs_h, _gs_p, conflate=True)

    _plan_travel = [0.0]     # planned pelvis XY travel (m) of the latest plan (GenStatus.plan_travel)

    def _is_quiet(ja, thr):
        """Quiet = joints still (<= thr) and planned XY travel <= --quiet_travel_m."""
        if ja is None or ja > thr:
            return False
        return not (args.quiet_travel_m > 0 and _plan_travel[0] > args.quiet_travel_m)

    def _gen_status():
        """Latest generator (joint_activity, gsubtask); both None until one arrives."""
        m = genstatus_sub.recv(0)
        if m is None:
            return None, None
        # No staleness check: the socket returns the last message forever, so a dead generator
        # would read as settled. Assumes the generator stays alive.
        _plan_travel[0] = float(m.plan_travel)
        return float(m.joint_activity), str(m.subtask)

    _last_state = [None]             # latest g1/state message (shared readers)
    _gp_bit = 10                     # Unitree remote X (g1/state.remote_buttons bit) = hold/resume
    _gp = {"prev": 0, "hold": False}    # gamepad hold toggle state

    def _state_msg():
        """Latest g1/state; the socket hands each message out once, so all readers share this cache."""
        m = state_sub.recv(0)
        if m is not None:
            _last_state[0] = m
        return _last_state[0]

    def _pose_text():
        m = _state_msg()
        if m is None:
            return None, None, None
        sv = RobotStateView.from_proto(m)
        w, x, y, z = [float(v) for v in sv.imu_quat_wxyz]
        yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        bh = float(sv.base_pos[2])
        xy = (float(sv.base_pos[0]), float(sv.base_pos[1]))   # world XY, for the stall/progress readout
        return f"base_height={bh:.2f}m, yaw={yaw:.0f}deg", bh, xy

    def _tilt_deg():
        """Torso tilt from vertical (deg) from the latest g1/state IMU quat; None without state."""
        m = _state_msg()
        if m is None:
            return None
        _, x, y, _ = [float(v) for v in RobotStateView.from_proto(m).imu_quat_wxyz]
        return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))

    def _seated_ok(bh):
        """Seated, not fallen: pelvis in [0.30, 0.60) m and torso within 40 deg of upright."""
        if bh is None:
            return True
        tl = _tilt_deg()
        return 0.30 <= bh < 0.60 and (tl is None or tl <= 40.0)

    def _direct_switch(skill):
        """Walk-like skills (go_to*, follow*) hand over to the next step at once when the robot is
        standing; sidewalk and contact/posture skills (crawl, crouch, sit, push) keep the quiet + hold switch."""
        sk = str(skill or "")
        return any(sk.startswith(p[:-1]) if p.endswith("*") else sk == p for p in DIRECT_SWITCH_SKILLS)

    _jsp = {"q": None, "t": 0.0, "speed": None}

    def _joint_speed():
        """Peak joint speed (rad/s) between two g1/state samples >= 50 ms apart; None without state."""
        m = _state_msg()
        if m is None or not len(m.joint_pos):
            return _jsp["speed"]
        t = time.time()
        if _jsp["q"] is None or t - _jsp["t"] >= 0.05:
            q = [float(v) for v in m.joint_pos]
            if _jsp["q"] is not None and len(q) == len(_jsp["q"]):
                _jsp["speed"] = max(abs(a - b) for a, b in zip(q, _jsp["q"])) / (t - _jsp["t"])
            _jsp.update(q=q, t=t)
        return _jsp["speed"]

    hold_period = 0.1  # hold branches sleep this long so they never spin

    # One run_id per orchestrator process (stamped into every plan).
    run_id = _mint_run_id()
    pub_tick = 0                   # per-decision id for published missions (flowing stream)
    _last_say = [""]   # latest step-decision line; rides g1/mission `why` to the viewer chat
    awaiting = False               # agent asked (need_input) -> hold until the operator replies
    # With no mission the orchestrator holds and waits for the operator's first instruction.
    wait_for_start = not (args.mission and args.mission.strip())
    if wait_for_start:
        print("[mission] no mission set — HOLDING; waiting for the operator's first instruction "
              "in the web viewer input box…", flush=True)
    pending_plan = False           # set when a new mission arrives -> build the checklist next tick
    # A preset (headless) mission bootstraps T1 plan authoring on the first frame.
    if not wait_for_start:
        pending_plan = True
        print("[mission] preset mission -> bootstrapping T1 plan authoring on the first frame",
              flush=True)
    # Deliberation epoch, bumped on every operator request; a T1 result built under an older
    # epoch is discarded (an in-flight call cannot be cancelled).
    delib_epoch = 0
    # Last plan version published on g1/plan; the tick loop publishes a newly installed plan at once.
    last_published_plan_version = -1
    delib_inflight = False         # a request is queued or running for the current epoch
    # Deliberation gate: open since `gate_since`; releases when a plan newer than
    # `gate_from_version` arrives.
    gate_since = None
    gate_from_version = 0
    # Mission-complete latch: robot holds and no VLM is called until a new operator mission.
    mission_complete = False
    _seq_prev = [None]
    # Agentic session state: per-step retry counters, insert cooldown.
    _retry = {}                    # step-id -> consecutive retry_step count (agentic retry cap)
    _last_insert = [0.0]           # wall-clock of the last insert_step (agentic cooldown)
    # Stall tracker (agentic): settles since the robot last moved >~0.3 m in world XY; a neutral
    # readout for the agent, not a gate.
    _stall = {"step": None, "count": 0, "ref_xy": None}
    _STALL_PROGRESS_M = 0.30       # XY move that RESETS the stall (real progress toward the target)
    # Per-step decision state: `seen` = generator first ran this step; `want` = agent said
    # complete while still active; `held` = quiet + not done, robot stands until next decision.
    # The idle monitor judges the running step back to back once `seen` is set.
    _dm = {"step": None, "seen": None, "want": False, "want_t": 0.0, "held": False, "pre": None}
    _step_t0 = {"step": None, "t": 0.0}     # when the current step began (for MIN_STEP_FAIL_S)

    def _seq_current_step(plan):
        """(idx, subtask) of the first open step, or (None, None)."""
        for i, s in enumerate(plan.steps):
            if s.status in ("pending", "in_progress"):
                return i, s.subtask
        return None, None
    # Session memory (tick thread only): the operator conversation, rendered to text before
    # being handed to the T1 worker.
    mission_hist = MissionHistory()
    # Recent ego frames sent as a time sequence; reset per mission.
    frame_ring = FrameRing()
    # Immutable Plan snapshot the sequencer reads each tick.
    plan_obj = unauthored(run_id, mission=orch.mission, steps=())
    # Async deliberation: T1 runs on a background thread; the tick loop reads the snapshot lock-free.
    plan_ref = PlanRef(plan_obj)

    def _deliberate_on_worker(prev_plan, payload):
        """Worker-thread T1 call; `orch.deliberate()` is pure, so this is safe off-thread."""
        payload = payload or {}
        new_plan = orch.deliberate(prev_plan, image=payload.get("image"),
                                   frames=payload.get("frames"),
                                   operator_instruction=payload.get("operator_instruction"),
                                   history_turns=payload.get("history_turns"))
        # Superseded: return the previous plan, which the worker reinstalls (= discard).
        if int(payload.get("epoch", 0)) != delib_epoch:
            print(f"[deliberate] DISCARDED a superseded result (built for epoch "
                  f"{payload.get('epoch')}, current is {delib_epoch}) -- the operator changed the "
                  f"request while this call was in flight", flush=True)
            return prev_plan
        if new_plan.reply:
            print(f"[deliberate] reply: {new_plan.reply}", flush=True)
        if not new_plan.steps:
            # Conversational request: answered, no action.
            print("[deliberate] no steps — conversational request, nothing to execute", flush=True)
        print(f"[deliberate] plan v{new_plan.version}: "
              f"{' -> '.join(s.subtask for s in new_plan.steps)}", flush=True)
        return new_plan

    delib = StaticDeliberator(plan_ref, _deliberate_on_worker,
                              logger=lambda m: print(f"[deliberate] {m}", flush=True))

    _last_plan_str = None          # dedup: only republish when the checklist actually changes
    t_start = time.time()          # orchestrator start, for the task-complete print

    _last_plan_snap = None         # freshest snapshot, for the g1/plan heartbeat (slow-joiner)
    _last_plan_pub_t = 0.0

    def _emit_plan(snap):
        """Publish the checklist on g1/plan; change-gated."""
        nonlocal _last_plan_str, _last_plan_snap, _last_plan_pub_t
        if not snap:
            return
        s = json.dumps(snap, sort_keys=True)
        if s == _last_plan_str:
            return
        _last_plan_str = s
        rec = dict(snap); rec["stamp"] = time.time()
        _last_plan_snap = rec
        activity.plan(rec)
        _last_plan_pub_t = time.time()

    # Plan heartbeat: republish the freshest snapshot so a late subscriber gets the current plan.
    PLAN_HEARTBEAT_S = 1.0

    def _plan_heartbeat():
        nonlocal _last_plan_pub_t
        if _last_plan_snap is None:
            return
        now = time.time()
        if now - _last_plan_pub_t < PLAN_HEARTBEAT_S:
            return
        _last_plan_pub_t = now
        activity.plan(_last_plan_snap)

    # Publish the empty unauthored plan at startup so a viewer never shows another run's checklist.
    _emit_plan(plan_obj.snapshot())

    def _publish_hold(reason_text):
        """Publish a HOLD mission every tick while holding (slow-joiner safe)."""
        nonlocal pub_tick
        pub_tick += 1
        # Name/id/prompt come from the registry so the id always matches the name.
        mc.publish_mission({"subtask_id": _registry.id_of(_reg, _registry.HOLD),
                            "subtask": _registry.HOLD, "prompt": _prompt_for(_registry.HOLD),
                            "reason": reason_text, "why": _last_say[0], "done": False,
                            "tick": pub_tick, "stamp": time.time()})

    def _publish_step(sub, reason=""):
        """Publish `sub` on g1/mission; returns the generator prompt sent."""
        nonlocal pub_tick
        pub_tick += 1
        prompt = prompt_map.get(sub) or _prompt_for(sub)
        mc.publish_mission({"subtask_id": _registry.id_of(_reg, sub), "subtask": sub,
                            "prompt": prompt, "reason": reason, "why": _last_say[0], "done": False,
                            "tick": pub_tick, "stamp": time.time()})
        return prompt

    def _log_switch(sub, prompt):
        if _seq_prev[0] != sub:
            print(f"[mission] === SWITCH -> {sub} | generator prompt: \"{prompt}\" ===", flush=True)
            _seq_prev[0] = sub
    # Operator inbox: the web viewer appends {"text": ...} JSON lines (operator instructions);
    # start at EOF to ignore previous runs.
    human_note = None
    human_file = Path(os.environ.get("HALO_CACHE", "~/.cache/halo")).expanduser() / "mission_human_input.jsonl"
    try:
        human_pos = human_file.stat().st_size if human_file.exists() else 0
    except OSError:
        human_pos = 0

    def _t1_payload(obs, instruction):
        """Build a T1 request payload on the tick thread (the worker never reads tick state)."""
        now = time.time()
        frames = frame_ring.labelled(now)
        # The ring is reset on a new mission; fall back to the current frame so T1 is never blind.
        if not frames and obs is not None and obs.rgb is not None:
            frames = [("ego frame: current", obs.rgb)]
        return {"image": obs.rgb, "frames": frames,
                "epoch": delib_epoch,
                "operator_instruction": instruction,
                "history_turns": mission_hist.turns()}

    print("[mission] waiting for g1/obs frames…  (Ctrl-C to stop)", flush=True)
    got_frame = False
    try:
        while True:
            # Cadence comes from the generator: g1/obs is published once per generated plan, so
            # the decision rate equals the generator's inference rate.
            obs = mc.recv_obs(timeout_ms=1000)
            if obs is None:
                # Keep g1/plan alive even when no ego frames arrive (robot idle / holding).
                _plan_heartbeat()
                continue
            if not got_frame:
                got_frame = True
                print(f"[mission] first g1/obs frame {obs.rgb.shape} — running decisions",
                      flush=True)
            # Publish a newly authored plan now: the T1 worker installs it without publishing, and
            # _emit_plan must stay on the tick thread.
            _authored = plan_ref.get()
            if _authored.version != last_published_plan_version:
                last_published_plan_version = _authored.version
                _emit_plan(_authored.snapshot())
                # A new authored version means the deliberation landed.
                delib_inflight = False
                # Record T1's reply in the session history here so conversational turns count too.
                if _authored.authored and _authored.reply:
                    mission_hist.attach_reply(_authored.reply)
            # Republish the checklist before every branch (hold paths `continue` early).
            _plan_heartbeat()
            # Frame stamp; 0 when the publisher sent none.
            frame_ring.push(obs.stamp if obs.stamp > 0 else time.time(), obs.rgb)
            # Gamepad hold: X toggles an operator hold (freeze in place); the next press resumes the plan. Rising edge only.
            _sm = _state_msg()
            _bits = int(getattr(_sm, "remote_buttons", 0) or 0) if _sm is not None else 0
            _down = (_bits >> _gp_bit) & 1
            if _down and not _gp["prev"]:
                _gp["hold"] = not _gp["hold"]
                if not _gp["hold"]:
                    _seq_prev[0] = None       # the resumed step logs as a SWITCH
                print(f"[mission][gamepad] X pressed -> {'hold' if _gp['hold'] else 'resume'}", flush=True)
            _gp["prev"] = _down
            # Read new operator inbox lines (latest wins, persists).
            try:
                if human_file.exists() and human_file.stat().st_size > human_pos:
                    with open(human_file) as hf:
                        hf.seek(human_pos)
                        for line in hf:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                _rec = json.loads(line)
                            except ValueError:
                                continue
                            if not isinstance(_rec, dict):
                                continue
                            txt = str(_rec.get("text", "")).strip()
                            if txt:
                                if awaiting and not wait_for_start:
                                    # The answer to an ask_operator question: a note, not a re-plan.
                                    human_note = txt
                                    print(f"[mission] operator (answer): {txt}", flush=True)
                                else:
                                    # First instruction => becomes the mission; later ones interrupt
                                    # and re-plan (agentic too: an empty conversational plan has no
                                    # step to settle on).
                                    first = wait_for_start
                                    if not first:
                                        # Unfinished steps become `cancelled`.
                                        plan_ref.set(plan_ref.get().stop_unfinished())
                                    orch.set_mission(txt)
                                    wait_for_start = False
                                    pending_plan = True   # build the checklist next tick
                                    mission_complete = False
                                    gate_from_version = plan_ref.get().version   # after stop_unfinished
                                    mission_hist.open_mission(txt)
                                    frame_ring = FrameRing()
                                    print(f"[mission] mission set by operator: \"{txt}\" — starting."
                                          if first else
                                          f"[mission] new instruction -> re-planning: \"{txt}\"",
                                          flush=True)
                                awaiting = False
                        human_pos = hf.tell()
            except OSError:
                pass
            _hold_msg = None
            if _gp["hold"]:
                # Gamepad hold; the generator freezes on the "operator hold" reason prefix (motiongen_zmq.py).
                _hold_msg = "operator hold (gamepad): freeze in place — press X to resume"
            elif wait_for_start:
                # No mission yet -> HOLD and send no task until the operator's first instruction.
                _hold_msg = "💬 Waiting for your instruction — type what the robot should do…"
            elif awaiting:
                # An unanswered agent question -> hold the robot; don't run the VLM.
                _hold_msg = "❓ Holding — waiting for your answer…"
            elif mission_complete:
                # Hold the pose and stop querying the VLM until a new mission.
                _hold_msg = "✅ Mission complete — holding pose."
            elif (plan_ref.get().authored and not plan_ref.get().steps
                    and not pending_plan and not delib_inflight):
                # Authored but empty (conversational reply) -> hold and skip T2. An unauthored empty
                # plan (bootstrap) still needs T2; `not pending_plan` keeps a fresh instruction alive.
                _hold_msg = "💬 Answered — no action needed. Ask me to do something."
            if _hold_msg is not None:
                _publish_hold(_hold_msg)
                time.sleep(hold_period)
                continue
            # Agentic executor: T1 authors the plan; the node walks it and calls the agent
            # (orch.decide_at_settle) at a settle, an operator utterance, or an idle-monitor tick.
            _pnow = plan_ref.get()
            _authoring = pending_plan or (delib.in_progress_since is not None)
            if _pnow.authored and _pnow.steps and not _authoring:
                ptext, bh, rxy = _pose_text()
                jact, _gsb = _gen_status()
                plan_obj = _pnow
                idx, cur = _seq_current_step(plan_obj)
                if cur is not None and plan_obj.steps[idx].status == "pending":
                    plan_obj = plan_obj.reopen(idx)      # show it in_progress in the web viewer
                    plan_ref.set(plan_obj); _emit_plan(plan_obj.snapshot())
                # The robot keeps running while the agent judges back to back; quiet = the latest
                # plan's joint activity is under the threshold.
                if _step_t0.get("step") != cur:
                    _step_t0.update(step=cur, t=time.time())
                _tnow = time.time()
                if _dm["step"] != cur:
                    _dm.update(step=cur, seen=None, want=False, held=False, pre=None)
                if cur is not None and _gsb == cur and _dm["seen"] is None:
                    _dm["seen"] = _tnow
                # Pre-switch hold: after a done verdict, stand until SWITCH_HOLD_S has run and
                # the joints are steady; no model calls meanwhile.
                if cur is not None and _dm["pre"] is not None:
                    _pr = _dm["pre"]; _el = _tnow - _pr["t0"]
                    _sp = _joint_speed()
                    if _sp is not None and _sp <= SWITCH_STEADY_SPEED:
                        _pr["still"] = _pr["still"] or _tnow
                    else:
                        _pr["still"] = None
                    _steady = (_sp is None) or (_pr["still"] is not None
                                                 and _tnow - _pr["still"] >= 0.3)
                    if (_el >= SWITCH_HOLD_S and _steady) or _el >= SWITCH_HOLD_MAX_S:
                        if _pr["act"] == "finish_mission":
                            plan_obj = plan_obj.close_all()
                        elif _pr["success"] is False:
                            plan_obj = plan_obj.fail_current()
                        else:
                            plan_obj = plan_obj.close_current()
                        plan_ref.set(plan_obj); _emit_plan(plan_obj.snapshot())
                        _how = ("steady" if _el < SWITCH_HOLD_MAX_S
                                else "max wait reached")
                        print(f"[mission][agentic] '{cur}' hold {_how} after {_el:.1f}s "
                              f"(joint speed {(_sp if _sp is not None else 0):.2f} rad/s) -> "
                              f"switching", flush=True)
                        _dm.update(step=None, want=False, held=False, pre=None)
                        continue      # next tick publishes the next step (or completes)
                    _publish_hold(f"'{cur}' done — settling in the hold before the next step")
                    _plan_heartbeat(); time.sleep(hold_period); continue
                _quiet = (cur is not None and _gsb == cur and _is_quiet(jact, SWITCH_ACTIVITY_THRESH))
                _stand_ok = ((bh is None) or (_seated_ok(bh) if cur in ENDS_LOW_SKILLS
                                              else (bh >= 0.60)))
                # Deferred switch: agent already said done while active -> hold once quiet.
                if cur is not None and _dm["want"] and (_quiet or _dm["held"]) and _stand_ok:
                    _why_q = ("held" if not _quiet else f"activity {jact:.4f} <= {SWITCH_ACTIVITY_THRESH:g}")
                    print(f"[mission][agentic] '{cur}' quiet ({_why_q}) -> HOLD, then the "
                          f"next step once steady (agent had said complete_step)", flush=True)
                    _dm.update(want=False, held=False,
                               pre={"t0": time.time(), "act": "complete_step",
                                    "success": True, "still": None})
                    _publish_hold(f"'{cur}' done — settling in the hold before the next step")
                    _plan_heartbeat(); time.sleep(hold_period); continue
                # While a switch is pending, poll instead of calling the model; re-ask after
                # COMPLETE_AFTER_S.
                _pending = _dm["want"] and (_tnow - _dm["want_t"]) < COMPLETE_AFTER_S
                _monitor_due = cur is not None and _dm["seen"] is not None and not _pending
                if cur is not None and (human_note is not None or _monitor_due):
                    # Stall readout: world-XY progress on the current step.
                    if _stall["step"] != cur or _stall["ref_xy"] is None:
                        _stall.update(step=cur, count=0, ref_xy=rxy)
                    _prog = None
                    if rxy is not None and _stall["ref_xy"] is not None:
                        _prog = math.hypot(rxy[0] - _stall["ref_xy"][0],
                                           rxy[1] - _stall["ref_xy"][1])
                        if _prog >= _STALL_PROGRESS_M:      # real progress -> not stalled
                            _stall.update(count=0, ref_xy=rxy); _prog = 0.0
                        else:
                            _stall["count"] += 1
                    _t_call = time.time()
                    # A 'not done' judged on an image of a still-moving robot must not trigger a hold.
                    _stopped_at_call = _dm["held"] or (
                        _gsb == cur and _is_quiet(jact, HOLD_ACTIVITY_THRESH))
                    try:
                        jd = orch.decide_at_settle(
                            plan=plan_obj, image=obs.rgb,
                            frames=frame_ring.labelled(time.time()),
                            running_subtask=cur, operator_note=human_note,
                            at_settle=_dm["held"],
                            pose_text=ptext,
                            stall_count=_stall["count"], progress_m=_prog)
                    except Exception as _exc:   # noqa: BLE001
                        # Transient LLM errors must not abort the mission: keep the current step and
                        # the pending operator note, retry next tick.
                        print(f"[mission][agentic] decide_at_settle FAILED "
                              f"({type(_exc).__name__}: {_exc}); keeping current step, retry next "
                              f"tick", flush=True)
                        _plan_heartbeat(); time.sleep(hold_period); continue
                    human_note = None
                    act = jd.get("action"); _say = jd.get("say", "")
                    # Guardrails: complete/finish only on a settle and with the robot standing
                    # (or seated for ENDS_LOW skills); otherwise coerce to retry_step.
                    if act in ("complete_step", "finish_mission"):
                        _ends_low = cur in ENDS_LOW_SKILLS
                        _standing = _ends_low or (bh is None) or (bh >= 0.60)
                        # A skill that ends low is done only when actually seated.
                        _seated = (not _ends_low) or _seated_ok(bh)
                        # Re-read after the call: the robot kept running while the model thought.
                        _ja2, _gs2 = _gen_status()
                        _settle_ok = _dm["held"] or (_gs2 == cur and _is_quiet(_ja2, SWITCH_ACTIVITY_THRESH))
                        # A step may not complete without having moved (NEVER_MOVES_SKILLS exempt):
                        # the model must have judged a frame of THIS step running.
                        _moved = (cur in NEVER_MOVES_SKILLS
                                  or (_dm["seen"] is not None and _dm["seen"] <= _t_call))
                        # Direct switch: walk-like skills switch at once when standing.
                        _direct = (act == "complete_step" and _direct_switch(cur)
                                   and _moved and (bh is None or bh >= 0.60))
                        if _direct:
                            _settle_ok = True
                        _only_moving = (not _settle_ok) and _standing and _seated and _moved
                        if _only_moving:
                            # The agent's verdict stands; switch once quiet.
                            if not _dm["want"]:
                                _dm["want_t"] = time.time()
                            _dm["want"] = True
                            print(f"[mission][agentic] '{cur}': complete_step while the plan is "
                                  f"still active -> switching as soon as it is quiet", flush=True)
                        if (not _settle_ok) or (not _standing) or (not _seated) or (not _moved):
                            _why = ("plan still active; switching when quiet" if not _settle_ok
                                    else f"not standing (base_height={bh:.2f})" if not _standing
                                    else (f"not seated (base_height={bh:.2f}, tilt "
                                          f"{(_tilt_deg() or 0):.0f} deg; fallen or still standing)")
                                    if not _seated
                                    else "step never moved (no motion seen since it began)")
                            print(f"[mission][agentic] guard: '{act}' blocked ({_why}) "
                                  f"-> retry_step", flush=True)
                            act = "retry_step"; jd["action"] = "retry_step"; jd["plan"] = plan_obj
                            # "Still moving" is not a failure: keep it out of the retry cap.
                            jd["_guard_still_moving"] = _only_moving
                        elif _direct:
                            # Step closes now; the next tick publishes the next step.
                            print(f"[mission][agentic] '{cur}' {act} accepted -> switching NOW "
                                  f"(direct switch: standing, no hold)", flush=True)
                            _dm.update(want=False, held=False, pre=None)
                        else:
                            # Accepted: pre-switch hold first; closes once steady.
                            print(f"[mission][agentic] '{cur}' {act} accepted -> HOLD, then the "
                                  f"next step once steady", flush=True)
                            _dm.update(want=False, held=False,
                                       pre={"t0": time.time(), "act": act,
                                            "success": jd.get("success", True), "still": None})
                            act = "complete_pending"; jd["action"] = act; jd["plan"] = plan_obj
                    # ask_operator: hold and surface the question; the reply returns as human_note.
                    if act == "ask_operator":
                        awaiting = True
                        _q = jd.get("question") or "I'm not sure what to do — can you guide me?"
                        print(f"[mission][agentic] ask_operator: {_q}", flush=True)
                        if _say:
                            _last_say[0] = _say
                        _publish_hold(f"❓ {_q}")
                        _plan_heartbeat(); time.sleep(hold_period); continue
                    if act == "insert_step":
                        # Drop a too-soon insert, or one equal to the running step.
                        _now = time.time()
                        if (_now - _last_insert[0]) < INSERT_COOLDOWN_S or jd.get("skill") == cur:
                            jd["plan"] = plan_obj      # keep running the current step
                        else:
                            _last_insert[0] = _now
                    plan_obj = jd.get("plan", plan_obj)
                    # retry cap: after N consecutive retries, force fail+advance
                    if act == "retry_step":
                        if not jd.get("_guard_still_moving"):
                            _retry[cur] = _retry.get(cur, 0) + 1
                        # Living follow is exempt from the retry cap (only the operator ends it).
                        _living = (_registry.target_kind_of(_reg, cur) == "living")
                        _age = time.time() - _step_t0.get("t", 0.0)
                        if (not _living and _retry.get(cur, 0) > MAX_RETRIES
                                and _age >= MIN_STEP_FAIL_S):
                            plan_obj = plan_obj.fail_current()
                            _retry[cur] = 0
                            print(f"[mission][agentic] retry cap on '{cur}' after {_age:.0f}s "
                                  f"-> fail+advance", flush=True)
                    else:
                        _retry.pop(cur, None)
                    plan_ref.set(plan_obj); _emit_plan(plan_obj.snapshot())
                    print(f"[mission][agentic] '{cur}' -> {act}"
                          + (f"  say: {_say}" if _say else ""), flush=True)
                    if _say:
                        _last_say[0] = _say
                    if act == "finish_mission" or plan_obj.done:
                        mission_complete = True
                    if act == "retry_step" and not jd.get("_guard_still_moving"):
                        # The agent's own 'not done' (not a guard coercion).
                        _dm["want"] = False
                        if _dm["held"]:
                            # held, judged again, still not done -> resume the step
                            _dm["held"] = False
                            print(f"[mission][agentic] '{cur}' not done after holding -> "
                                  f"resuming the step", flush=True)
                        elif not _living and _stopped_at_call:
                            # (A 'not done' judged on a still-moving image is stale: no hold.)
                            _ja3, _gs3 = _gen_status()
                            # Hold threshold: only when really stopped.
                            if _gs3 == cur and _is_quiet(_ja3, HOLD_ACTIVITY_THRESH):
                                _dm["held"] = True
                                print(f"[mission][agentic] '{cur}' quiet but not done -> HOLD "
                                      f"until the next decision", flush=True)
                    idx, cur = _seq_current_step(plan_obj)
                if mission_complete or cur is None:
                    # An empty plan is a conversational reply, not a finished task.
                    _empty_plan = not plan_obj.steps
                    if not mission_complete:
                        mission_complete = True
                        if _empty_plan:
                            print("[mission] nothing to execute (conversational reply) — "
                                  "holding; say what the robot should do.", flush=True)
                        else:
                            print(f"[mission] ✅ TASK COMPLETE (all steps done) after "
                                  f"{time.time() - t_start:.1f}s", flush=True)
                    _publish_hold("💬 Waiting for your instruction…" if _empty_plan
                                  else "✅ Task complete.")
                    _plan_heartbeat(); time.sleep(hold_period); continue
                # Publish the (possibly advanced/inserted) current step.
                if _dm["pre"] is not None and _dm["step"] == cur:
                    _publish_hold(f"'{cur}' done — settling in the hold before the next step")
                    _plan_heartbeat(); time.sleep(hold_period); continue
                if _dm["held"] and _dm["step"] == cur:
                    _publish_hold(f"settled — '{cur}' not done yet, holding for the next decision")
                    _plan_heartbeat(); time.sleep(hold_period); continue
                _log_switch(cur, _publish_step(cur))
                # Sleep hold_period; busy-spinning here starves the tracker/generator of CPU.
                _plan_heartbeat(); time.sleep(hold_period); continue
            # No authored plan yet (or authoring in flight) -> fall through to the T1 gate.
            # T1 authoring + deliberation gate (no plan yet, or a re-plan in flight).
            try:
                plan_version_used = plan_ref.get().version
                if pending_plan:
                    # Operator-requested deliberation; bumping the epoch discards older results.
                    delib_epoch += 1
                    delib_inflight = True
                    # Enqueue T1 on the background worker (never inline).
                    delib.request(_t1_payload(obs, human_note))
                    pending_plan = False
                    gate_since = time.time()
                    print(f"[mission] deliberation ENQUEUED (background) on plan "
                          f"v{plan_version_used}; HOLDING the robot until the plan lands", flush=True)
                # Deliberation gate: hold until T1's plan arrives, for the bootstrap and operator
                # re-plans; the code never invents a skill.
                if gate_since is not None:
                    waited = time.time() - gate_since
                    if delib.in_progress_since is not None or plan_ref.get().version <= gate_from_version:
                        if waited < GATE_TIMEOUT_S:
                            # Contract: the web viewer recognises a deliberation hold by the
                            # leading "Deliberating" in the reason; keep it first and ASCII.
                            _publish_hold(f"Deliberating — planning the mission before moving ({waited:.1f}s) 🤔")
                            time.sleep(hold_period)
                            continue
                        # Budget expired: release the gate so the mission still starts.
                        print(f"[mission] WARNING: deliberation did not produce a plan within "
                              f"{GATE_TIMEOUT_S:.0f}s (waited {waited:.1f}s) — releasing the "
                              f"gate; the robot keeps holding until a plan arrives (plan v{plan_ref.get().version}).",
                              flush=True)
                        gate_since = None
                    else:
                        # The plan landed: the sequencer executes it from the next tick.
                        plan_version_used = plan_ref.get().version
                        print(f"[mission] plan v{plan_version_used} authored after {waited:.1f}s — "
                              f"releasing the gate", flush=True)
                        gate_since = None
                # A freshly landed plan with steps: publish its current step (empty plan holds).
                _pn = plan_ref.get()
                _c = _seq_current_step(_pn)[1] if (_pn.authored and _pn.steps) else None
                if _c is not None:
                    _publish_step(_c, "plan ready")
                else:
                    _publish_hold("Planning… 🤔")
            except Exception as exc:          # noqa: BLE001 — an error must not kill the loop
                print(f"[mission] tick error: {type(exc).__name__}: {exc}", flush=True)
            _plan_heartbeat(); time.sleep(hold_period)
    except KeyboardInterrupt:
        print("\n[mission] stopping.", flush=True)
    finally:
        try:
            mc.close()
        except Exception:  # noqa: BLE001
            pass
    # Exit now: ZMQ context teardown can hang and keep port 5554 bound. `os` must stay a
    # module-level import (a local import would shadow it for all of main()).
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
