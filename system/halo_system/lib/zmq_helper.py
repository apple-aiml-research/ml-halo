#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""zmq_helper — generator-side halo_comm clients for the tracker<->generator channels.

``TrackerClient`` is a thin wrapper over ``halo_comm.Node`` that talks protobuf to
a C++ tracker node (``motiontrack_mujoco_zmq`` / ``motiontrack_real_zmq``).
``ObsPublisher`` is its publish-side counterpart: the generator re-publishes the
ego frame it already holds on the ``g1/obs`` topic (see that class for why).
Imported by the generator node (``motiongen_zmq.py``) and the launchers.

Channels (must match system/cpp/transport/tracker_server.hpp):

  service  g1/get_state  StateRequest  -> RobotState    (REQ, port 5555)
  service  g1/put_plan   MotionPlan    -> PlanAck        (REQ, port 5556)
  service  g1/get_trim   TrimRequest   -> TrimReply       (REQ, port 5557)
  topic    g1/state      RobotState                        (SUB, port 5558)
  topic    g1/obs        ObsImage                           (SUB, port 5559)  [generator-published]
  topic    g1/mission    MissionCommand                      (PUB/SUB, port 5554)  [orchestrator-published]
  topic    g1/gen_status GenStatus                            (PUB/SUB, port 5561)  [generator-published]
  topic    g1/plan         MissionPlanMsg                     (PUB/SUB, port 5564)  [orchestrator-published]

Note: 5560 is the halo_comm discovery REGISTRY port (see halo_comm), not a data channel — g1/mission
must NOT default there, or an orchestrator binding it collides with the registry ("Address already in
use :5560"). g1/mission uses 5554 by convention across this stack (orchestrator_zmq.py,
visualization/server.py, and this default all agree).

Plan/trim rows are flat row-major doubles, ``cols`` wide, each row
``[base_xyz(3), base_quat_xyzw(4), joints_deploy(29)]`` (cols = 36).
"""

from __future__ import annotations

import time

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import zmq

from halo_comm import Node, discovery, state_pb2, motion_pb2

PLAN_COLS = 36  # base_xyz(3) + base_quat_xyzw(4) + joints_deploy(29)

# Per-chunk plan flags — mirror policy::kFlag* (raw_motion_plan.hpp).
FLAG_ALIGN_YAW = 1 << 0
FLAG_AUTO_START_OFFSET = 1 << 1
FLAG_ROBOT_ANCHOR = 1 << 2   # install with align_anchor="robot" (re-ground on the robot's pose)


@dataclass
class RobotStateView:
    """Decoded state reply (the fields the generator's rebase step needs)."""

    joint_pos: np.ndarray          # (29,) deployment order
    imu_quat_wxyz: np.ndarray      # (4,)
    base_pos: np.ndarray           # (3,)
    tick: int
    ref_motion: Optional[dict] = None  # {'pos','quat_wxyz','joints'} or None
    image: Optional[np.ndarray] = None  # (H, W, 3) uint8 RGB, or None (REP path only)
    tracker_ready: bool = False        # latched by the tracker when ready to accept plans

    @classmethod
    def from_proto(cls, msg: "state_pb2.RobotState") -> "RobotStateView":
        ref = None
        if msg.HasField("ref_motion"):
            rm = msg.ref_motion
            ref = {
                "pos": np.asarray(rm.pos, dtype=np.float64),
                "quat_wxyz": np.asarray(rm.quat_wxyz, dtype=np.float64),
                "joints": np.asarray(rm.joints, dtype=np.float64),
            }
        # Optional ego-camera frame (REP get_state path only; never the topic).
        # Absent on trackers that don't render (image=None).
        image = None
        if msg.HasField("image") and msg.image.width > 0 and msg.image.height > 0:
            im = msg.image
            image = np.frombuffer(im.rgb, dtype=np.uint8).reshape(
                im.height, im.width, im.channels).copy()
        return cls(
            joint_pos=np.asarray(msg.joint_pos, dtype=np.float64),
            imu_quat_wxyz=np.asarray(msg.imu_quat_wxyz, dtype=np.float64),
            base_pos=np.asarray(msg.base_pos, dtype=np.float64),
            tick=int(msg.tick),
            ref_motion=ref,
            image=image,
            tracker_ready=bool(msg.tracker_ready),
        )


@dataclass
class TrimResult:
    """Decoded trim reply. ``plan`` is (n_rows, 36); empty => motion exhausted."""

    plan: np.ndarray
    input_fps: int


def _resolve(node: Node, channel: str) -> tuple:
    """(host, port) of `channel` from the node's halo_comm registry; waits until the provider
    appears, so start order does not matter."""
    reg = node._registry
    waited = 0.0
    while True:
        got = discovery.resolve_or_wait(node._ctx, reg, channel, timeout_s=5.0)
        if got is not None:
            print(f"[discovery] resolved {channel} -> {got[0]}:{got[1]} "
                  f"(registry {reg[0]}:{reg[1]})", flush=True)
            return got[0], int(got[1])
        waited += 5.0
        print(f"[discovery] still waiting for {channel} in registry {reg[0]}:{reg[1]}... "
              f"({waited:.0f}s)  Ctrl-C to abort.", flush=True)


class TrackerClient:
    def __init__(
        self,
        tracker_host: Optional[str] = None,
        state_port: int = 5555,
        plan_port: int = 5556,
        trim_port: int = 5557,
        state_pub_port: int = 5558,
        rcv_ms: int = 5000,
        snd_ms: int = 2000,
    ):
        self.node = Node("generator")
        # With no explicit tracker_host, resolve the tracker's host from the registry (all four
        # g1/* channels share it -- the tracker binds them).
        if tracker_host is None:
            tracker_host, state_port = _resolve(self.node, "g1/get_state")
        self.host = tracker_host
        self._state_cli = self.node.create_client(
            state_pb2.StateRequest, state_pb2.RobotState,
            "g1/get_state", tracker_host, state_port, rcv_ms, snd_ms)
        self._plan_cli = self.node.create_client(
            motion_pb2.MotionPlan, motion_pb2.PlanAck,
            "g1/put_plan", tracker_host, plan_port, rcv_ms, snd_ms)
        self._trim_cli = self.node.create_client(
            motion_pb2.TrimRequest, motion_pb2.TrimReply,
            "g1/get_trim", tracker_host, trim_port, rcv_ms, snd_ms)
        self._state_pub_port = state_pub_port

    # ── services (blocking REQ/REP) ──────────────────────────────────────────
    def get_state(self) -> RobotStateView:
        """Blocking get_state RPC. Raises zmq.Again on timeout."""
        return RobotStateView.from_proto(self._state_cli.call(state_pb2.StateRequest()))

    def send_plan(self, qpos_data: np.ndarray, input_fps: int = 30,
                  align_yaw: bool = True, auto_start_offset: bool = False,
                  align_anchor: str = "reference"
                  ) -> bool:
        """Push a motion plan. ``qpos_data`` is (n_rows, 36) row-major.

        Returns whether the tracker accepted it. Yaw alignment is server-side (FLAG_ALIGN_YAW) — the generator
        sends the plan in the model's own frame and does NOT pre-rebase it; the
        tracker yaw-aligns the chunk's frame 0 to the robot's measured heading.
        Pass ``align_yaw=False`` to install the plan verbatim (no rebase).

        ``align_anchor`` picks what the tracker rebases frame 0 onto:
        ``"reference"`` (default — continuous with the previous reference) or
        ``"robot"`` (re-ground on the robot's current pose, forgiving drift;
        sets FLAG_ROBOT_ANCHOR). Used for plan #1 and the standing hold.
        Honored by the C++ tracker.
        """
        flags = 0
        if align_yaw:
            flags |= FLAG_ALIGN_YAW
        if auto_start_offset:
            flags |= FLAG_AUTO_START_OFFSET
        if align_anchor == "robot":
            flags |= FLAG_ROBOT_ANCHOR
        arr = np.ascontiguousarray(qpos_data, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != PLAN_COLS:
            raise ValueError(f"qpos_data must be (n_rows, {PLAN_COLS}); got {arr.shape}")
        req = motion_pb2.MotionPlan(
            input_fps=int(input_fps),
            flags=int(flags),
            cols=PLAN_COLS,
            values=arr.reshape(-1).tolist(),
        )
        ack = self._plan_cli.call(req)
        return bool(ack.ok)

    def get_trim(self) -> TrimResult:
        """RTC trim RPC. Returns the trimmed plan (n_rows, 36) or empty."""
        rep = self._trim_cli.call(motion_pb2.TrimRequest())
        cols = rep.cols or PLAN_COLS
        vals = np.asarray(rep.values, dtype=np.float64)
        n_rows = len(vals) // cols
        plan = vals.reshape(n_rows, cols) if n_rows else np.empty((0, cols), dtype=np.float64)
        return TrimResult(plan=plan, input_fps=int(rep.input_fps))

    # ── topic (SUB, freshest state) ──────────────────────────────────────────
    def subscribe_state(self, on_message: Callable) -> None:
        """Subscribe to the g1/state topic; ``on_message(RobotStateView)`` fires on a background recv thread."""
        self.node.create_subscription(
            state_pb2.RobotState, "g1/state", self.host, self._state_pub_port,
            conflate=True, on_message=lambda m: on_message(RobotStateView.from_proto(m)))

    def close(self) -> None:
        self.node.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# ObsPublisher — generator-side ego-image publisher for the g1/obs topic
# ═══════════════════════════════════════════════════════════════════════════
# The ego camera image lives on DIFFERENT nodes in sim vs on the real robot:
#   - sim  : the *tracker* renders MuJoCo's ego camera and ships it on the
#            g1/get_state REP path (only when asked).
#   - real : the *generator* owns the physical OAK camera; the
#            real tracker ships state only (no image).
# Either way the GENERATOR already HOLDS an ego frame every cycle (sim: the one
# it pulled from get_state for its own policy; real: its own camera frame). This
# helper RE-PUBLISHES that held frame on a dedicated topic g1/obs so the browser
# viewer and the orchestrator can subscribe instead of hammering get_state (one
# extra MuJoCo render per pull in sim). Fire-and-forget + CONFLATE: a slow/absent
# subscriber never back-pressures the generator.
# RAW (~1.2 MiB/frame) is fine on localhost; JPEG q80 (~100 KB) on robot Wi-Fi.

class ObsPublisher:
    """Publishes ``ObsImage`` frames on the ``g1/obs`` topic.

    Parameters
    ----------
    node : halo_comm.Node
        The generator's existing node/context.
    port : int
        Bind port (g1/obs, 5559 by convention).
    jpeg : bool
        If True, JPEG-encode each frame with cv2 (recommended on the real robot).
    quality : int
        JPEG quality (1-100) when ``jpeg`` is True.
    """

    def __init__(self, node: Node, port: int, jpeg: bool = False, quality: int = 80):
        self._port = int(port)
        self._jpeg = bool(jpeg)
        self._quality = int(quality)
        self._pub = node.create_publisher(state_pb2.ObsImage, "g1/obs", self._port)
        self._tick = 0
        self._logged = False           # one-line bandwidth log on first publish
        self._cv2 = None
        if self._jpeg:
            import cv2
            self._cv2 = cv2

    def _encode_jpeg(self, rgb: np.ndarray) -> Optional[bytes]:
        """RGB uint8 (H,W,3) -> JPEG bytes, or None on failure."""
        cv2 = self._cv2
        try:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality])
            return buf.tobytes() if ok else None
        except Exception as exc:  # noqa: BLE001
            print(f"[obs] WARN: JPEG encode failed ({exc}); dropping frame.")
        return None

    def publish(self, rgb: np.ndarray, tick: Optional[int] = None) -> bool:
        """Publish one RGB uint8 (H,W,3) frame on g1/obs. Returns True if a frame was sent."""
        if rgb is None:
            return False
        arr = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        if arr.ndim != 3 or arr.shape[2] < 3:
            return False
        h, w = int(arr.shape[0]), int(arr.shape[1])
        arr = arr[:, :, :3]
        msg = state_pb2.ObsImage()
        msg.width, msg.height, msg.channels = w, h, 3
        msg.tick = int(self._tick if tick is None else tick)
        msg.stamp = time.time()
        data = self._encode_jpeg(arr) if self._jpeg else None
        if data is not None:
            msg.encoding, msg.data = state_pb2.ObsImage.JPEG, data
        else:                          # RAW, or JPEG encode failed
            msg.encoding, msg.data = state_pb2.ObsImage.RAW, arr.tobytes()
        try:
            self._pub.publish(msg)
        except zmq.ZMQError:   # socket closed at shutdown
            return False
        self._tick += 1
        if not self._logged:
            enc = "JPEG" if msg.encoding == state_pb2.ObsImage.JPEG else "RAW"
            print(f"[obs] publishing g1/obs :{self._port}  {msg.width}x{msg.height} "
                  f"{enc}  {len(msg.data)/1024:.1f} KiB/frame", flush=True)
            self._logged = True
        return True

    def close(self) -> None:
        try:
            self._pub.close()
        except Exception:  # noqa: BLE001
            pass


# ─────────────────────────────────────────────────────────────────────────────
# MissionClient — transport for the VLM mission orchestrator (SEPARATE from TrackerClient).
# SUB g1/obs (5559, generator-published ego image) + PUB/SUB g1/mission (5554, MissionCommand).
# Lives here so all transport stays in one file, but is its own class (no TrackerClient pollution).
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ObsImageView:
    """Decoded g1/obs frame."""
    rgb: np.ndarray            # (H, W, 3) uint8 RGB
    tick: int
    stamp: float

    @classmethod
    def from_proto(cls, msg: "state_pb2.ObsImage") -> Optional["ObsImageView"]:
        if msg.width <= 0 or msg.height <= 0 or not msg.data:
            return None
        if msg.encoding == state_pb2.ObsImage.JPEG:
            rgb = _decode_jpeg(msg.data)
            if rgb is None:
                return None
        else:  # RAW
            rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, msg.channels).copy()
        return cls(rgb=rgb[:, :, :3], tick=int(msg.tick), stamp=float(msg.stamp))


def _decode_jpeg(data: bytes) -> Optional[np.ndarray]:
    import cv2
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) if arr is not None else None


class MissionClient:
    """SUB the generator's ego image (g1/obs) and PUB/SUB the mission command (g1/mission).

    Orchestrator side:   subscribe_obs() + recv_obs()  →  publish_mission(cmd_dict)
    Generator side: subscribe_mission() + recv_mission()

    Default mission_port=5554 (NOT 5560 — that's the halo_comm discovery registry port and
    binding there collides with it; see the channel table above).
    """

    def __init__(self, node: Optional[Node] = None, host: str = "127.0.0.1",
                 obs_port: int = 5559, mission_port: int = 5554):
        self.node = node or Node("mission")
        self.host = host
        self._obs_port = int(obs_port)
        self._mission_port = int(mission_port)
        self._obs_sub = None
        self._mission_sub = None
        self._mission_pub = None

    # ── orchestrator: consume ego image ───────────────────────────────────────────
    def resolve_host(self, channel: str, port: int) -> tuple:
        """(host, port) for `channel`; resolved from the halo_comm registry (see `_resolve`) when
        host is None/"auto" (split-host runs)."""
        if self.host is None or str(self.host).lower() == "auto":
            return _resolve(self.node, channel)
        return self.host, int(port)

    def subscribe_obs(self):
        _h, _p = self.resolve_host("g1/obs", self._obs_port)
        self._obs_sub = self.node.create_subscription(
            state_pb2.ObsImage, "g1/obs", _h, _p, conflate=True, on_message=None)
        return self._obs_sub

    def recv_obs(self, timeout_ms: int = 20) -> Optional[ObsImageView]:
        """Needs subscribe_obs() first."""
        msg = self._obs_sub.recv(timeout_ms)
        return ObsImageView.from_proto(msg) if msg is not None else None

    # ── orchestrator: publish the chosen subtask ──────────────────────────────────
    def ensure_mission_pub(self):
        """Bind + announce g1/mission at orchestrator startup (the generator resolves it before it
        publishes g1/obs, so a split-host run needs it registered before the first decision)."""
        if self._mission_pub is not None:
            return self._mission_pub
        # Bind with a short retry: if a just-killed predecessor's socket is still
        # releasing (kill -> port release is not instantaneous), retry rather than
        # crashing the whole orchestrator on a transient 'Address already in use'.
        for attempt in range(10):
            try:
                self._mission_pub = self.node.create_publisher(
                    motion_pb2.MissionCommand, "g1/mission", self._mission_port)
                return self._mission_pub
            except Exception as exc:  # noqa: BLE001 — zmq EADDRINUSE etc.
                last_exc = exc
                print(f"[mission] g1/mission port {self._mission_port} busy "
                      f"(attempt {attempt + 1}/10): {exc} — retrying in 0.5s. "
                      f"If this persists, a stale orchestrator holds it: "
                      f"lsof -nP -tiTCP:{self._mission_port} | xargs kill -9", flush=True)
                time.sleep(0.5)
        raise last_exc

    def publish_mission(self, cmd: dict) -> None:
        """Publish a MissionCommand from a dict (subtask_id/subtask/prompt/reason/done/tick/stamp)."""
        self.ensure_mission_pub()
        m = motion_pb2.MissionCommand()
        m.subtask_id = int(cmd.get("subtask_id", 0))
        m.subtask = str(cmd.get("subtask", ""))
        m.prompt = str(cmd.get("prompt", ""))
        m.reason = str(cmd.get("reason", ""))
        m.why = str(cmd.get("why", ""))
        m.done = bool(cmd.get("done", False))
        m.tick = int(cmd.get("tick", 0))
        # Sender wall clock (the viewer uses it for liveness).
        m.stamp = float(cmd.get("stamp") or time.time())
        self._mission_pub.publish(m)

    # ── generator: consume the mission ───────────────────────────────────────
    def subscribe_mission(self):
        # MUST go through resolve_host: self.host is None/"auto" for a split-host run, and
        # create_subscription would build the literal address "tcp://None:5554" -- a socket that
        # connects to nothing, so the generator silently never received a subtask switch. This is
        # the generator side of the split (the orchestrator BINDS g1/mission on the other box).
        _h, _p = self.resolve_host("g1/mission", self._mission_port)
        self._mission_sub = self.node.create_subscription(
            motion_pb2.MissionCommand, "g1/mission", _h, _p, conflate=True, on_message=None)
        return self._mission_sub

    def recv_mission(self, timeout_ms: int = 20):
        """Freshest MissionCommand (proto) or None. Needs subscribe_mission() first."""
        return self._mission_sub.recv(timeout_ms)

    def close(self) -> None:
        self.node.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator plan topic (orchestrator -> viewer).
# ─────────────────────────────────────────────────────────────────────────────
class OrchestratorActivityPublisher:
    """PUB the orchestrator's checkpoint checklist on ``g1/plan`` (the live viewer's only plan
    source; a plan that stopped being published simply stops arriving).

    THREAD SAFETY. A zmq PUB socket is not thread-safe, so every publish goes through one lock,
    held only for the serialize+send.

    Every method is BEST-EFFORT: a transport error is swallowed with a one-time warning. Nothing
    in the control path depends on this topic, so a broken socket must never take the
    orchestrator down with it.
    """

    def __init__(self, node: Node, plan_port: int = 5564):
        import threading
        self.node = node
        self._ports = {"g1/plan": int(plan_port)}
        self._pubs: dict = {}
        self._lock = threading.Lock()
        self._log = lambda m: print(m, flush=True)
        self._warned: set = set()

    def _pub(self, channel: str, msg_type):
        """Lazily bind `channel`. Returns None (once-warned) if the port cannot be bound."""
        p = self._pubs.get(channel)
        if p is not None:
            return p
        if channel in self._warned:
            return None
        try:
            # sndhwm=64: a short burst of plan snapshots is never dropped by a slow subscriber.
            p = self.node.create_publisher(msg_type, channel, self._ports[channel], sndhwm=64)
            self._pubs[channel] = p
            self._log(f"[activity] publishing {channel} :{self._ports[channel]}")
            return p
        except Exception as exc:  # noqa: BLE001 — diagnostics must never break the orchestrator
            self._warned.add(channel)
            self._log(f"[activity] WARNING: could not bind {channel} on port "
                      f"{self._ports[channel]}: {type(exc).__name__}: {exc} — the live studio "
                      f"will not show this stream (the mission itself is unaffected)")
            return None

    def _send(self, channel: str, msg_type, fill: Callable) -> None:
        with self._lock:
            pub = self._pub(channel, msg_type)
            if pub is None:
                return
            try:
                m = msg_type()
                fill(m)
                pub.publish(m)
            except Exception as exc:  # noqa: BLE001
                if channel not in self._warned:
                    self._warned.add(channel)
                    self._log(f"[activity] WARNING: {channel} publish failed: "
                              f"{type(exc).__name__}: {exc} (suppressing further warnings)")

    def plan(self, snap: dict) -> None:
        """Publish one checklist snapshot (the dict `Plan.snapshot()` returns)."""
        if not isinstance(snap, dict):
            return

        def fill(m):
            m.run_id = str(snap.get("run_id") or "")
            m.version = int(snap.get("version") or 0)
            m.revision = int(snap.get("revision") or 0)
            m.stamp = float(snap.get("stamp") or 0.0)
            m.mission = str(snap.get("mission") or "")
            m.authored = bool(snap.get("authored"))
            m.reply = str(snap.get("reply") or "")
            m.current = int(snap.get("current") or 0)
            m.done = bool(snap.get("done"))
            for st in (snap.get("steps") or []):
                if not isinstance(st, dict):
                    continue
                s = m.steps.add()
                s.id = str(st.get("id") or "")
                s.subtask = str(st.get("subtask") or "")
                s.title = str(st.get("title") or "")
                s.status = str(st.get("status") or "")
        self._send("g1/plan", motion_pb2.MissionPlanMsg, fill)
