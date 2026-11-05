#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Web viewer — a multi-panel browser viewer for live runs.

Stdlib ``http.server`` serving the three.js frontend in ``static/``, the G1 URDF/STL and a
JSON API; scene-mesh helpers live in ``dataprocesslib.py``.

Run (from the repository root, with ``system/`` on PYTHONPATH)::

    # boots onto the tracker stream
    python -m halo_system.visualization.server --port 8090

Then open http://127.0.0.1:8090/ . qpos joint order == URDF actuated-joint order.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

import numpy as np

_HERE = Path(__file__).resolve().parent
_WEB = _HERE / "static"

REPO_ROOT = Path(__file__).resolve().parents[3]

from halo_system.visualization import dataprocesslib as rr   # noqa: E402  (numpy-only at import time)
from halo_system.lib.assets import resolve_asset                  # noqa: E402

DEFAULT_ROBOT_ROOT = REPO_ROOT / "assets" / "robots" / "g1"

# Operator inbox root (HALO_CACHE, default ~/.cache/halo; the orchestrator reads the same path).
_HALO_CACHE = Path(os.environ.get("HALO_CACHE", "~/.cache/halo")).expanduser()


def _scene_path(spec) -> str:
    """Local scene XML path for a scene spec (~/env path, absolute or repo-relative)."""
    return str(resolve_asset(str(spec)))

# 29 deployment-order joint names == URDF actuated order.
JOINT_NAMES = rr.DEFAULT_G1_JOINT_NAMES

def _lan_ip():
    """First private (RFC1918) address of a physical interface, for the LAN URL.

    Not the default-route trick: behind a VPN that returns the tunnel address.
    """
    import subprocess
    try:
        order = subprocess.run(["networksetup", "-listnetworkserviceorder"],
                               capture_output=True, text=True, timeout=5).stdout
    except Exception:
        order = ""
    ifaces = re.findall(r"Device: (en\d+)", order) or ["en0", "en1", "en2", "en3"]
    for i in dict.fromkeys(ifaces):
        try:
            a = subprocess.run(["ipconfig", "getifaddr", i],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            continue
        if a.startswith(("192.168.", "10.")) or re.match(r"172\.(1[6-9]|2\d|3[01])\.", a):
            return a
    return ""


class _ViewerHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that stays quiet about client disconnects."""

    def handle_error(self, request, client_address):
        # Client disconnects surface as OSError and are benign; print only real bugs.
        import traceback
        exc = sys.exc_info()[1]
        if isinstance(exc, OSError):
            return
        traceback.print_exc()


def _hostname():
    """This machine's mDNS short name, so the printed URL is DHCP-stable (<name>.local)."""
    try:
        import subprocess
        n = subprocess.run(["scutil", "--get", "LocalHostName"],
                           capture_output=True, text=True).stdout.strip()
        if n:
            return n
    except Exception:
        pass
    import socket
    return socket.gethostname().split(".")[0]


def _resolve_endpoint(registry, name, fallback_host, fallback_port):
    """(host, port) for a channel from the registry if it knows ``name``, else the fallback."""
    if registry:
        try:
            import zmq
            from halo_comm import discovery
            got = discovery.resolve(zmq.Context.instance(), discovery.registry_addr(registry), name)
            if got:
                return got
        except Exception:  # noqa: BLE001
            pass
    return fallback_host, fallback_port


class _Sub:
    """halo_comm subscriber holder: start() sets _node/_sub; stop() tears them down."""

    _sub = None
    _node = None

    def _subscribe(self, pb, msg_name, channel, conflate, node="visualization") -> bool:
        """Subscribe via a Node (registers the viewer in the halo_comm graph); errors -> self.error."""
        try:
            import importlib
            from halo_comm.node import Node
            msg_cls = getattr(importlib.import_module(f"halo_comm.msgs.{pb}"), msg_name)
            self._msg_cls = msg_cls
            self._node = Node(node)
            self._sub = self._node.create_subscription(
                msg_cls, channel, self.host, self.port, conflate=conflate, on_message=self._on_msg)
            return True
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)
            return False

    def stop(self):
        try:
            if self._node:
                self._node.shutdown()
        except Exception:  # noqa: BLE001
            pass


class LiveState(_Sub):
    """Background halo_comm subscriber for the g1/state topic. Thread-safe latest()."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5558):
        self.host, self.port = host, port
        self._latest = None
        self._n = 0
        self._t0 = None
        self._t_last = None
        self._t_made = time.time()
        self._lock = threading.Lock()
        self.channel = "g1/state"
        self.error = None

    def start(self) -> bool:
        return self._subscribe("state_pb2", "RobotState", "g1/state", True)

    def _on_msg(self, m):
        with self._lock:
            self._latest = m
            self._n += 1
            self._t_last = time.time()
            if self._t0 is None:
                self._t0 = self._t_last

    def latest(self) -> dict:
        with self._lock:
            m = self._latest; n = self._n; t0 = self._t0
        if m is None:
            return {"connected": bool(self._sub), "have": False,
                    "error": self.error, "n": n}
        jp = list(m.joint_pos)
        base = list(m.base_pos) if len(m.base_pos) == 3 else [0.0, 0.0, 0.75]
        quat = list(m.imu_quat_wxyz) if len(m.imu_quat_wxyz) == 4 else [1.0, 0, 0, 0]
        qpos = [round(v, 5) for v in (base + quat + jp)]   # 3+4+29 = 36, deploy order
        ref_qpos = None
        rm = getattr(m, "ref_motion", None)
        if rm is not None and len(rm.joints) == len(jp) and len(rm.pos) == 3 and len(rm.quat_wxyz) == 4:
            ref_qpos = [round(v, 5) for v in (list(rm.pos) + list(rm.quat_wxyz) + list(rm.joints))]
        hz = (n / (time.time() - t0)) if (t0 and time.time() > t0) else 0.0
        age = round(time.time() - self._t_last, 2) if self._t_last else None
        return {"connected": True, "have": True, "n": n, "hz": round(hz, 1), "age_s": age,
                "tracker_ready": bool(getattr(m, "tracker_ready", False)),
                "qpos": qpos, "ref_qpos": ref_qpos, "joint_names": JOINT_NAMES}


class ObsSubscriber(_Sub):
    """Background subscriber for the generator's g1/obs topic. Thread-safe latest_jpeg()."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5559, stale_s: float = 1.5):
        self.host, self.port = host, port
        self.stale_s = stale_s
        self._latest = None
        self._t = 0.0
        self._t_made = time.time()
        self._lock = threading.Lock()
        self.error = None

    def start(self) -> bool:
        return self._subscribe("state_pb2", "ObsImage", "g1/obs", True)

    def _on_msg(self, m):
        with self._lock:
            self._latest = m
            self._t = time.time()

    def latest_jpeg(self):
        """Freshest g1/obs frame as JPEG bytes (RAW is transcoded), or None if none or stale."""
        with self._lock:
            m = self._latest; t = self._t
        if m is None or (t and (time.time() - t) > self.stale_s):
            return None
        try:
            if m.encoding == self._msg_cls.JPEG:
                return bytes(m.data)
            arr = np.frombuffer(m.data, dtype=np.uint8).reshape(
                m.height, m.width, m.channels)[:, :, :3]
            from PIL import Image as _Img
            import io as _io
            buf = _io.BytesIO()
            _Img.fromarray(arr).save(buf, format="JPEG", quality=80)
            return buf.getvalue()
        except Exception:  # noqa: BLE001
            return None


class DepthSubscriber(_Sub):
    """Background subscriber for the tracker's g1/depth topic (float32 metres). Thread-safe."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5553, stale_s: float = 2.0):
        self.host, self.port = host, port
        self.stale_s = stale_s
        self._latest = None
        self._t = 0.0
        self._lock = threading.Lock()
        self._png = None          # (tick, png bytes) cache
        self._t_made = time.time()
        self.error = None

    def start(self) -> bool:
        return self._subscribe("state_pb2", "DepthImage", "g1/depth", True, node="visualization_depth")

    def _on_msg(self, m):
        with self._lock:
            self._latest = m
            self._t = time.time()

    def latest_png(self, near: float = 0.2, far: float = 2.5):
        """Freshest depth frame as a colour-mapped PNG (near = warm, far = dark), or None."""
        with self._lock:
            m = self._latest; t = self._t
        if m is None or (time.time() - t) > self.stale_s or m.width <= 0 or m.height <= 0:
            return None
        key = m.tick
        if self._png is not None and self._png[0] == key:
            return self._png[1]
        try:
            d = np.frombuffer(m.data, dtype="<f4").reshape(m.height, m.width)
            x = 1.0 - (np.clip(np.nan_to_num(d, nan=far, posinf=far), near, far) - near) / (far - near)
            # simple "turbo-like" ramp: dark blue (far) -> cyan -> yellow -> red (near)
            r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
            g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
            b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
            rgb = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
            from PIL import Image as _Img
            import io as _io
            img = _Img.fromarray(rgb)
            out_w = m.width * 6
            img = img.resize((out_w, max(1, round(out_w * m.height / m.width))), _Img.NEAREST)
            buf = _io.BytesIO()
            img.save(buf, format="PNG")
            png = buf.getvalue()
            self._png = (key, png)
            return png
        except Exception:  # noqa: BLE001
            return None


_OBS_REPROBE_S = 5.0
_DEPTH_LOCK = threading.Lock()
_OBS_LOCK = threading.Lock()


def _ensure_sub(attr, lock, factory, resolver, label):
    """Ensure ViewerHandler.<attr> subscribes the current endpoint of `label`.

    A feed silent for _OBS_REPROBE_S is re-resolved (`resolver()` -> (host, port)) and rebuilt only
    if the endpoint changed (ZMQ reconnects to an unchanged one)."""
    cls = ViewerHandler
    with lock:
        s = getattr(cls, attr)
        now = time.time()
        if s is not None and s._sub is not None and (now - (s._t or s._t_made)) <= _OBS_REPROBE_S:
            return s
        if s is not None and now - getattr(s, "_t_probe", 0.0) < _OBS_REPROBE_S:
            return s
        host, port = resolver()
        if s is not None and (s.host, s.port) != (host, port):
            print(f"[visualization] {label} endpoint changed {s.host}:{s.port} -> {host}:{port}; "
                  f"resubscribing", flush=True)
            s.stop(); s = None
        if s is None:
            s = factory(host, port)
            setattr(cls, attr, s)
        if s._sub is None:
            s.start()
        s._t_probe = now
        return s


def _ensure_depth():
    """Ensure ViewerHandler._depth subscribes g1/depth (registry, else --live_host)."""
    cls = ViewerHandler
    return _ensure_sub("_depth", _DEPTH_LOCK, DepthSubscriber,
                       lambda: _resolve_endpoint(cls.registry, "g1/depth", cls.live_host,
                                                 5553), "g1/depth")


def _ensure_obs():
    """Ensure ViewerHandler._obs subscribes g1/obs: --obs_host, else the registry, else
    --live_host."""
    cls = ViewerHandler

    def _resolver():
        if cls.obs_host:
            return cls.obs_host, cls.obs_port
        return _resolve_endpoint(cls.registry, "g1/obs", cls.live_host, cls.obs_port)
    return _ensure_sub("_obs", _OBS_LOCK, ObsSubscriber, _resolver, "g1/obs")


_LIVE_REPROBE_S = 3.0          # g1/state is 50 Hz
_LIVE_LOCK = threading.Lock()


def _ensure_live():
    """Ensure ViewerHandler._live receives the current g1/state stream.

    A feed silent for _LIVE_REPROBE_S is re-resolved and always rebuilt (a silent socket may not
    reconnect by itself); at most one rebuild per _LIVE_REPROBE_S.
    """
    cls = ViewerHandler
    with _LIVE_LOCK:
        lv = cls._live
        now = time.time()
        if lv is not None and lv._sub is not None:
            last = lv._t_last or lv._t_made
            if now - last <= _LIVE_REPROBE_S:
                return lv
            if now - getattr(lv, "_t_probe", lv._t_made) < _LIVE_REPROBE_S:
                return lv
        host, port = _resolve_endpoint(cls.registry, "g1/state", cls.live_host, cls.live_port)
        if lv is not None:
            print(f"[visualization] g1/state silent {now - (lv._t_last or lv._t_made):.1f}s "
                  f"({lv.host}:{lv.port}) -> resubscribing at {host}:{port}", flush=True)
            lv.stop()
        lv = LiveState(host, port)
        lv.start()
        lv._t_probe = now
        cls._live = lv
        return lv


class LiveMission(_Sub):
    """Background subscriber for g1/mission keeping a deduped decision history (not conflated)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5554, keep: int = 400):
        self.host, self.port = host, port
        self.keep = keep
        self._hist = []              # oldest first
        self._seen = set()           # (tick, subtask) dedup
        self._n = 0
        self._switches = 0           # subtask changes, not re-decisions
        self._decisions = []         # step-decision lines (`why` changes), newest last
        self._last_why = ""
        self._lock = threading.Lock()
        self.error = None

    def start(self) -> bool:
        return self._subscribe("motion_pb2", "MissionCommand", "g1/mission", False)

    def _on_msg(self, m):
        d = {"subtask_id": int(m.subtask_id), "subtask": str(m.subtask),
             "prompt": str(m.prompt), "reason": str(m.reason),
             "why": str(m.why),
             "done": bool(m.done), "tick": int(m.tick),
             "stamp": float(m.stamp), "t_recv": time.time()}
        with self._lock:
            w = d["why"].strip()
            if w and w != self._last_why:          # a new decision line from the orchestrator
                self._last_why = w
                self._decisions.append({"stamp": d["stamp"], "say": w,
                                        "subtask": d["subtask"]})
                self._decisions = self._decisions[-20:]
            key = (d["tick"], d["subtask"])
            if key in self._seen:
                return
            self._seen.add(key)
            # The orchestrator republishes holds every tick; refresh the last identical hold in place.
            if (d["subtask"] in ("hold", "stand") and self._hist
                    and self._hist[-1]["subtask"] == d["subtask"]
                    and self._hist[-1].get("reason") == d["reason"]):
                self._hist[-1]["tick"] = d["tick"]; self._hist[-1]["stamp"] = d["stamp"]
                return
            prev = self._hist[-1]["subtask"] if self._hist else None
            d["switch"] = (prev is not None and d["subtask"] != prev)
            if d["switch"]:
                self._switches += 1
            d["seq"] = self._n            # monotonic id for the browser to diff on
            self._hist.append(d)
            self._n += 1
            if len(self._hist) > self.keep:
                self._hist = self._hist[-self.keep:]

    # Liveness = a recent MissionCommand.stamp (a SUB connect succeeds with no publisher). The orchestrator
    # cannot publish while blocked on a VLM call, so keep this above the orchestrator's 45 s VLM timeout.
    LIVENESS_WINDOW_S = 60.0

    def snapshot(self) -> dict:
        with self._lock:
            hist = list(self._hist)
            dec = list(self._decisions)
            n, sw = self._n, self._switches
        last_stamp = hist[-1]["stamp"] if hist else None
        heartbeat_age_s = (time.time() - last_stamp) if last_stamp is not None else None
        connected = heartbeat_age_s is not None and heartbeat_age_s < self.LIVENESS_WINDOW_S
        # `have` = history non-empty; liveness is `connected` / `heartbeat_age_s`.
        return {"connected": connected, "heartbeat_age_s": heartbeat_age_s,
                "have": bool(hist),
                "error": self.error, "n": n, "switches": sw,
                "current": (hist[-1] if hist else None), "history": hist,
                "decisions": dec}


class LivePlan(_Sub):
    """Background subscriber for g1/plan, the live checkpoint checklist.

    Conflated: only the freshest plan matters (the orchestrator republishes it ~1 Hz).
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5564):
        self.host, self.port = host, port
        self._plan = None
        self._t = 0.0
        self._lock = threading.Lock()
        self.error = None

    def start(self) -> bool:
        return self._subscribe("motion_pb2", "MissionPlanMsg", "g1/plan", True)

    def _on_msg(self, m):
        # Subset of Plan.snapshot() the live viewer uses.
        plan = {
            "run_id": str(m.run_id),
            "version": int(m.version), "revision": int(m.revision), "stamp": float(m.stamp),
            "mission": str(m.mission),
            "authored": bool(m.authored), "reply": str(m.reply),
            "current": int(m.current), "done": bool(m.done),
            "steps": [{"id": str(s.id), "subtask": str(s.subtask), "title": str(s.title),
                       "status": str(s.status)}
                      for s in m.steps],
        }
        with self._lock:
            self._plan, self._t = plan, time.time()

    def snapshot(self) -> dict:
        """``{"plan", "run_id", "age_s", "source"}``; ``age_s`` is measured from receipt."""
        with self._lock:
            plan, t = self._plan, self._t
        if plan is None:
            return {"plan": None, "run_id": None, "age_s": None, "source": "g1/plan"}
        return {"plan": plan, "run_id": plan.get("run_id"),
                "age_s": max(0.0, time.time() - t), "source": "g1/plan"}


def _rows_to_qpos(pos: np.ndarray, quat_wxyz: np.ndarray, joints: np.ndarray) -> list:
    """[T,3]+[T,4]+[T,29] -> [[x,y,z, qw,qx,qy,qz, j0..j28], ...] (JSON-ready)."""
    T = len(pos)
    out = np.zeros((T, 7 + joints.shape[1]), dtype=np.float32)
    out[:, 0:3] = pos
    out[:, 3:7] = quat_wxyz
    out[:, 7:] = joints
    return np.round(out, 5).tolist()


_LIVE_PLAN_CACHE = {"key": None, "out": None}


def _live_plan(run_dir):
    """Newest plan of a running launcher in world frame, for the LIVE plan overlay.

    <run>/live_plan.npz (overwritten by the generator): plan [T,36] = pos (relative to frame 0,
    yaw-only gauge), quat XYZW, 29 joints; t0 = [stamp, pos, quat WXYZ] odom pose at planning;
    idx = plan counter. World = t0 position + t0 yaw applied to the plan.
    """
    if not run_dir:
        return None
    import math
    p = Path(run_dir) / "live_plan.npz"
    try:
        st = p.stat()
    except OSError:
        return None
    key = (str(p), st.st_mtime_ns)
    if _LIVE_PLAN_CACHE["key"] == key:
        return _LIVE_PLAN_CACHE["out"]
    try:
        with np.load(p) as z:
            P = np.asarray(z["plan"], float)
            t0 = np.asarray(z["t0"], float).ravel()
            best = int(z["idx"])
    except Exception:  # noqa: BLE001 -- half-replaced file
        return _LIVE_PLAN_CACHE["out"]
    if P.ndim != 2 or P.shape[0] < 2 or P.shape[1] < 36 or t0.size < 8:
        return _LIVE_PLAN_CACHE["out"]
    tw, tx, ty, tz = t0[4:8] / max(np.linalg.norm(t0[4:8]), 1e-9)
    yaw = math.atan2(2 * (tw * tz + tx * ty), 1 - 2 * (ty * ty + tz * tz))
    c, s_ = math.cos(yaw), math.sin(yaw)
    rel = P[:, 0:3] - P[0, 0:3]
    pos = np.stack([t0[1] + c * rel[:, 0] - s_ * rel[:, 1],
                    t0[2] + s_ * rel[:, 0] + c * rel[:, 1],
                    t0[3] + rel[:, 2]], axis=1)
    qx, qy, qz, qw = P[:, 3], P[:, 4], P[:, 5], P[:, 6]
    hw, hz = math.cos(yaw / 2), math.sin(yaw / 2)          # Rz(yaw) (wxyz) * q_rel
    quat = np.stack([hw * qw - hz * qz, hw * qx - hz * qy, hw * qy + hz * qx, hw * qz + hz * qw], axis=1)
    out = {"idx": best, "stamp": float(t0[0]), "fps": 30.0,
           "qpos": _rows_to_qpos(pos, quat, P[:, 7:36])}
    _LIVE_PLAN_CACHE.update(key=key, out=out)
    return out


def _encode_texture(pil):
    """(bytes, mime) for a scene texture: JPEG (much smaller for photo scans), PNG if it has alpha."""
    import io as _io
    buf = _io.BytesIO()
    if "A" in pil.mode:
        pil.save(buf, format="PNG")
        return buf.getvalue(), "image/png"
    pil.convert("RGB").save(buf, format="JPEG", quality=92, optimize=False, subsampling=0)
    return buf.getvalue(), "image/jpeg"


def _scene_fingerprint(xml: str, hidden_bodies=()) -> str:
    """Hash of the XML, every referenced asset (size, mtime) and the knobs that shape the payload."""
    import hashlib
    h = hashlib.sha256()
    xp = Path(xml).expanduser().resolve()
    h.update(str(xp).encode())
    def stamp(f: Path):
        try:
            st = f.stat()
            h.update(f"{f}|{st.st_size}|{int(st.st_mtime)}".encode())
        except OSError:
            h.update(f"{f}|missing".encode())
    stamp(xp)
    try:
        txt = xp.read_text()
    except OSError:
        txt = ""
    # honour the compiler's meshdir/texturedir
    mesh_dir = (re.search(r'meshdir\s*=\s*"([^"]+)"', txt) or [None, ""])[1]
    tex_dir = (re.search(r'texturedir\s*=\s*"([^"]+)"', txt) or [None, ""])[1]
    for ref in re.findall(r'file\s*=\s*"([^"]+)"', txt):
        cands = [xp.parent / ref]
        if mesh_dir:
            cands.append(xp.parent / mesh_dir / ref)
        if tex_dir:
            cands.append(xp.parent / tex_dir / ref)
        hit = next((c for c in cands if c.exists()), None)
        stamp(hit) if hit is not None else h.update(ref.encode())
    h.update(f"|hidden={','.join(hidden_bodies or ())}".encode())
    h.update(b"|v5-freebodies")                    # bump when the payload FORMAT changes
    return h.hexdigest()[:20]


def _scene_cache_path(fp: str) -> Path:
    d = _HALO_CACHE / "visualization_scene_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"scene_{fp}.json.gz"


def _scene_json_bytes(scene_xml: str, hidden_bodies=()):
    """``(raw, gzipped)`` scene payload bytes, cached on disk by _scene_fingerprint (cold builds are slow)."""
    import gzip
    fp = _scene_fingerprint(scene_xml, hidden_bodies)
    cp = _scene_cache_path(fp)
    if cp.exists():
        t0 = time.perf_counter()
        gz = cp.read_bytes()
        raw = gzip.decompress(gz)
        print(f"[visualization] scene from cache ({len(raw)/1e6:.1f} MB payload, "
              f"{(time.perf_counter()-t0)*1e3:.0f} ms) {cp.name}", flush=True)
        return raw, gz
    t0 = time.perf_counter()
    sj = _scene_json(scene_xml, hidden_bodies)
    raw = json.dumps(sj).encode()
    print(f"[visualization] scene built in {time.perf_counter()-t0:.1f} s "
          f"({len(raw)/1e6:.1f} MB) -> cached as {cp.name}", flush=True)
    gz = gzip.compress(raw, 6)
    try:
        tmp = cp.with_suffix(".tmp")
        tmp.write_bytes(gz)
        os.replace(tmp, cp)
    except OSError as exc:
        print(f"[visualization] could not write the scene cache: {exc}", flush=True)
    return raw, gz


def _scene_json(scene_xml: str, hidden_bodies=()) -> dict:
    """Non-robot geometry of a MuJoCo scene XML as three.js JSON meshes (textured where possible).
    Geoms of `hidden_bodies` (task YAML hide_bodies, hidden in the sim) are left out. Free bodies
    (e.g. the box) go body-local with their pose, so /api/objects can move them; the rest is
    world-baked."""
    import base64
    import mujoco
    xml = str(Path(scene_xml).expanduser().resolve())
    from halo_system.lib.assets import load_scene_model
    m = load_scene_model(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    skip = rr._robot_geom_ids(m)
    hidden = set()
    for n in hidden_bodies or ():
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, str(n))
        if b < 0:
            print(f"[visualization] hide_bodies {n!r} not in the scene", flush=True)
        else:
            hidden.add(int(b))
    pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    robot_root = int(m.body_rootid[pel]) if pel >= 0 else -1
    free_bodies = {int(m.jnt_bodyid[j]) for j in range(m.njnt)
                   if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                   and int(m.jnt_bodyid[j]) != robot_root}

    def _add_mesh(entry, v, bid):
        """Append a mesh entry; verts go body-local (+ the body pose) on a free body."""
        if bid in free_bodies:
            Rb = np.asarray(d.xmat[bid], float).reshape(3, 3)
            tb = np.asarray(d.xpos[bid], float)
            w, x, y, z = (float(q) for q in d.xquat[bid])
            v = (v - tb) @ Rb
            entry["body"] = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            entry["pos"] = [float(c) for c in tb]
            entry["quat"] = [x, y, z, w]                 # three.js order
        entry["v"] = np.round(v, 4).astype(np.float32).ravel().tolist()
        meshes.append(entry)

    meshes = []
    for g in range(m.ngeom):
        if g in skip:
            continue
        if int(m.geom_bodyid[g]) in hidden:
            continue
        if float(m.geom_rgba[g][3]) <= 0.0:   # collision-only / invisible
            continue
        # Skip collision proxies, identified by the `_col` / `_collision` name suffix.
        _gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if (_gname.endswith("_col") or _gname.endswith("_collision")):
            continue
        R = np.asarray(d.geom_xmat[g], float).reshape(3, 3)
        t = np.asarray(d.geom_xpos[g], float)

        # 1) textured mesh, rendered unlit.
        textured = None
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            try:
                textured = rr._build_textured_mesh(m, g)
            except Exception:  # noqa: BLE001
                textured = None
        if textured is not None and getattr(textured.visual, "uv", None) is not None:
            vis = textured.visual
            pil = getattr(getattr(vis, "material", None), "image", None) or getattr(vis, "image", None)
            if pil is None:
                textured = None
            else:
                v = (np.asarray(textured.vertices, float) @ R.T) + t
                uv = np.asarray(vis.uv, np.float32)
                fidx = np.asarray(textured.faces, np.int64)
                buf, mime = _encode_texture(pil)
                _add_mesh({
                    "v": None,                          # keeps "v" the first JSON key
                    "f": fidx.astype(np.int32).ravel().tolist(),
                    "uv": uv.ravel().tolist(),
                    "tex": f"data:{mime};base64," + base64.b64encode(buf).decode(),
                    "unlit": True,
                }, v, int(m.geom_bodyid[g]))
                continue

        # 2) flat-colored fallback.
        try:
            tm = rr._geom_to_trimesh(m, g)
        except Exception:  # noqa: BLE001
            tm = None
        if tm is None or len(tm.faces) == 0:
            continue
        v = (np.asarray(tm.vertices, float) @ R.T) + t
        fidx = np.asarray(tm.faces, np.int64)
        rgba = m.geom_rgba[g]
        _add_mesh({
            "f": fidx.astype(np.int32).ravel().tolist(),
            "color": [float(rgba[0]), float(rgba[1]), float(rgba[2])],
            "opacity": float(np.clip(rgba[3], 0.0, 1.0)),
        }, v, int(m.geom_bodyid[g]))
    return {"meshes": meshes, "n": len(meshes)}


_SCALED = {"key": None, "out": None}
_SCALED_LOCK = threading.Lock()


def _scaled_jpeg(jpg: bytes, w: int) -> bytes:
    """`jpg` resized to width `w` (q80); the input if already that small or cv2 is missing."""
    w = max(160, min(1920, int(w) // 32 * 32))       # snap so small resizes reuse the cache
    key = (len(jpg), hash(jpg), w)
    with _SCALED_LOCK:
        if _SCALED["key"] == key:
            return _SCALED["out"]
    out = jpg
    try:
        import cv2
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is not None and img.shape[1] > w:
            h = max(1, round(img.shape[0] * w / img.shape[1]))
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                out = enc.tobytes()
    except Exception:  # noqa: BLE001
        out = jpg
    with _SCALED_LOCK:
        _SCALED.update(key=key, out=out)
    return out


class ViewerHandler(BaseHTTPRequestHandler):
    # Keep-alive for the high-rate live polls; every response must send Content-Length.
    protocol_version = "HTTP/1.1"
    timeout = 60          # idle keep-alive timeout (one thread each)
    # class attrs set in main()
    robot_root: Path = DEFAULT_ROBOT_ROOT
    scene_xml: str | None = None
    # (xml, serialized payload bytes) for /api/scene
    _scene_cache: tuple | None = None
    live_host: str = "127.0.0.1"
    live_port: int = 5558
    obs_port: int = 5559
    # Pins the g1/obs host, skipping registry/live_host resolution for obs only. None = resolve.
    obs_host: str | None = None
    mission_port: int = 5554               # g1/mission
    plan_port: int = 5564                  # g1/plan
    hidden_bodies: list = []   # task YAML hide_bodies: left out of the scene payload (hidden in the sim at startup)
    human_input_file: str = str(_HALO_CACHE / "mission_human_input.jsonl")
    follow_default: bool = False           # nobody drives the view: single-task (non-agentic) run
    live_run_dir: str | None = None        # the running launcher's run folder (live_plan.npz)
    registry: str | None = None            # registry host:port (or HALO_COMM_REGISTRY)
    _live = None
    _obs = None
    _depth = None
    _mission = None
    _plan = None

    def log_message(self, *a):
        pass

    def _send(self, code, body: bytes, ctype="application/octet-stream", gz: bytes | None = None):
        """Send a response, gzipping large bodies; `gz` = the same body already gzipped, if available."""
        try:
            enc = None
            ae = self.headers.get("Accept-Encoding", "") if self.headers else ""
            if "gzip" in ae.lower() and gz is not None:
                body, enc = gz, "gzip"
            elif len(body) > 262144 and "gzip" in ae.lower():
                import gzip as _gz
                body = _gz.compress(body, 6)
                enc = "gzip"
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            if enc:
                self.send_header("Content-Encoding", enc)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except OSError:
            # client went away mid-response
            self.close_connection = True

    def _send_image(self, body: bytes, ctype, extra=()):
        """Live image frame: never gzipped (already compressed; saves per-frame CPU)."""
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _send_file(self, path: Path, ctype=None):
        if not path.is_file():
            self._send(404, b"not found", "text/plain"); return
        ctype = ctype or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self._send(200, path.read_bytes(), ctype)

    def _read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length", 0) or 0)
        return self.rfile.read(n) if n else b""

    def _append_inbox(self, rec):
        """One JSON line to the orchestrator's operator inbox."""
        path = Path(self.human_input_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", buffering=1) as fh:
            fh.write(json.dumps(rec) + "\n")

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        # Not every endpoint drains its body, which would corrupt a kept-alive connection.
        self.close_connection = True
        u = urlparse(self.path)
        p = unquote(u.path)

        if p == "/api/mission_input":
            # Operator text -> the orchestrator's inbox.
            try:
                text = str(json.loads(self._read_body() or b"{}").get("text", "")).strip()
                if not text:
                    return self._send_json({"error": "empty message"}, 400)
                self._append_inbox({"text": text, "stamp": time.time()})
                return self._send_json({"ok": True, "text": text})
            except Exception as e:  # noqa: BLE001
                return self._send_json({"error": repr(e)}, 500)

        return self._send_json({"error": "unknown endpoint"}, 404)

    def do_GET(self):
        u = urlparse(self.path)
        p = unquote(u.path)
        q = parse_qs(u.query)

        if p in ("/", "/index.html"):
            # Version app.js by mtime so an edited file is refetched on reload.
            html = (_WEB / "index.html").read_text()
            ver = int((_WEB / "app.js").stat().st_mtime)
            html = html.replace('src="/app.js"', f'src="/app.js?v={ver}"')
            return self._send(200, html.encode(), "text/html")
        if p == "/app.js":
            return self._send_file(_WEB / "app.js", "text/javascript")
        if p in ("/favicon.svg", "/favicon.png", "/favicon.ico"):
            fn = p.lstrip("/")
            ct = ("image/svg+xml" if fn.endswith(".svg")
                  else "image/x-icon" if fn.endswith(".ico") else "image/png")
            return self._send_file(_WEB / fn, ct)
        if p.startswith("/vendor/"):
            return self._send_file(_WEB / p.lstrip("/"))
        if p.startswith("/robot/"):
            return self._send_file(self.robot_root / p[len("/robot/"):])

        if p == "/api/live/start":
            # Each channel is resolved independently via the registry, else --live_host.
            cls = ViewerHandler
            reg = cls.registry
            fb = cls.live_host
            state_host, state_port = _resolve_endpoint(reg, "g1/state", fb, cls.live_port)

            if cls._live is not None and (cls._live.host != state_host or
                                          cls._live.port != state_port):
                cls._live.stop(); cls._live = None
            if cls._live is None:
                cls._live = LiveState(state_host, state_port)
            ok = bool(cls._live._sub) or cls._live.start()
            _ensure_obs()
            mis_host, mis_port = _resolve_endpoint(reg, "g1/mission", fb, cls.mission_port)
            if cls._mission is not None and (cls._mission.host != mis_host or
                                             cls._mission.port != mis_port):
                cls._mission.stop(); cls._mission = None
            if cls._mission is None:
                cls._mission = LiveMission(mis_host, mis_port)
            if cls._mission._sub is None:
                cls._mission.start()
            plan_host, plan_port = _resolve_endpoint(reg, "g1/plan", fb, cls.plan_port)
            if cls._plan is not None and (cls._plan.host != plan_host or
                                          cls._plan.port != plan_port):
                cls._plan.stop(); cls._plan = None
            if cls._plan is None:
                cls._plan = LivePlan(plan_host, plan_port)
            if cls._plan._sub is None:
                cls._plan.start()
            return self._send_json({"ok": ok, "host": cls._live.host,
                                    "follow_default": bool(ViewerHandler.follow_default),
                                    "port": cls._live.port,
                                    # joint order for posing the robot
                                    "joint_names": list(JOINT_NAMES),
                                    "channel": cls._live.channel, "error": cls._live.error})

        if p == "/api/live_mission":
            cls = ViewerHandler
            if cls._mission is None or cls._plan is None:
                return self._send_json({"error": "LIVE not started"})
            snap = cls._mission.snapshot()
            snap["plan"] = cls._plan.snapshot()
            # T1 reasoning for the Thinking box comes from the plan message (not on g1/mission).
            _p = (snap.get("plan") or {}).get("plan") or {}
            snap["deliberation"] = {
                "plan_version": _p.get("version"),
                "authored": bool(_p.get("authored")),
                "reply": _p.get("reply") or "",
            } if _p else None
            return self._send_json(snap)

        if p == "/api/live_plan":
            return self._send_json({"now": time.time(),
                                    "plan": _live_plan(ViewerHandler.live_run_dir)})

        if p == "/api/live":
            return self._send_json(_ensure_live().latest())

        if p == "/api/objects":
            # Live poses of the free bodies, written by the sim tracker to HALO_OBJ_FILE (one
            # 'name x y z qw qx qy qz [hidden]' line each); 204 when nothing publishes.
            path = Path(os.environ.get("HALO_OBJ_FILE", "/tmp/halo_objects.txt"))
            out = {}
            try:
                if time.time() - path.stat().st_mtime < 5.0:
                    for line in path.read_text().splitlines():
                        f = line.split()
                        if len(f) in (8, 9):
                            out[f[0]] = {"pos": [float(c) for c in f[1:4]],
                                         "quat": [float(f[5]), float(f[6]), float(f[7]), float(f[4])],
                                         "hidden": len(f) == 9 and f[8] == "1"}
            except (OSError, ValueError):
                pass
            if not out:
                return self._send(204, b"", "text/plain")
            return self._send_json({"objects": out})

        if p == "/api/live_depth":
            # Depth camera from the tracker's g1/depth (only when a depth policy runs); 204 if none.
            d = _ensure_depth()
            png = d.latest_png() if d is not None else None
            if png is None:
                return self._send(204, b"", "text/plain")
            return self._send_image(png, "image/png")

        if p == "/api/live_image":
            # Ego image from the generator's g1/obs; subscribed on demand so an open tab recovers.
            cls = ViewerHandler
            _ensure_obs()
            jpg = cls._obs.latest_jpeg() if cls._obs is not None else None
            if jpg is None:
                if cls._obs is None:
                    return self._send(503, b"", "text/plain")
                return self._send(204, b"", "text/plain")
            # ?w=N: downscale to the panel's pixel width to save bandwidth.
            try:
                w = int(q.get("w", ["0"])[0])
            except ValueError:
                w = 0
            if w > 0:
                jpg = _scaled_jpeg(jpg, w)
            return self._send_image(jpg, "image/jpeg", (("X-Obs-Source", "g1/obs"),))

        if p == "/api/scene":
            xml = self.scene_xml
            if not xml:
                return self._send_json({"meshes": [], "n": 0})
            try:
                cls = ViewerHandler
                hb = tuple(ViewerHandler.hidden_bodies)
                key = (xml, hb)
                if cls._scene_cache is None or cls._scene_cache[0] != key:
                    raw, gz = _scene_json_bytes(_scene_path(xml), hb)
                    cls._scene_cache = (key, raw, gz)
                return self._send(200, cls._scene_cache[1], "application/json",
                                  gz=cls._scene_cache[2])
            except Exception as e:  # noqa: BLE001
                return self._send_json({"error": repr(e), "meshes": []}, 500)

        return self._send(404, b"not found", "text/plain")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no_scene", action="store_true",
                    help="Do not load any scene into the viewport (robot only), even if "
                         "--task_config names one. run_real --web uses this: "
                         "the real robot is not in the sim scene, and loading the scan costs seconds "
                         "and tens of MB per browser.")
    ap.add_argument("--task_config", dest="mission_config", default=None,
                    help="The SAME configs/*.yaml the launcher is running. Reads `scene_xml` from it "
                         "so the viewport shows the scene the robot is actually in, without "
                         "repeating the path here.")
    ap.add_argument("--live_host", default="127.0.0.1",
                    help="tracker host for Live mode (g1/state PUB)")
    ap.add_argument("--live_port", type=int, default=5558,
                    help="g1/state PUB port (default 5558)")
    ap.add_argument("--obs_port", type=int, default=5559,
                    help="g1/obs ego-image PUB port (generator-published; default 5559)")
    ap.add_argument("--obs_host", default=None,
                    help="PIN the ego-image (g1/obs) subscription to this host, skipping registry/"
                         "--live_host resolution for obs ONLY. Use 127.0.0.1 in the split deployment "
                         "(generator on this host, tracker g1/state on the robot computer): keeps the "
                         "image working when the registry advertises g1/obs at a stale LAN IP or "
                         "--live_host points at the robot. Default: resolve like the other channels.")
    ap.add_argument("--live_run_dir", default=None,
                    help="Run folder of the launcher being watched LIVE (the generator's live_plan.npz "
                         "feeds the plan overlay). run_sim / run_real --web set it.")
    ap.add_argument("--mission_port", type=int, default=5554,
                    help="g1/mission PUB port (VLM orchestrator decisions; default 5554)")
    ap.add_argument("--plan_port", type=int, default=5564,
                    help="g1/plan PUB port — the checkpoint checklist (default 5564)")
    ap.add_argument("--registry", default=None,
                    help="master registry host:port for multi-host discovery "
                         "(default HALO_COMM_REGISTRY env). When set, LIVE resolves "
                         "g1/state + g1/obs at their real hosts automatically.")
    ap.add_argument("--port", type=int, default=8090)
    # All interfaces by default so another machine on the LAN can connect; --host 127.0.0.1 for local-only.
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args(argv)

    # --task_config supplies scene_xml and hide_bodies.
    _mc = {}
    if args.mission_config:
        _mc_path = Path(args.mission_config).expanduser()
        if not _mc_path.exists():
            raise SystemExit(f"task_config not found: {_mc_path}")
        try:
            import yaml as _yaml
            _mc = _yaml.safe_load(_mc_path.read_text()) or {}
        except Exception as exc:                                      # noqa: BLE001
            raise SystemExit(f"could not read {_mc_path}: {exc}")
        ViewerHandler.follow_default = not bool(_mc.get("agentic"))

        def _names(v):
            if v is None:
                return []
            return [str(x).strip() for x in (v if isinstance(v, list) else str(v).split(","))
                    if str(x).strip()]
        ViewerHandler.hidden_bodies = _names(_mc.get("hide_bodies"))
    if args.live_run_dir:
        ViewerHandler.live_run_dir = str(Path(args.live_run_dir).expanduser().resolve())
    scene_xml = None
    if args.no_scene:
        print("[visualization] --no_scene: viewport shows the robot only", flush=True)
    elif args.mission_config:
        _sx = _mc.get("scene_xml")
        if _sx:
            _sxp = Path(_scene_path(_sx))
            if not _sxp.exists():
                raise SystemExit(f"scene_xml from {_mc_path.name} not found: {_sxp}")
            scene_xml = str(_sxp)
            print(f"[visualization] scene from {_mc_path.name}: {_sxp.name}", flush=True)
        else:
            print(f"[visualization] {_mc_path.name} has no scene_xml (real-robot mission?) — "
                  f"running without a viewport scene", flush=True)

    ViewerHandler.scene_xml = _scene_path(scene_xml) if scene_xml else None
    ViewerHandler.live_host = args.live_host
    ViewerHandler.live_port = args.live_port
    ViewerHandler.obs_port = args.obs_port
    ViewerHandler.obs_host = args.obs_host
    ViewerHandler.mission_port = args.mission_port
    ViewerHandler.plan_port = args.plan_port
    ViewerHandler.registry = args.registry or os.environ.get("HALO_COMM_REGISTRY")

    print("[visualization] LIVE: booting onto the tracker stream")

    srv = _ViewerHTTPServer((args.host, args.port), ViewerHandler)

    # Print usable URLs instead of 0.0.0.0.
    if args.host in ("0.0.0.0", "::"):
        _lan = _lan_ip()
        print(f"[visualization] serving HTTP on ALL interfaces (port {args.port})  (Ctrl-C to stop)")
        print(f"[visualization]   local  : http://localhost:{args.port}/")
        if _lan:
            print(f"[visualization]   LAN    : http://{_lan}:{args.port}/")
        # mDNS may not advertise the LAN IP (e.g. behind a VPN), so it is listed last.
        print(f"[visualization]   (mdns) : http://{_hostname()}.local:{args.port}/  "
              f"— only if mDNS advertises your LAN IP")
    else:
        print(f"[visualization] serving http://{args.host}:{args.port}/  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[visualization] bye")


if __name__ == "__main__":
    main()
