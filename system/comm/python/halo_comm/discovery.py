#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Master registry for halo_comm discovery (see msgs/discovery.proto).

ZMQ has no discovery: a channel is identified on the wire only by its TCP port,
so consumers must be told where each provider binds. When providers live on
different hosts (tracker on the robot NX, generator on a Mac mini) a single
--host can't reach them all. Discovery is always on, through one registry
(ROS2 "Discovery Server" model):

  - providers (nodes that BIND a service/publisher) REGISTER their real
    host:port with one well-known registry, then heartbeat to stay alive;
  - consumers query the registry and connect to each channel at its actual host.

The registry's OWN address is CONFIGURED (HALO_COMM_REGISTRY=host:port, default
localhost), never discovered — no UDP multicast (which wouldn't cross a NAT/hotspot anyway). It is
discovery-only: the ZMQ data path is brokerless, so a registry outage never
drops live connections. Everything here is best-effort — discovery must never
crash or block a node.

Layers:
  * ``RegistryTable`` — pure in-memory endpoint table with TTL expiry (no socket;
    used by the daemon in registry.py and by the tests).
  * client helpers — ``registry_addr``, ``advertise_host``, ``register``,
    ``unregister``, ``list_endpoints``, ``resolve`` — thin REQ/REP calls a Node
    (or CLI) makes to the registry.
"""
from __future__ import annotations

import os
import socket
import time
from typing import Optional

from .msgs import discovery_pb2

DEFAULT_REGISTRY_PORT = 5560
DEFAULT_TTL_S = 10.0
REGISTRY_CHANNEL = "halo_comm/registry"    # the registry's own REQ/REP channel name
DEFAULT_REGISTRY_HOST = "127.0.0.1"     # single-host default; export HALO_COMM_REGISTRY for multi-host


# --------------------------------------------------------------------------- #
# Config resolution
# --------------------------------------------------------------------------- #
def registry_addr(override: Optional[str] = None) -> tuple[str, int]:
    """(host, port) of the master registry. Discovery is ALWAYS ON: this never
    returns None. Priority: explicit ``override`` > ``HALO_COMM_REGISTRY`` env >
    the localhost default. For multi-host, ``export HALO_COMM_REGISTRY=<mini-ip>:5560``
    in your shell rc on every machine.
    """
    v = override or os.environ.get("HALO_COMM_REGISTRY")
    if v:
        v = v.strip()
    if not v:
        return (DEFAULT_REGISTRY_HOST, DEFAULT_REGISTRY_PORT)
    if ":" in v:
        host, _, port = v.rpartition(":")
        try:
            return (host, int(port))
        except ValueError:
            return (v, DEFAULT_REGISTRY_PORT)
    return (v, DEFAULT_REGISTRY_PORT)


def advertise_host(registry_host: str) -> str:
    """Best-effort local IP that CONSUMERS can reach this node at.

    A provider binds 0.0.0.0 but must register a routable address. We take the
    source IP the kernel would use on the route toward the registry (a UDP
    "connect" sets the route without sending a packet). ``HALO_COMM_ADVERTISE_HOST``
    overrides (use it when auto-detect picks the wrong interface).
    """
    env = os.environ.get("HALO_COMM_ADVERTISE_HOST")
    if env:
        return env
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((registry_host, 9))          # no datagram is actually sent
            return s.getsockname()[0]
        finally:
            s.close()
    except Exception:  # noqa: BLE001
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:  # noqa: BLE001
            return "127.0.0.1"


def _is_local_host(host: str) -> bool:
    """True if ``host`` names THIS machine (so we're the one that should run the
    registry). Loopback, or the source IP toward it equals it (same box)."""
    if host in ("127.0.0.1", "localhost", "::1", "0.0.0.0"):
        return True
    # If the route "toward" host lands on host itself, host is a local iface.
    return advertise_host(host) == host


def _port_listening(host: str, port: int, timeout_s: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def init(addr: Optional[tuple[str, int]] = None) -> bool:
    """Bootstrap discovery for this machine. If ``addr`` (default from
    ``HALO_COMM_REGISTRY``) points at THIS machine and no registry is listening
    there yet, spawn ``halo_comm registry`` detached. Idempotent — a redundant spawn
    just fails to bind and exits, so it's safe to call from every launcher.
    Returns True if a registry is (or was just started to be) available locally;
    False otherwise (remote master, or spawn failed). Never raises."""
    host, port = addr or registry_addr()
    if not _is_local_host(host):
        return False                                # remote master: not ours to start
    if _port_listening(host, port):
        return True                                 # already up
    import subprocess
    import sys
    # The daemon imports halo_comm from this same source tree, installed or not.
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [pkg_root] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    try:
        # Detached so it outlives the launcher (the registry keeps running).
        subprocess.Popen(
            [sys.executable, "-m", "halo_comm", "registry", "--port", str(port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True, env=env)
    except Exception:  # noqa: BLE001
        return False
    # Wait briefly for the bind so callers can register immediately after.
    for _ in range(40):
        if _port_listening(host, port):
            print(f"[discovery] started registry on {host}:{port}")
            return True
        time.sleep(0.05)
    return _port_listening(host, port)


# --------------------------------------------------------------------------- #
# In-memory table (socket-free; daemon + tests)
# --------------------------------------------------------------------------- #
def _key(ep: "discovery_pb2.Endpoint") -> tuple:
    """Unique key for an endpoint. Providers differ by (name,host,port); consumers
    (port 0) differ by node+pid, so many consumers of the same channel coexist."""
    return (ep.name, ep.host, int(ep.port), ep.node, int(ep.pid))


class RegistryTable:
    """Endpoint table (providers AND consumers), with per-entry TTL expiry.

    Not thread-safe on its own; the daemon serializes access on the service
    poller thread (single writer). ``now`` is injectable for deterministic tests.
    """

    def __init__(self, default_ttl_s: float = DEFAULT_TTL_S):
        self.default_ttl_s = default_ttl_s
        self._eps: dict[tuple, dict] = {}          # key -> {endpoint, expire_at}

    def register(self, ep: "discovery_pb2.Endpoint", ttl_s: float = 0.0,
                 now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        ttl = ttl_s if ttl_s and ttl_s > 0 else self.default_ttl_s
        e = discovery_pb2.Endpoint()
        e.CopyFrom(ep)
        e.stamp = now
        self._eps[_key(ep)] = {"ep": e, "expire": now + ttl}

    def unregister(self, ep: "discovery_pb2.Endpoint") -> None:
        self._eps.pop(_key(ep), None)

    def prune(self, now: Optional[float] = None) -> int:
        """Drop expired entries; return how many were removed."""
        now = time.time() if now is None else now
        dead = [k for k, v in self._eps.items() if v["expire"] <= now]
        for k in dead:
            del self._eps[k]
        return len(dead)

    def list(self, now: Optional[float] = None) -> list["discovery_pb2.Endpoint"]:
        """Live endpoints (prunes expired first)."""
        self.prune(now)
        return [v["ep"] for v in self._eps.values()]

    def __len__(self) -> int:
        return len(self._eps)


# --------------------------------------------------------------------------- #
# Client helpers (thin REQ/REP calls; best-effort)
# --------------------------------------------------------------------------- #
def _new_client(ctx, host: str, port: int, rcv_ms: int = 500, snd_ms: int = 500):
    from .node import Client                       # lazy: avoid import cycle
    return Client(ctx, discovery_pb2.RegistryRequest, discovery_pb2.RegistryReply,
                  REGISTRY_CHANNEL, host, port, rcv_ms=rcv_ms, snd_ms=snd_ms)


def _call(ctx, addr: tuple[str, int], req) -> Optional["discovery_pb2.RegistryReply"]:
    cli = None
    try:
        cli = _new_client(ctx, addr[0], addr[1])
        return cli.call(req)
    except Exception:  # noqa: BLE001 — registry down / unreachable: best-effort
        return None
    finally:
        if cli is not None:
            try:
                cli.close()
            except Exception:  # noqa: BLE001
                pass


def make_endpoint(name: str, pattern: str, host: str, port: int, node: str = "",
                  req_type: str = "", rep_type: str = "", msg_type: str = "",
                  role: str = "provider") -> "discovery_pb2.Endpoint":
    ep = discovery_pb2.Endpoint(
        name=name, pattern=pattern, host=host, port=int(port), node=node,
        req_type=req_type, rep_type=rep_type, msg_type=msg_type, role=role,
        pid=os.getpid())
    return ep


def register(ctx, addr, ep: "discovery_pb2.Endpoint", ttl_s: float = 0.0) -> bool:
    req = discovery_pb2.RegistryRequest(
        op=discovery_pb2.RegistryRequest.REGISTER, endpoint=ep, ttl_s=ttl_s)
    rep = _call(ctx, addr, req)
    return bool(rep and rep.ok)


def unregister(ctx, addr, ep: "discovery_pb2.Endpoint") -> bool:
    req = discovery_pb2.RegistryRequest(
        op=discovery_pb2.RegistryRequest.UNREGISTER, endpoint=ep)
    rep = _call(ctx, addr, req)
    return bool(rep and rep.ok)


def list_endpoints(ctx, addr) -> list["discovery_pb2.Endpoint"]:
    req = discovery_pb2.RegistryRequest(op=discovery_pb2.RegistryRequest.LIST)
    rep = _call(ctx, addr, req)
    return list(rep.endpoints) if rep else []


def resolve(ctx, addr, name: str) -> Optional[tuple[str, int]]:
    """(host, port) of the freshest PROVIDER for ``name``, or None."""
    ep = resolve_endpoint(ctx, addr, name)
    return (ep.host, int(ep.port)) if ep is not None else None


def resolve_endpoint(ctx, addr, name: str):
    """The freshest live PROVIDER Endpoint for ``name`` (consumers ignored — they
    don't offer an address), or None."""
    eps = [e for e in list_endpoints(ctx, addr)
           if e.name == name and e.role != "consumer"]
    if not eps:
        return None
    return max(eps, key=lambda e: e.stamp)


def message_class(type_name: str):
    """Protobuf message CLASS for a (short or full) type name, via the descriptor
    pool — so tools deserialize a channel by the type NAME the registry carries,
    with no hand-maintained name->class table. The generated _pb2 modules must be
    imported (they are, through `halo_comm`); we import them here to be sure. Returns
    None if the type is unknown."""
    if not type_name:
        return None
    from google.protobuf import symbol_database
    from .msgs import state_pb2, motion_pb2, discovery_pb2  # noqa: F401 (register types)
    db = symbol_database.Default()
    for full in (type_name, f"motion_msgs.{type_name}"):
        try:
            return db.GetSymbol(full)
        except KeyError:
            continue
    return None



def resolve_or_wait(ctx, addr, name: str, timeout_s: float = 3.0,
                    poll_s: float = 0.2) -> Optional[tuple[str, int]]:
    """Like resolve(), but poll the registry until ``name`` appears or the
    timeout elapses — for startup ordering (a consumer may come up just before
    its provider registers). Returns None if still absent (caller falls back)."""
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        got = resolve(ctx, addr, name)
        if got is not None:
            return got
        if time.monotonic() >= deadline:
            return None
        time.sleep(poll_s)

