#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""halo_comm — a small ROS2-style middleware over ZMQ (TCP) with protobuf messages.

Primitives (symmetric with the C++ side in include/halo_comm/):
    node = Node("tracker")                                  # server role
    node.create_service(StateRequest, RobotState, "g1/get_state", port=5555, handler=fn)
    pub = node.create_publisher(RobotState, "g1/state", port=5558)
    node.spin_once()                                        # poll REP sockets + dispatch

    node = Node("generator")                                # client role
    cli = node.create_client(StateRequest, RobotState, "g1/get_state", host="10.0.0.5", port=5555)
    rep = cli.call(StateRequest())                          # blocking REQ/REP
    sub = node.create_subscription(RobotState, "g1/state", host="10.0.0.5", port=5558,
                                   conflate=True, on_message=cb)   # callback (bg thread)
    latest = sub.recv(timeout_ms=20)                        # OR poll

Transport is TCP (cross-device). Server binds tcp://{bind_ip}:{port}; client
connects tcp://{host}:{port}. Messages are protobuf: the middleware calls
SerializeToString()/ParseFromString(); no hand byte-packing.

NOTE: a Publisher publishes; a Client calls — they are different ZMQ socket
types (PUB vs REQ) and are not interchangeable.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional

import zmq


def _addr(host: str, port: int) -> str:
    return f"tcp://{host}:{port}"


class Publisher:
    """PUB socket. One-way fire-and-forget (e.g. the 50 Hz g1/state topic)."""

    def __init__(self, ctx: "zmq.Context", name: str, port: int,
                 bind_ip: str = "0.0.0.0", sndhwm: int = 1):
        self.name = name
        self._sock = ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, sndhwm)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(_addr(bind_ip, port))

    def publish(self, msg) -> None:
        self._sock.send(msg.SerializeToString(), flags=zmq.DONTWAIT)

    def close(self) -> None:
        self._sock.close(0)


class Subscriber:
    """SUB socket. recv() polls for the freshest message (CONFLATE); pass
    on_message= to instead receive via a background callback thread."""

    def __init__(self, ctx: "zmq.Context", msg_type, name: str, host: str, port: int,
                 conflate: bool = True, on_message: Optional[Callable] = None):
        self.name = name
        self._msg_type = msg_type
        self._sock = ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        if conflate:
            self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.connect(_addr(host, port))
        self._on_message = on_message
        self._running = False
        self._thread: Optional[threading.Thread] = None
        if on_message is not None:
            self._running = True
            self._thread = threading.Thread(target=self._loop, name=f"sub:{name}", daemon=True)
            self._thread.start()

    def recv(self, timeout_ms: int = 0):
        """Freshest message or None on timeout. Used by the synchronous loop."""
        if self._sock.poll(timeout_ms) == 0:
            return None
        m = self._msg_type()
        m.ParseFromString(self._sock.recv())
        return m

    def _loop(self) -> None:
        # Trivial, exception-safe: parse + hand to the (trivial) callback.
        while self._running:
            try:
                if self._sock.poll(50) == 0:
                    continue
                m = self._msg_type()
                m.ParseFromString(self._sock.recv())
                self._on_message(m)
            except Exception:  # never let a bad frame kill the recv thread
                continue

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._sock.close(0)


class Client:
    """REQ socket. Blocking request/reply with timeouts + socket reset on
    timeout (mirrors the legacy TrackerClient resilience)."""

    def __init__(self, ctx: "zmq.Context", req_type, rep_type, name: str,
                 host: str, port: int, rcv_ms: int = 5000, snd_ms: int = 2000):
        self.name = name
        self._ctx = ctx
        self._req_type = req_type
        self._rep_type = rep_type
        self._host = host
        self._port = port
        self._rcv_ms = rcv_ms
        self._snd_ms = snd_ms
        self._sock = None
        self._connect()

    def _connect(self) -> None:
        if self._sock is not None:
            self._sock.close(0)
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, self._rcv_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, self._snd_ms)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(_addr(self._host, self._port))

    def call(self, req):
        """Send req, return parsed reply. Raises zmq.Again on timeout (after
        resetting the socket so the next call can retry cleanly)."""
        try:
            self._sock.send(req.SerializeToString())
            data = self._sock.recv()
        except zmq.Again:
            self._connect()  # REQ socket is stuck after a timeout; rebuild it
            raise
        rep = self._rep_type()
        rep.ParseFromString(data)
        return rep

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(0)


class _Service:
    """REP socket + typed handler. Polled by Node.spin_once()."""

    def __init__(self, ctx: "zmq.Context", req_type, rep_type, name: str,
                 port: int, handler: Callable, bind_ip: str = "0.0.0.0"):
        self.name = name
        self._req_type = req_type
        self._rep_type = rep_type
        self._handler = handler
        self.sock = ctx.socket(zmq.REP)
        self.sock.bind(_addr(bind_ip, port))

    def handle_ready(self) -> None:
        """Called by the poller when this REP socket has a request waiting."""
        req = self._req_type()
        req.ParseFromString(self.sock.recv())
        rep = self._handler(req)               # handler(req) -> rep proto
        self.sock.send(rep.SerializeToString())


class Node:
    """Owns a ZMQ context + the registered endpoints. Server side calls
    create_service/create_publisher then spin_once() in its loop; client side calls
    create_client/create_subscription and uses the returned handles."""

    def __init__(self, name: str, enable_discovery: bool = True):
        self.name = name
        self._ctx = zmq.Context.instance()
        self._services: list[_Service] = []
        self._publishers: list[Publisher] = []
        self._subscriptions: list[Subscriber] = []
        self._clients: list[Client] = []
        # ── discovery (see discovery.py) ─────────────────────────────────────
        # Registry address resolved once (HALO_COMM_REGISTRY, else localhost). Providers that
        # bind (create_service/create_publisher) auto-register + heartbeat; consumers
        # (create_client/create_subscription) may resolve host from it. enable_discovery=False
        # is used by the registry daemon itself (it IS the registry).
        from . import discovery                      # local import: no cycle
        self._disc = discovery
        self._registry = discovery.registry_addr() if enable_discovery else None
        self._advertise = (discovery.advertise_host(self._registry[0])
                           if self._registry else None)
        self._registered: list = []                  # Endpoints we announced
        self._hb_thread = None
        self._hb_stop = threading.Event()

    # ── server side ──────────────────────────────────────────────────────────
    def create_service(self, req_type, rep_type, name: str, port: int,
                        handler: Callable, bind_ip: str = "0.0.0.0") -> None:
        self._services.append(_Service(self._ctx, req_type, rep_type, name, port,
                                       handler, bind_ip))
        self._announce(name, "REQ/REP", port,
                       req_type=req_type.DESCRIPTOR.name,
                       rep_type=rep_type.DESCRIPTOR.name)

    def create_publisher(self, msg_type, name: str, port: int, bind_ip: str = "0.0.0.0",
                         sndhwm: int = 1) -> Publisher:
        pub = Publisher(self._ctx, name, port, bind_ip, sndhwm)
        self._publishers.append(pub)
        self._announce(name, "PUB/SUB", port, msg_type=msg_type.DESCRIPTOR.name)
        return pub

    # ── client side ──────────────────────────────────────────────────────────
    def create_client(self, req_type, rep_type, name: str, host: Optional[str], port: int,
                      rcv_ms: int = 5000, snd_ms: int = 2000) -> Client:
        host, port = self._resolve(name, host, port)
        cli = Client(self._ctx, req_type, rep_type, name, host, port, rcv_ms, snd_ms)
        self._clients.append(cli)
        self._announce(name, "REQ/REP", 0, role="consumer",
                       req_type=req_type.DESCRIPTOR.name, rep_type=rep_type.DESCRIPTOR.name)
        return cli

    def create_subscription(self, msg_type, name: str, host: Optional[str], port: int,
                           conflate: bool = True, on_message: Optional[Callable] = None
                           ) -> Subscriber:
        host, port = self._resolve(name, host, port)
        sub = Subscriber(self._ctx, msg_type, name, host, port, conflate, on_message)
        self._subscriptions.append(sub)
        self._announce(name, "PUB/SUB", 0, role="consumer",
                       msg_type=msg_type.DESCRIPTOR.name)
        return sub

    # ── discovery helpers (all best-effort; no-op when registry disabled) ─────
    def _announce(self, name: str, pattern: str, port: int, req_type: str = "",
                  rep_type: str = "", msg_type: str = "", role: str = "provider") -> None:
        """Register an endpoint with the master registry + start the heartbeat
        thread on first announce. Providers register their bound host:port;
        consumers register with port 0 (they don't bind — the entry says 'this
        node uses this channel'). Silent no-op if discovery is off."""
        if not self._registry:
            return
        ep = self._disc.make_endpoint(
            name, pattern, self._advertise, port, node=self.name, role=role,
            req_type=req_type, rep_type=rep_type, msg_type=msg_type)
        self._disc.register(self._ctx, self._registry, ep)
        self._registered.append(ep)                  # keep full metadata for heartbeat
        if self._hb_thread is None:
            self._hb_thread = threading.Thread(
                target=self._heartbeat_loop, name=f"disc-hb:{self.name}", daemon=True)
            self._hb_thread.start()

    def _heartbeat_loop(self) -> None:
        # Re-register every ttl/3 so the registry never expires a live endpoint.
        # Re-send the FULL endpoint (not a stub) so metadata survives the refresh.
        period = max(1.0, self._disc.DEFAULT_TTL_S / 3.0)
        while not self._hb_stop.wait(period):
            for ep in list(self._registered):
                self._disc.register(self._ctx, self._registry, ep)

    def _resolve(self, name: str, host: Optional[str], port: int) -> tuple[str, int]:
        """Return (host, port) to connect to. Explicit host wins;
        host=None/"auto" triggers a registry lookup; falls back to the given port
        on 127.0.0.1 if the registry can't resolve it."""
        if host not in (None, "", "auto"):
            return host, port
        if self._registry:
            got = self._disc.resolve(self._ctx, self._registry, name)
            if got:
                return got
        return (host or "127.0.0.1"), port

    # ── server poller ────────────────────────────────────────────────────────
    def spin_once(self, timeout_ms: int = 100) -> None:
        """Poll all service REP sockets once and dispatch any ready requests."""
        if not self._services:
            return
        poller = zmq.Poller()
        for svc in self._services:
            poller.register(svc.sock, zmq.POLLIN)
        ready = dict(poller.poll(timeout_ms))
        for svc in self._services:
            if ready.get(svc.sock) == zmq.POLLIN:
                svc.handle_ready()

    def shutdown(self) -> None:
        # Stop heartbeating + withdraw our endpoints from the registry (so the
        # graph reflects the shutdown immediately instead of waiting for TTL).
        self._hb_stop.set()
        if self._registry:
            for ep in self._registered:
                self._disc.unregister(self._ctx, self._registry, ep)
            self._registered.clear()
        for grp in (self._services, self._publishers, self._subscriptions, self._clients):
            for ep in grp:
                close = getattr(ep, "close", None)
                if close:
                    close()
                elif hasattr(ep, "sock"):
                    ep.sock.close(0)
