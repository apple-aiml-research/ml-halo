#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Serving service: let other processes run jobs on this process's engine (one GPU owner).

    # GPU-owner process (halo_serve.deploy.attach does this when HALO_SERVE_ENDPOINT is set)
    from halo_serve.runtime.service import start_service
    start_service(global_engine(), "ipc:///tmp/halo-serve-x.sock")

    # any other process
    from halo_serve.runtime.service import call
    out = call("ipc:///tmp/halo-serve-x.sock", "chat", {"messages": [...], "schema": ..., ...})

Transport: ZeroMQ ROUTER over a Unix socket, JSON bodies (no pickle). Replies are sent on job
completion, so several requests can be in flight. Ops are registered by name (`register_op`).
"""
from __future__ import annotations

import json
import os
import queue
import threading
from typing import Callable, Dict

_OPS: Dict[str, Callable] = {}
_started: Dict[str, threading.Thread] = {}


def register_op(name: str, fn: Callable) -> None:
    """An op takes the request dict and returns a Future of a JSON-able dict."""
    _OPS[name] = fn


def default_endpoint(tag: str) -> str:
    """A short ipc path (macOS caps Unix socket paths at 104 bytes; run dirs are longer)."""
    import hashlib
    return f"ipc:///tmp/halo-serve-{hashlib.sha1(str(tag).encode()).hexdigest()[:10]}.sock"


def start_service(engine, endpoint: str) -> threading.Thread:
    if endpoint in _started:
        return _started[endpoint]
    import zmq
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.ROUTER)
    sock.bind(endpoint)
    if endpoint.startswith("ipc://"):
        try:
            os.chmod(endpoint[len("ipc://"):], 0o600)     # this user only
        except OSError:
            pass
    done: "queue.Queue" = queue.Queue()

    def reply(ident, fut):
        try:
            body = {"ok": True, "result": fut.result()}
        except Exception as e:                            # noqa: BLE001
            body = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        done.put((ident, json.dumps(body).encode()))

    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(4, thread_name_prefix="halo-serve-op")

    def dispatch(ident, raw):
        try:
            req = json.loads(raw)
            _OPS[req["op"]](req.get("args") or {}).add_done_callback(
                lambda f, i=ident: reply(i, f))
        except Exception as e:                            # noqa: BLE001
            done.put((ident, json.dumps({"ok": False,
                                         "error": f"{type(e).__name__}: {e}"}).encode()))

    def loop():
        poller = zmq.Poller(); poller.register(sock, zmq.POLLIN)
        while True:
            if dict(poller.poll(2)).get(sock):
                ident, _, raw = sock.recv_multipart()
                # ops run on a worker: a first call may load a model for seconds
                pool.submit(dispatch, ident, raw)
            while not done.empty():
                ident, out = done.get()
                sock.send_multipart([ident, b"", out])

    t = threading.Thread(target=loop, name="halo-serve-service", daemon=True)
    t.start()
    _started[endpoint] = t
    print(f"[halo_serve] service on {endpoint} (ops: {sorted(_OPS)})", flush=True)
    return t


def call(endpoint: str, op: str, args: dict, timeout_s: float = 60.0) -> dict:
    """Blocking client call; a fresh socket per call because ZeroMQ sockets are not thread-safe."""
    import zmq
    s = zmq.Context.instance().socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0)
    s.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
    try:
        s.connect(endpoint)
        s.send(json.dumps({"op": op, "args": args}).encode())
        try:
            body = json.loads(s.recv())
        except zmq.Again:
            raise TimeoutError(f"halo_serve service {endpoint}: no reply to {op!r} in {timeout_s:.0f}s "
                               f"(is the GPU-owner process running?)") from None
    finally:
        s.close()
    if not body.get("ok"):
        raise RuntimeError(f"halo_serve service {op!r} failed: {body.get('error')}")
    return body["result"]
