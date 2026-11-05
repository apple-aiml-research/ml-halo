#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""The halo_comm master registry daemon (`halo_comm registry`).

A single well-known REQ/REP service that providers register with and consumers
query — the "master address" model (ROS2 Discovery Server). See discovery.py for
the design rationale and the client-side helpers; this file is just the server.

    provider (Node)  --REGISTER/heartbeat-->  registry  <--LIST--  consumer
                                                 |
                                          RegistryTable (TTL)

Run it on one always-on box (e.g. the Mac mini) and point every node/tool at it:

    halo_comm registry --bind 0.0.0.0 --port 5560
    export HALO_COMM_REGISTRY=<that-box-ip>:5560     # on every machine

Discovery-only: the ZMQ data path is brokerless, so if this daemon dies, live
connections keep running — only NEW discovery pauses until it is back.
"""
from __future__ import annotations

import time

from . import discovery
from .msgs import discovery_pb2
from .node import Node


def handle(table: discovery.RegistryTable,
           req: "discovery_pb2.RegistryRequest") -> "discovery_pb2.RegistryReply":
    """Pure request handler (socket-free; unit-testable). Mutates ``table``."""
    rep = discovery_pb2.RegistryReply(ok=True, now=time.time())
    op = req.op
    if op == discovery_pb2.RegistryRequest.REGISTER:
        table.register(req.endpoint, req.ttl_s)
    elif op == discovery_pb2.RegistryRequest.UNREGISTER:
        table.unregister(req.endpoint)
    elif op == discovery_pb2.RegistryRequest.LIST:
        rep.endpoints.extend(table.list())
    else:  # unknown op
        rep.ok = False
    return rep


def serve(bind: str = "0.0.0.0", port: int = discovery.DEFAULT_REGISTRY_PORT,
          ttl_s: float = discovery.DEFAULT_TTL_S) -> None:
    """Run the registry until Ctrl-C."""
    table = discovery.RegistryTable(default_ttl_s=ttl_s)
    # enable_discovery=False: the registry must not try to register with itself.
    node = Node("halo_comm_registry", enable_discovery=False)
    node.create_service(
        discovery_pb2.RegistryRequest, discovery_pb2.RegistryReply,
        discovery.REGISTRY_CHANNEL, port, lambda r: handle(table, r), bind_ip=bind)
    print(f"halo_comm registry: serving {discovery.REGISTRY_CHANNEL} on "
          f"{bind}:{port}  (ttl {ttl_s:.0f}s)")
    print("  set HALO_COMM_REGISTRY=<this-host>:%d on every node/tool. Ctrl-C to stop." % port)
    last_prune = time.time()
    try:
        while True:
            node.spin_once(200)
            now = time.time()
            if now - last_prune >= 2.0:               # periodic TTL sweep + status
                n = table.prune(now)
                last_prune = now
                if n:
                    print(f"  [registry] expired {n} stale endpoint(s); "
                          f"{len(table)} live")
    except KeyboardInterrupt:
        print("\nhalo_comm registry: stopping.")
    finally:
        node.shutdown()
