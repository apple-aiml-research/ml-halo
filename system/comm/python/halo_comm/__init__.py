#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""halo_comm — ROS2-style middleware over ZMQ (TCP) with protobuf messages."""
from .node import Node, Publisher, Subscriber, Client  # noqa: F401

# Generated protobuf message modules (built by `make protos` / pip install).
from .msgs import state_pb2, motion_pb2, discovery_pb2  # noqa: F401

# Discovery — the master registry (see discovery.py). There is no static channel
# table: providers register by name, consumers resolve by name. Call halo_comm.init()
# once at startup to bootstrap the registry when this machine is the master (no-op
# when the master is remote). discovery.* has the client helpers (registry_addr,
# resolve, resolve_endpoint, message_class, ...).
from . import discovery  # noqa: F401
from .discovery import init  # noqa: F401

__all__ = ["Node", "Publisher", "Subscriber", "Client", "state_pb2", "motion_pb2",
           "discovery_pb2", "discovery", "init"]
