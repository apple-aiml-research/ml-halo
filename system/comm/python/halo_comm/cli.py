#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""halo_comm -- a ros2-style command-line tool for the halo_comm middleware.

Subcommands (invoke as `halo_comm <cmd>` or `python -m halo_comm <cmd>`):

  echo <topic>     subscribe to a PUB topic and pretty-print each message live
  list             list live registered endpoints from the master registry
  hz <topic>       measure a topic's receive rate (ros2 topic hz)
  call <service>   send a REQ to a REQ/REP service and print the reply
  pub <topic>      publish to a PUB topic (ros2 topic pub)
  graph            live channel dashboard in a browser (rqt_graph-style)
  monitor          live channel status in the terminal (no browser)
  registry         run the master discovery registry daemon
  register         manually register one endpoint (stopgap)

Output defaults to protobuf text_format (indented blocks, like `ros2 topic
echo`); --json / --csv switch formats and --out FILE appends to a log.

There is NO static channel table. A channel is resolved BY NAME from the master
registry (HALO_COMM_REGISTRY, default 127.0.0.1:5560): the registry gives host,
port, pattern, and type names; the message class comes from the protobuf
descriptor pool. See discovery.py.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from contextlib import contextmanager

import zmq
from google.protobuf import text_format
from google.protobuf.json_format import MessageToJson
from .node import Node

PLAN_COLS = 36  # base_xyz(3) + base_quat_xyzw(4) + joints_deploy(29); see zmq_helper


# ── message formatting ────────────────────────────────────────────────────────

def _robotstate_to_csv_row(rs) -> str:
    """One headerless CSV row:
    [unix_time, base_pos(3), imu_quat_wxyz(4), joint_pos(29)] = 37 cols.
    np.loadtxt(delimiter=',')-compatible -- do NOT add a header."""
    if rs.DESCRIPTOR.name != "RobotState":
        raise SystemExit(f"--csv only supports RobotState topics, not {rs.DESCRIPTOR.name}")
    vals = [time.time(), *rs.base_pos, *rs.imu_quat_wxyz, *rs.joint_pos]
    return ",".join(f"{v:.9g}" for v in vals)


def format_msg(msg, fmt: str) -> str:
    """Render a protobuf message as text (default), JSON, or a CSV row."""
    if fmt == "json":
        return MessageToJson(msg, indent=2)
    if fmt == "csv":
        return _robotstate_to_csv_row(msg)
    return text_format.MessageToString(msg, as_utf8=True).rstrip("\n")


def _clip_image(msg):
    """Return (msg_to_show, elided_bytes). RobotState.image.rgb can be MB of
    raw pixels; blank it (on a copy) so text/JSON output stays terminal-safe.
    Returns the original message untouched when there's nothing to elide."""
    if (msg.DESCRIPTOR.name == "RobotState" and msg.HasField("image")
            and len(msg.image.rgb) > 64):
        clone = type(msg)()
        clone.CopyFrom(msg)
        n = len(clone.image.rgb)
        clone.image.rgb = b""
        return clone, n
    return msg, 0


def _emit_msg(emit, msg, fmt: str, full: bool) -> None:
    """Format one message (eliding image bytes unless --full) and emit it."""
    shown, elided = (msg, 0) if full else _clip_image(msg)
    emit(format_msg(shown, fmt))
    if elided and fmt != "csv":
        emit(f"# image.rgb: {elided} bytes elided (use --full)")


# ── request / message builders ────────────────────────────────────────────────

def _load_into(msg, file: str):
    """Populate a protobuf message from a file. .pbtxt/.txt/.prototxt are parsed
    as protobuf text_format; a MotionPlan may instead come from an (n, 36) CSV."""
    text = pathlib.Path(file).read_text()
    if file.endswith((".pbtxt", ".txt", ".prototxt")):
        text_format.Parse(text, msg)
        return
    if msg.DESCRIPTOR.name == "MotionPlan":
        import numpy as np
        arr = np.loadtxt(file, delimiter=",").reshape(-1, PLAN_COLS)
        msg.input_fps = 30
        msg.cols = PLAN_COLS
        msg.values[:] = arr.reshape(-1).tolist()
        return
    text_format.Parse(text, msg)  # fall back to proto-text for other types


def _build_request(req_cls, file):
    """Build a service request from its message class. StateRequest/TrimRequest are
    empty by design; MotionPlan (put_plan) needs --file."""
    req = req_cls()
    if not req_cls.DESCRIPTOR.fields:          # StateRequest / TrimRequest
        return req
    if not file:
        raise SystemExit(f"service needs --file (a {req_cls.DESCRIPTOR.name} payload)")
    _load_into(req, file)
    return req


def _build_topic_msg(msg_cls, file):
    """Build a message to publish from its class. With --file, load it; otherwise
    emit a sane zero-valued default (RobotState with the right field lengths)."""
    msg = msg_cls()
    if file:
        _load_into(msg, file)
        return msg
    if msg_cls.DESCRIPTOR.name == "RobotState":
        msg.joint_pos[:] = [0.0] * 29
        msg.imu_quat_wxyz[:] = [1.0, 0.0, 0.0, 0.0]
        msg.base_pos[:] = [0.0, 0.0, 0.75]
    return msg


# ── resource helpers ──────────────────────────────────────────────────────────

@contextmanager
def _node(name: str):
    """A halo_comm Node with guaranteed shutdown() (closes every socket)."""
    node = Node(name)
    try:
        yield node
    finally:
        node.shutdown()


@contextmanager
def _sink(path):
    """An emit(str) callable that appends to a file (--out) or prints to stdout."""
    if path:
        f = open(path, "a")
        try:
            yield lambda s: (f.write(s + "\n"), f.flush())
        finally:
            f.close()
    else:
        yield lambda s: print(s, flush=True)


# ── subcommands ───────────────────────────────────────────────────────────────

# ── subcommands ───────────────────────────────────────────────────────────────

def _ep_or_die(name: str, registry=None, wait_s: float = 1.0):
    """Resolve the full live Endpoint for ``name`` from the master registry
    (brief wait for startup ordering). Everything a tool needs — pattern, type
    names, host, port — comes from here; there is no static channel table.
    SystemExit with a clear message if it's not registered."""
    from . import discovery
    addr = discovery.registry_addr(registry)
    ctx = zmq.Context.instance()
    deadline = time.monotonic() + max(0.0, wait_s)
    ep = discovery.resolve_endpoint(ctx, addr, name)
    while ep is None and time.monotonic() < deadline:
        time.sleep(0.15)
        ep = discovery.resolve_endpoint(ctx, addr, name)
    if ep is None:
        raise SystemExit(f"'{name}' is not registered on {addr[0]}:{addr[1]} — "
                         f"is it running? (`halo_comm list` shows live endpoints)")
    return ep


def _class_or_die(type_name: str):
    from . import discovery
    cls = discovery.message_class(type_name)
    if cls is None:
        raise SystemExit(f"unknown message type {type_name!r} (not in the protobuf "
                         f"descriptor pool — is the right _pb2 imported?)")
    return cls


def cmd_echo(args) -> int:
    ep = _ep_or_die(args.topic)
    if ep.pattern != "PUB/SUB":
        raise SystemExit(f"{args.topic} is a service, not a topic; use `halo_comm call`")
    msg_cls = _class_or_die(ep.msg_type)
    host, port = (args.host or ep.host), ep.port
    with _node("halo_comm_echo") as node, _sink(args.out) as emit:
        # conflate=False by default so we see EVERY message, not just the freshest
        # (a conflating SUB drops most of a 50 Hz stream). --conflate opts back in.
        sub = node.create_subscription(
            msg_cls, args.topic, host, port, conflate=args.conflate)
        n = 0
        while args.count == 0 or n < args.count:
            msg = sub.recv(timeout_ms=1000)   # 1s poll keeps Ctrl-C responsive
            if msg is None:
                continue
            _emit_msg(emit, msg, args.fmt, args.full)
            if args.fmt == "text":
                emit("---")                   # ros2-style record separator
            n += 1
    return 0


def cmd_list(args) -> int:
    """List the LIVE registered endpoints from the master registry (real hosts,
    multi-machine). Discovery is always-on; --registry overrides which master."""
    from . import discovery
    reg_addr = discovery.registry_addr(getattr(args, "registry", None))
    eps = discovery.list_endpoints(zmq.Context.instance(), reg_addr)
    if args.json:
        print(json.dumps([{"name": e.name, "pattern": e.pattern, "role": e.role or "provider",
                           "host": e.host, "port": e.port, "node": e.node,
                           "types": (f"{e.req_type} -> {e.rep_type}"
                                     if e.req_type else e.msg_type)}
                          for e in eps], indent=2))
        return 0
    rows = [["NAME", "ROLE", "PATTERN", "HOST", "PORT", "NODE", "TYPES"]]
    for e in sorted(eps, key=lambda x: ((x.role == "consumer"), x.name, x.host)):
        rows.append([e.name, (e.role or "provider"), e.pattern, e.host,
                     (str(e.port) if e.role != "consumer" else "-"), e.node,
                     (f"{e.req_type} -> {e.rep_type}" if e.req_type else e.msg_type)])
    if len(rows) == 1:
        print(f"registry {reg_addr[0]}:{reg_addr[1]}: no live endpoints "
              f"(is anything running?)")
        return 0
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    print(f"# live endpoints from registry {reg_addr[0]}:{reg_addr[1]}")
    for r in rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(r, widths)))
    return 0


def cmd_hz(args) -> int:
    ep = _ep_or_die(args.topic)
    if ep.pattern != "PUB/SUB":
        raise SystemExit(f"{args.topic} is a service, not a topic")
    msg_cls = _class_or_die(ep.msg_type)
    host, port = (args.host or ep.host), ep.port
    with _node("halo_comm_hz") as node:
        sub = node.create_subscription(msg_cls, args.topic, host, port, conflate=False)
        stamps: list[float] = []
        t_report = time.monotonic()
        while True:
            if sub.recv(timeout_ms=1000) is not None:
                stamps.append(time.monotonic())
            now = time.monotonic()
            if now - t_report >= args.window:
                if len(stamps) >= 2:
                    dts = [b - a for a, b in zip(stamps, stamps[1:])]
                    rate = (len(stamps) - 1) / (stamps[-1] - stamps[0])
                    mean = sum(dts) / len(dts)
                    print(f"average rate: {rate:7.3f} Hz  "
                          f"min/mean/max period: "
                          f"{min(dts) * 1e3:.2f}/{mean * 1e3:.2f}/{max(dts) * 1e3:.2f} ms  "
                          f"jitter {(max(dts) - min(dts)) * 1e3:.2f} ms  (n={len(stamps)})")
                else:
                    print("no messages received in window")
                stamps, t_report = [], now
    return 0


def cmd_call(args) -> int:
    ep = _ep_or_die(args.service)
    if ep.pattern != "REQ/REP":
        raise SystemExit(f"{args.service} is a topic, not a service; use `halo_comm echo`")
    req_cls, rep_cls = _class_or_die(ep.req_type), _class_or_die(ep.rep_type)
    req = _build_request(req_cls, args.file)
    host, port = (args.host or ep.host), ep.port
    with _node("halo_comm_call") as node, _sink(args.out) as emit:
        cli = node.create_client(req_cls, rep_cls, args.service, host, port,
                                 rcv_ms=args.timeout_ms)
        rep = cli.call(req)   # raises zmq.Again on timeout (caught in main)
        _emit_msg(emit, rep, args.fmt, args.full)
    return 0


def cmd_pub(args) -> int:
    # A publisher BINDS a port, so it can't discover its own address. If the
    # topic is already registered, reuse its type+port; otherwise the user must
    # give --type and --port (there's no static table to look them up in).
    from . import discovery
    ctx = zmq.Context.instance()
    ep = discovery.resolve_endpoint(ctx, discovery.registry_addr(args.registry), args.topic)
    if ep is not None and ep.pattern != "PUB/SUB":
        raise SystemExit(f"{args.topic} is a service, not a topic; use `halo_comm call`")
    type_name = ep.msg_type if ep is not None else args.type
    port = ep.port if ep is not None else args.port
    if not type_name or not port:
        raise SystemExit(f"{args.topic} is not registered — pass --type <MsgName> "
                         f"and --port <p> to publish it")
    msg_cls = _class_or_die(type_name)
    msg = _build_topic_msg(msg_cls, args.file)
    with _node("halo_comm_pub") as node:
        pub = node.create_publisher(
            msg_cls, args.topic, int(port), bind_ip=args.bind_ip)
        period = 1.0 / args.rate if args.rate > 0 else 0.0
        time.sleep(0.2)   # PUB slow-joiner: let subscribers connect before first send
        n = 0
        while args.count == 0 or n < args.count:
            pub.publish(msg)
            n += 1
            if period:
                time.sleep(period)
    return 0


def cmd_graph(args) -> int:
    from . import graph, discovery
    # Registry-driven always (discovery is always-on); --registry just overrides
    # which master (else HALO_COMM_REGISTRY / localhost default).
    registry = discovery.registry_addr(args.registry)
    reg_str = f"{registry[0]}:{registry[1]}"
    if args.open:
        # open the browser shortly after the server starts listening
        import threading
        import webbrowser
        url = f"http://{args.bind}:{args.port}"
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    graph.serve(port=args.port, bind=args.bind, do_probe=not args.no_probe, registry=reg_str)
    return 0


def cmd_monitor(args) -> int:
    """Terminal live view — all channels' up/down + topic Hz, refreshed in place.
    The graph's value without a browser or an :8080 bind (works over SSH)."""
    from . import graph, discovery
    registry = discovery.registry_addr(args.registry)
    reg_str = f"{registry[0]}:{registry[1]}"
    eps = graph._registry_topic_endpoints(reg_str)
    monitor = graph.LiveMonitor(endpoints=eps)
    hide, show = "\x1b[?25l", "\x1b[?25h"   # hide/show cursor
    sys.stdout.write(hide)
    try:
        while True:
            snap = graph.snapshot(monitor, do_probe=True, registry=reg_str)
            lines = [f"halo_comm monitor — registry {reg_str}   "
                     f"({time.strftime('%H:%M:%S')}, every {args.interval:g}s, Ctrl-C to stop)",
                     ""]
            lines.append(f"  {'CHANNEL':<14} {'HOST':<17} {'PORT':<6} {'NODE':<12} "
                         f"{'LIVE':<6} {'RATE':<10}")
            for c in snap["channels"]:
                live = "up" if c["live"] else "down"
                st = c.get("stats") or {}
                rate = f"{st['hz']:.1f} Hz" if st.get("hz") else "—"
                lines.append(f"  {c['name']:<14} {c.get('host',''):<17} "
                             f"{c['port']:<6} {c.get('node',''):<12} {live:<6} {rate:<10}")
            if not snap["channels"]:
                lines.append("  (registry has no live endpoints — is anything running?)")
            # redraw from the top each tick (clear-screen + home)
            sys.stdout.write("\x1b[2J\x1b[H" + "\n".join(lines) + "\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    finally:
        sys.stdout.write(show)
        sys.stdout.flush()
        monitor.close()
    return 0



# ── argument parsing ──────────────────────────────────────────────────────────
def cmd_registry(args) -> int:
    """Run the master registry daemon (the 'master address' every node targets)."""
    from . import registry, discovery
    # honor HALO_COMM_REGISTRY's port if set, else the default
    port = args.port if args.port is not None else discovery.registry_addr()[1]
    ttl = args.ttl if args.ttl is not None else discovery.DEFAULT_TTL_S
    registry.serve(bind=args.bind, port=port, ttl_s=ttl)
    return 0


def cmd_register(args) -> int:
    """Manually register one endpoint — a stopgap for a provider that can't
    self-register. Pattern + type names are given explicitly (there's no static
    channel table to look them up in)."""
    from . import discovery
    addr = discovery.registry_addr(args.registry)
    if args.pattern == "REQ/REP":
        kw = {"req_type": args.req_type or "", "rep_type": args.rep_type or ""}
    else:
        kw = {"msg_type": args.msg_type or ""}
    ep = discovery.make_endpoint(args.channel, args.pattern, args.host, args.port,
                                 node=args.node, **kw)
    ok = discovery.register(zmq.Context.instance(), addr, ep, ttl_s=args.ttl)
    print(f"register {args.channel} @ {args.host}:{args.port} -> "
          f"{'ok' if ok else 'FAILED (registry unreachable?)'}")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="halo_comm", description="ros2-style CLI for the halo_comm middleware.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # shared output flags (echo, call)
    fmt = argparse.ArgumentParser(add_help=False)
    g = fmt.add_mutually_exclusive_group()
    g.add_argument("--json", action="store_const", dest="fmt", const="json",
                   help="output MessageToJson instead of text")
    g.add_argument("--csv", action="store_const", dest="fmt", const="csv",
                   help="output a headerless 37-col CSV row [time, base_pos, imu_quat_wxyz, joints] (RobotState only)")
    fmt.set_defaults(fmt="text")
    fmt.add_argument("--out", metavar="FILE", help="append formatted messages to FILE")
    fmt.add_argument("--full", action="store_true", help="do not elide image.rgb bytes")

    pe = sub.add_parser("echo", parents=[fmt], help="print a topic's messages live")
    pe.add_argument("topic")
    pe.add_argument("-n", "--count", type=int, default=0,
                    help="stop after N messages (0 = forever)")
    pe.add_argument("--host", default=None,
                    help="override the host (default: resolve from the registry)")
    pe.add_argument("--conflate", action="store_true",
                    help="show only the freshest message (default: every message)")
    pe.set_defaults(func=cmd_echo)

    pl = sub.add_parser("list", help="list live registered endpoints (real hosts)")
    pl.add_argument("--json", action="store_true", help="machine-readable output")
    pl.add_argument("--registry", default=None,
                    help="master registry host:port (default HALO_COMM_REGISTRY / localhost)")
    pl.set_defaults(func=cmd_list)

    ph = sub.add_parser("hz", help="measure a topic's receive rate")
    ph.add_argument("topic")
    ph.add_argument("--host", default=None,
                    help="override the host (default: resolve from the registry)")
    ph.add_argument("-w", "--window", type=float, default=5.0,
                    help="report every W seconds (default 5)")
    ph.set_defaults(func=cmd_hz)

    pc = sub.add_parser("call", parents=[fmt], help="call a REQ/REP service")
    pc.add_argument("service")
    pc.add_argument("--host", default=None,
                    help="override the host (default: resolve from the registry)")
    pc.add_argument("--file", help="request payload (CSV / proto-text; required for put_plan)")
    pc.add_argument("--timeout-ms", type=int, default=5000)
    pc.set_defaults(func=cmd_call)

    pp = sub.add_parser("pub", help="publish to a topic")
    pp.add_argument("topic")
    pp.add_argument("--file", help="message source (CSV row / proto-text)")
    pp.add_argument("--type", default=None,
                    help="message type name (needed only if the topic isn't registered)")
    pp.add_argument("--port", type=int, default=None,
                    help="bind port (needed only if the topic isn't registered)")
    pp.add_argument("--registry", default=None, help="master registry host:port")
    pp.add_argument("-r", "--rate", type=float, default=1.0,
                    help="publish rate Hz (default 1)")
    pp.add_argument("-n", "--count", type=int, default=0,
                    help="stop after N publishes (0 = forever)")
    pp.add_argument("--bind-ip", default="0.0.0.0")
    pp.set_defaults(func=cmd_pub)

    pg = sub.add_parser("graph", help="live channel dashboard in a browser (rqt_graph-style)")
    pg.add_argument("--port", type=int, default=8080, help="server port (default 8080)")
    pg.add_argument("--bind", default="127.0.0.1", help="server bind address")
    pg.add_argument("--no-open", dest="open", action="store_false", default=True,
                    help="do not open a browser")
    pg.add_argument("--no-probe", action="store_true",
                    help="skip the per-poll reachability probe")
    pg.add_argument("--registry", default=None,
                    help="master registry host:port (default HALO_COMM_REGISTRY / localhost)")
    pg.set_defaults(func=cmd_graph)

    pm = sub.add_parser("monitor", help="live channel status in the terminal (no browser)")
    pm.add_argument("--registry", default=None,
                    help="master registry host:port (default HALO_COMM_REGISTRY / localhost)")
    pm.add_argument("-i", "--interval", type=float, default=1.0,
                    help="refresh interval seconds (default 1)")
    pm.set_defaults(func=cmd_monitor)

    preg = sub.add_parser("registry", help="run the master discovery registry daemon")
    preg.add_argument("--port", type=int, default=None,
                      help="registry port (default 5560 / HALO_COMM_REGISTRY)")
    preg.add_argument("--bind", default="0.0.0.0", help="bind address (default 0.0.0.0)")
    preg.add_argument("--ttl", type=float, default=None,
                      help="endpoint lifetime seconds (default 10)")
    preg.set_defaults(func=cmd_registry)

    prg = sub.add_parser("register",
                         help="manually register one endpoint (stopgap, e.g. C++ tracker)")
    prg.add_argument("channel", help="channel name, e.g. g1/state")
    prg.add_argument("--host", required=True, help="advertised host consumers connect to")
    prg.add_argument("--port", type=int, required=True)
    prg.add_argument("--registry", default=None,
                     help="registry host:port (default HALO_COMM_REGISTRY)")
    prg.add_argument("--pattern", default="REQ/REP", choices=["REQ/REP", "PUB/SUB"],
                     help="channel pattern")
    prg.add_argument("--req-type", default=None, help="request type name (REQ/REP)")
    prg.add_argument("--rep-type", default=None, help="reply type name (REQ/REP)")
    prg.add_argument("--msg-type", default=None, help="message type name (PUB/SUB)")
    prg.add_argument("--node", default="manual", help="owning node label")
    prg.add_argument("--ttl", type=float, default=0.0,
                     help="lifetime seconds (0 = registry default; re-run to refresh)")
    prg.set_defaults(func=cmd_register)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args) or 0
    except KeyboardInterrupt:
        return 130
    except zmq.Again:
        print("timeout: no reply within deadline (is the tracker running?)",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
