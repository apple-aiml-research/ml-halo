#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Live channel graph for halo_comm (`halo_comm graph`) — server + renderer in one file.

A stdlib-only dashboard on localhost showing the tracker<->generator link in real
time. Each poll re-probes channel liveness and, for topics, reports the measured
receive rate + latest sample. Two endpoints:

    GET /            -> the dashboard HTML (polls /api/graph every ~1.5 s)
    GET /api/graph   -> a JSON snapshot of the registry + live stats

Live-only by design (no static-file mode): the page is worthless frozen. The
routing is a pure function (`route`) so the whole request surface is unit-testable
without ever binding a socket; `serve()` wraps it in a ThreadingHTTPServer.

`LiveMonitor` (the background topic subscriber that tracks Hz + latest sample) is
also reused by the terminal view, `halo_comm monitor`.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import zmq

from . import discovery
from .node import Node

POLL_MS = 1500   # browser poll interval


class LiveMonitor:
    """Background subscriber to every PUB/SUB topic; tracks receive rate and the
    latest sample. Uses halo_comm's own callback-thread subscriptions (conflate off
    so the rate is honest). Safe to construct even if nothing is publishing —
    stats just stay empty."""

    def __init__(self, window_s: float = 3.0, endpoints=None):
        self.window_s = window_s
        self._node = Node("halo_comm_graph_monitor")
        self._stamps: dict[str, deque] = {}
        self._latest: dict[str, str] = {}
        self._latest_tick: dict[str, Optional[int]] = {}
        self._lock = threading.Lock()
        # endpoints: (name, host, port, msg_cls) — each topic subscribed at its
        # REAL host (from the registry).
        for (name, thost, tport, mcls) in endpoints or []:
            if mcls is None:
                continue                       # unknown type: can't parse for Hz
            self._stamps[name] = deque(maxlen=1024)
            self._node.create_subscription(
                mcls, name, thost, tport,
                conflate=False, on_message=self._make_cb(name))

    def _make_cb(self, name: str):
        def cb(msg):
            from . import cli  # lazy: reuse the canonical formatter/eliding
            now = time.monotonic()
            shown, _ = cli._clip_image(msg)
            text = cli.format_msg(shown, "text")
            tick = int(msg.tick) if msg.DESCRIPTOR.fields_by_name.get("tick") else None
            with self._lock:
                self._stamps[name].append(now)
                self._latest[name] = text
                self._latest_tick[name] = tick
        return cb

    def stats(self, name: str) -> dict:
        now = time.monotonic()
        with self._lock:
            stamps = [t for t in self._stamps.get(name, ()) if now - t <= self.window_s]
            latest = self._latest.get(name)
            tick = self._latest_tick.get(name)
            last = self._stamps[name][-1] if self._stamps.get(name) else None
        hz = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) >= 2 else 0.0
        return {
            "hz": round(hz, 2),
            "last_tick": tick,
            "last_seen_age": round(now - last, 2) if last is not None else None,
            "latest": latest,
        }

    def close(self) -> None:
        self._node.shutdown()


def _probe_hostport(host: str, port: int, timeout_s: float = 0.3) -> bool:
    """Reachability probe: can THIS machine open a TCP connection to host:port?
    In multi-host, this is the key diagnostic — the registry may know an
    endpoint's address while the local machine can't route to it."""
    import socket
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_s):
            return True
    except OSError:
        return False


def _rows_from_endpoints(eps, monitor: Optional[LiveMonitor], do_probe: bool) -> list:
    """Build graph rows from live registry endpoints (pure aside from probe/
    monitor reads). Each row carries the REAL host/port/node."""
    rows = []
    for e in eps:
        is_topic = (e.pattern == "PUB/SUB")
        types = (f"{e.req_type} -> {e.rep_type}" if e.req_type else e.msg_type)
        is_consumer = (e.role == "consumer")
        if is_consumer:
            direction = "subscribes" if is_topic else "calls"
        else:
            direction = "publishes" if is_topic else "serves"
        row = {
            "name": e.name, "pattern": e.pattern, "host": e.host, "port": int(e.port),
            "node": e.node, "types": types, "is_topic": is_topic,
            "role": e.role or "provider", "direction": direction,
            # consumers don't bind a port, so there's nothing to probe.
            "live": (_probe_hostport(e.host, e.port)
                     if (do_probe and not is_consumer and e.port) else None),
        }
        if is_topic and not is_consumer and monitor is not None:
            row["stats"] = monitor.stats(e.name)
        rows.append(row)
    return rows


def _endpoints(registry: Optional[str]):
    """(registry address, live endpoints) for ``registry`` (or HALO_COMM_REGISTRY / localhost)."""
    reg_addr = discovery.registry_addr(registry)
    return reg_addr, discovery.list_endpoints(zmq.Context.instance(), reg_addr)


def snapshot(monitor: Optional[LiveMonitor] = None,
             do_probe: bool = True, registry: Optional[str] = None) -> dict:
    """Build the JSON payload served at /api/graph. Registry-driven: rows are the
    LIVE registered endpoints (real hosts/nodes/ports), so the graph is correct
    across machines. ``registry`` (or HALO_COMM_REGISTRY) picks which master; it
    defaults to localhost."""
    reg_addr, eps = _endpoints(registry)
    return {"registry": f"{reg_addr[0]}:{reg_addr[1]}",
            "channels": _rows_from_endpoints(eps, monitor, do_probe)}


def render_html(initial: dict) -> str:
    """Return the live dashboard HTML, seeded with an initial snapshot; the page
    then polls /api/graph and re-renders. Offline (inline CSS/JS, no CDN)."""
    return (_TEMPLATE
            .replace("__DATA__", json.dumps(initial))
            .replace("__POLL_MS__", str(POLL_MS)))


def route(path: str, monitor: Optional[LiveMonitor] = None,
          do_probe: bool = True, registry: Optional[str] = None) -> tuple[int, str, bytes]:
    """Map an HTTP path to (status, content_type, body). Pure & socket-free."""
    path = path.split("?", 1)[0]
    if path in ("/", "/index.html"):
        html = render_html(snapshot(monitor, do_probe, registry))
        return 200, "text/html; charset=utf-8", html.encode()
    if path == "/api/graph":
        body = json.dumps(snapshot(monitor, do_probe, registry)).encode()
        return 200, "application/json", body
    return 404, "text/plain; charset=utf-8", b"not found"


def _registry_topic_endpoints(registry: str):
    """(name, host, port, msg_cls) for each PUB topic the registry knows — so the
    LiveMonitor can subscribe at each topic's REAL host. msg_cls comes from the
    registered type name; unknown types are skipped for Hz."""
    _, eps = _endpoints(registry)
    out = []
    for e in eps:
        if e.pattern != "PUB/SUB":
            continue
        # Skip heavy topics (images) so the monitor never pulls frames just to
        # count them — heuristic on the registered type name (no static table).
        if "image" in (e.msg_type or "").lower():
            continue
        out.append((e.name, e.host, int(e.port), discovery.message_class(e.msg_type)))
    return out


def serve(port: int = 8080, bind: str = "127.0.0.1", do_probe: bool = True,
          registry: Optional[str] = None) -> None:
    """Run the dashboard until Ctrl-C. ``bind``/``port`` is where the dashboard itself
    listens; the graph shows the LIVE registered endpoints at their real hosts."""
    # Subscribe each topic at its registered host for Hz.
    monitor = LiveMonitor(endpoints=_registry_topic_endpoints(registry))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (http.server API)
            status, ctype, body = route(self.path, monitor, do_probe, registry)
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # quiet by default
            pass

    httpd = ThreadingHTTPServer((bind, port), Handler)
    print(f"halo_comm graph: serving http://{bind}:{port}  (observing registry {registry})")
    print("  Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
        monitor.close()


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>halo_comm channel graph</title>
<style>
  :root { --bg:#0f1419; --panel:#1a2029; --ink:#e6edf3; --muted:#8b98a5;
    --gen:#1f6feb; --trk:#238636; --up:#3fb950; --down:#6e7681; --edge:#58a6ff;
    --line:#30363d; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
    font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  header { padding:18px 24px; border-bottom:1px solid var(--line);
    display:flex; justify-content:space-between; align-items:flex-start; }
  header h1 { margin:0; font-size:18px; }
  header .sub { color:var(--muted); font-size:13px; margin-top:4px; }
  .status { font-size:12px; color:var(--muted); text-align:right; white-space:nowrap; }
  .status .dot { display:inline-block; width:8px; height:8px; border-radius:50%;
    margin-right:5px; background:var(--down); }
  .status.on .dot { background:var(--up); }
  .wrap { display:flex; height:calc(100vh - 92px); }
  .canvas { flex:1; overflow:hidden; }
  svg#graph { display:block; width:100%; height:100%; }
  .role .label { font-weight:700; font-size:15px; fill:#fff; }
  .role .role-sub { font-size:11px; fill:rgba(255,255,255,.75); }
  .edge path { fill:none; stroke:var(--edge); stroke-width:2.5;
    transition:stroke .2s, stroke-width .1s, opacity .1s; }
  .edge.up path { stroke:var(--up); }
  .edge.down path { stroke:var(--down); stroke-dasharray:6 5; }
  svg text { fill:var(--ink); }                 /* default: never black-on-dark */
  .name { font-weight:600; font-size:13px; fill:var(--ink); }
  .types { font-size:11px; fill:var(--muted); }
  .edge .name { font-weight:600; font-size:13px; fill:var(--ink); }
  .edge .types { font-size:11px; fill:var(--muted); }
  .edge .hz { font-size:11px; fill:var(--up); font-weight:600; }
  .edge .port { font-size:11px; fill:var(--ink); }
  .edge .portbg { fill:var(--panel); stroke:var(--line); rx:6; }
  .edge:hover path { stroke-width:4; }
  .edge.dim { opacity:.2; }
  aside { width:340px; border-left:1px solid var(--line); padding:18px;
    overflow:auto; background:var(--panel); }
  aside h2 { font-size:13px; text-transform:uppercase; letter-spacing:.05em;
    color:var(--muted); margin:0 0 10px; }
  .card { border:1px solid var(--line); border-radius:8px; padding:12px; margin-bottom:10px; }
  .card:hover { border-color:var(--edge); }
  .card .top { display:flex; justify-content:space-between; align-items:center; }
  .card .cname { font-weight:600; }
  .card .badge { font-size:11px; padding:2px 7px; border-radius:10px;
    background:#21262d; color:var(--muted); }
  .card .badge.up { background:rgba(63,185,80,.15); color:var(--up); }
  .card .badge.down { background:rgba(110,118,129,.2); color:var(--down); }
  .card .meta { color:var(--muted); font-size:12px; margin-top:6px; }
  .card pre { margin:8px 0 0; padding:8px; background:#0d1117; border-radius:6px;
    font-size:11px; max-height:150px; overflow:auto; color:var(--ink);
    white-space:pre-wrap; word-break:break-word; }
  .legend { display:flex; gap:16px; margin-top:8px; font-size:12px;
    color:var(--muted); align-items:center; }
  .legend .dot { display:inline-block; width:10px; height:10px; border-radius:50%;
    margin-right:5px; vertical-align:middle; }
  code { background:#21262d; padding:1px 5px; border-radius:4px; font-size:12px; }
</style>
</head>
<body>
<header>
  <div>
    <h1>halo_comm channel graph</h1>
    <div class="sub">tracker &harr; generator link &middot; host <code id="host"></code></div>
    <div class="legend">
      <span><span class="dot" style="background:var(--up)"></span>live</span>
      <span><span class="dot" style="background:var(--down)"></span>down (no listener)</span>
    </div>
  </div>
  <div class="status" id="status"><span class="dot"></span><span id="status-text">connecting…</span></div>
</header>
<div class="wrap">
  <div class="canvas"><svg id="graph"></svg></div>
  <aside><h2>Channels</h2><div id="cards"></div></aside>
</div>
<script>
let DATA = __DATA__;
const POLL_MS = __POLL_MS__;
const SVGNS = "http://www.w3.org/2000/svg";
const el = (n, a={}) => { const e=document.createElementNS(SVGNS,n);
  for (const k in a) e.setAttribute(k, a[k]); return e; };
const svg = document.getElementById("graph");
const edgeEls = {};
let focus = null;
const GPOS = {};          // persistent {key: {x,y}} so drags survive re-renders
let DRAGGING = false;     // pause poll re-render while dragging
let LAST_SIG = "";        // block-set signature; re-layout only when it changes
let LAST_VB = null;       // frozen viewBox during a drag

// Fast layered layout (Sugiyama barycenter) — assign columns by longest-path
// rank, then order each column to minimize edge crossings via a few median/
// barycenter sweeps. O(sweeps * edges); microseconds for this graph. Returns
// {key: {x,y}}. Compact (contiguous columns) so nothing lands far off-screen.
function layoutGraph(nodeKeys, chanKeys, E) {
  const allKeys = [...nodeKeys, ...chanKeys];
  const rank = {}; allKeys.forEach(k => rank[k] = 0);
  for (let it = 0; it < allKeys.length; it++) {      // longest path (cycle-bounded)
    let ch = false;
    E.forEach(e => { if (rank[e.dst] < rank[e.src] + 1) { rank[e.dst] = rank[e.src] + 1; ch = true; } });
    if (!ch) break;
  }
  const layers = {}; allKeys.forEach(k => (layers[rank[k]] = layers[rank[k]] || []).push(k));
  const rs = Object.keys(layers).map(Number).sort((a, b) => a - b);
  const nbr = {}; allKeys.forEach(k => nbr[k] = []);
  E.forEach(e => { nbr[e.src].push(e.dst); nbr[e.dst].push(e.src); });
  const pos = {}; rs.forEach(r => layers[r].forEach((k, i) => pos[k] = i));
  const bary = (k, set) => {
    const xs = nbr[k].filter(n => set.has(n)).map(n => pos[n]);
    return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : pos[k];
  };
  for (let sweep = 0; sweep < 6; sweep++) {           // down/up barycenter sweeps
    const order = sweep % 2 ? rs.slice().reverse() : rs;
    order.forEach((r, ri) => {
      if (ri === 0) return;
      const set = new Set(layers[order[ri - 1]]);
      layers[r].sort((a, b) => bary(a, set) - bary(b, set));
      layers[r].forEach((k, i) => pos[k] = i);
    });
  }
  const COLW = 236, ROWH = 78, TOP = 40, X0 = 40;
  const maxRows = Math.max(1, ...rs.map(r => layers[r].length));
  const P = {};
  rs.forEach((r, ci) => {
    const list = layers[r], off = TOP + (maxRows - list.length) * ROWH / 2;
    list.forEach((k, i) => P[k] = {x: X0 + ci * COLW, y: off + i * ROWH});
  });
  return P;
}

function drawCards() {
  const cards = document.getElementById("cards");
  cards.innerHTML = "";
  DATA.channels.forEach(c => {
    const card = document.createElement("div");
    card.className = "card";
    const badge = c.live===true ? '<span class="badge up">live</span>'
                : c.live===false ? '<span class="badge down">down</span>'
                : `<span class="badge">${c.pattern}</span>`;
    let live = "";
    if (c.stats) {
      const s = c.stats;
      const age = s.last_seen_age==null ? "—" : s.last_seen_age + "s ago";
      live = `<div class="meta">rate <code>${s.hz} Hz</code> &middot; `
           + `tick <code>${s.last_tick==null?"—":s.last_tick}</code> &middot; last ${age}</div>`
           + (s.latest ? `<pre>${escapeHtml(s.latest).slice(0,1200)}</pre>` : ``);
    }
    card.innerHTML =
      `<div class="top"><span class="cname">${c.name}</span>${badge}</div>` +
      `<div class="meta">@ <code>${c.host}:${c.port}</code>`
        + (c.node ? ` &middot; node <code>${c.node}</code>` : ``)
        + (c.direction ? ` &middot; ${c.direction}` : ``) + `</div>` +
      `<div class="meta">${c.types}</div>` + live;
    card.addEventListener("mouseenter", () => setFocus(c.name));
    card.addEventListener("mouseleave", () => setFocus(null));
    cards.appendChild(card);
  });
}

function escapeHtml(s){ return s.replace(/[&<>]/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[m])); }
function setFocus(name){ focus = name;
  Object.entries(edgeEls).forEach(([n, g]) => g.classList.toggle("dim", name!==null && n!==name)); }

function renderAll() {
  document.getElementById("host").textContent = "registry " + DATA.registry;
  drawGraph();                          // rqt_graph-style node<->channel graph
  drawCards();
  if (focus) setFocus(focus);
}

// rqt_graph view: one NODE per process (rounded pill), CHANNELS as topic-
// RECTANGLE / service-DIAMOND. Data flow: publish/call = node->channel; subscribe/
// serve = channel->node. On EACH block, input edges attach to one side and output
// edges to the OPPOSITE side (never the same side); which side is in vs out is
// chosen per block from where its neighbours sit, and re-decided while dragging.
// Layered (rank) initial layout; every shape draggable; canvas grows to fit.
function drawGraph() {
  const rows = DATA.channels || [];
  const nodeMap = {}, chanMap = {}, raw = [];
  rows.forEach(c => {
    const nk = c.node + "@" + c.host;
    (nodeMap[nk] = nodeMap[nk] || {label: c.node, host: c.host});
    const ch = chanMap[c.name] || (chanMap[c.name] = {name: c.name, topic: c.is_topic,
                                                      live: null, hz: 0});
    if (c.role !== "consumer") { ch.live = c.live; if (c.stats && c.stats.hz) ch.hz = c.stats.hz; }
    const toChan = (c.direction === "publishes" || c.direction === "calls");
    raw.push({a: "n:" + nk, b: "c:" + c.name, toChan});
  });
  const nodeKeys = Object.keys(nodeMap).sort().map(k => "n:" + k);
  const chanKeys = Object.keys(chanMap).sort().map(k => "c:" + k);
  const allKeys = [...nodeKeys, ...chanKeys];
  const NW = 178, NH = 46, CW = 168, CH = 46, VGAP = 30, TOP = 40, COLW = 250;
  const wOf = k => k[0] === "c" ? CW : NW, hOf = k => k[0] === "c" ? CH : NH;
  const cxOf = k => GPOS[k] ? GPOS[k].x + wOf(k) / 2 : 0;

  // directed edges (data flow src -> dst); a = node, b = channel
  const seen = {}, E = [];
  raw.forEach(e => {
    const k = e.a + ">" + e.b + (e.toChan ? "1" : "0"); if (seen[k]) return; seen[k] = 1;
    e.src = e.toChan ? e.a : e.b; e.dst = e.toChan ? e.b : e.a; E.push(e);
  });

  // (re)layout only when the block SET changes (crossing-minimized, compact), so
  // manual drags persist across the live poll.
  const sig = allKeys.slice().sort().join("|");
  if (sig !== LAST_SIG) {
    const P = layoutGraph(nodeKeys, chanKeys, E);
    Object.keys(GPOS).forEach(k => delete GPOS[k]);
    Object.assign(GPOS, P);
    LAST_SIG = sig;
  }

  // group edges per block: as source (out) / as dest (in)
  const outB = {}, inB = {};
  E.forEach(e => { (outB[e.src] = outB[e.src] || []).push(e); (inB[e.dst] = inB[e.dst] || []).push(e); });

  // per block: pick out-side & in-side (opposite) from neighbour x. All inputs
  // share one anchor on the in-side, all outputs one anchor on the out-side
  // (arrows may overlap there — fine); the only rule is in and out on OPPOSITE sides.
  allKeys.forEach(k => {
    const outs = outB[k] || [], ins = inB[k] || [];
    const mean = (arr, other) => arr.length
      ? arr.reduce((s, e) => s + cxOf(other(e)), 0) / arr.length : cxOf(k);
    const outMean = mean(outs, e => e.dst), inMean = mean(ins, e => e.src);
    let outRight;
    if (outs.length && ins.length) outRight = outMean >= inMean;   // out toward its side
    else if (outs.length) outRight = outMean >= cxOf(k);
    else outRight = !(inMean >= cxOf(k));
    const rX = GPOS[k].x + wOf(k), lX = GPOS[k].x, midY = GPOS[k].y + hOf(k) / 2;
    const outX = outRight ? rX : lX, inX = outRight ? lX : rX;
    outs.forEach(e => { e.sx = outX; e.sy = midY; });
    ins.forEach(e => { e.dx = inX; e.dy = midY; });
  });

  // viewBox = bounding box of all shapes + pad, with a minimum size so small
  // graphs don't over-zoom. Frozen while dragging so scaling stays consistent.
  // The svg fills the pane (CSS 100% + preserveAspectRatio meet), so every block
  // is always visible — the whole graph scales to fit.
  svg.innerHTML = "";
  let vb;
  if (DRAGGING && LAST_VB) {
    vb = LAST_VB;
  } else {
    let minX = 1e9, minY = 1e9, maxX = -1e9, maxY = -1e9;
    allKeys.forEach(k => {
      const p = GPOS[k]; if (!p) return;
      minX = Math.min(minX, p.x); minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x + wOf(k)); maxY = Math.max(maxY, p.y + hOf(k));
    });
    if (!isFinite(minX)) { minX = 0; minY = 0; maxX = 200; maxY = 120; }
    const PAD = 50; minX -= PAD; minY -= PAD; maxX += PAD; maxY += PAD;
    let w = maxX - minX, h = maxY - minY;
    const MINW = 900, MINH = 500;
    if (w < MINW) { minX -= (MINW - w) / 2; w = MINW; }
    if (h < MINH) { minY -= (MINH - h) / 2; h = MINH; }
    vb = {x: minX, y: minY, w, h}; LAST_VB = vb;
  }
  svg.setAttribute("viewBox", `${vb.x} ${vb.y} ${vb.w} ${vb.h}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  const defs = el("defs");
  for (const [id, col] of [["ah", "var(--edge)"], ["ah-up", "var(--up)"]]) {
    const m = el("marker", {id, viewBox: "0 0 10 10", refX: 9, refY: 5,
      markerWidth: 7, markerHeight: 7, orient: "auto"});
    m.appendChild(el("path", {d: "M0,0 L10,5 L0,10 z", fill: col}));
    defs.appendChild(m);
  }
  svg.appendChild(defs);

  // edges: out-port of src -> in-port of dst; control points leave/enter along the
  // side each port is on, so the curve exits the out-side and enters the in-side.
  E.forEach(e => {
    if (e.sx == null || e.dx == null) return;
    const dirS = e.sx === GPOS[e.src].x ? -1 : 1;
    const dirD = e.dx === GPOS[e.dst].x ? -1 : 1;
    const c1x = e.sx + dirS * 55, c2x = e.dx + dirD * 55;
    const up = chanMap[e.b.slice(2)] && chanMap[e.b.slice(2)].live === true;
    const p = el("path", {
      d: `M ${e.sx} ${e.sy} C ${c1x} ${e.sy}, ${c2x} ${e.dy}, ${e.dx} ${e.dy}`,
      stroke: up ? "var(--up)" : "var(--edge)", fill: "none", "stroke-width": 2,
      opacity: 0.8, "marker-end": `url(#${up ? "ah-up" : "ah"})`});
    svg.appendChild(p);
  });

  const drag = (g, key) => {
    g.style.cursor = "grab";
    g.addEventListener("mousedown", ev => {
      ev.preventDefault();
      const r = svg.getBoundingClientRect(), vb = svg.viewBox.baseVal;
      const sx = vb.width / r.width, sy = vb.height / r.height;
      const ox = GPOS[key].x, oy = GPOS[key].y, mx = ev.clientX, my = ev.clientY;
      DRAGGING = true; g.style.cursor = "grabbing";
      const mv = e2 => {
        GPOS[key] = {x: ox + (e2.clientX - mx) * sx, y: oy + (e2.clientY - my) * sy};
        drawGraph();
      };
      const up = () => { DRAGGING = false; removeEventListener("mousemove", mv);
                         removeEventListener("mouseup", up); };
      addEventListener("mousemove", mv); addEventListener("mouseup", up);
    });
  };

  // one pill per node
  nodeKeys.forEach(key => {
    const p = GPOS[key], n = nodeMap[key.slice(2)];
    const g = el("g", {});
    g.appendChild(el("rect", {x: p.x, y: p.y, width: NW, height: NH, rx: NH / 2,
      fill: "var(--gen)", opacity: 0.92, stroke: "var(--line)"}));
    const t = el("text", {x: p.x + NW / 2, y: p.y + 20, "text-anchor": "middle",
      class: "name", fill: "#fff"}); t.textContent = n.label; g.appendChild(t);
    const s = el("text", {x: p.x + NW / 2, y: p.y + 35, "text-anchor": "middle",
      "font-size": "10", fill: "rgba(255,255,255,.8)"}); s.textContent = n.host; g.appendChild(s);
    drag(g, key); svg.appendChild(g);
  });

  // channels: topic = rectangle, service = diamond
  chanKeys.forEach(key => {
    const p = GPOS[key], c = chanMap[key.slice(2)];
    const col = c.live === true ? "var(--up)" : c.live === false ? "var(--down)" : "var(--edge)";
    const cx = p.x + CW / 2, cy = p.y + CH / 2;
    const g = el("g", {});
    if (c.topic) {
      g.appendChild(el("rect", {x: p.x, y: p.y, width: CW, height: CH, rx: 4,
        fill: "var(--panel)", stroke: col, "stroke-width": 1.5}));
    } else {
      const hw = CW / 2, hh = CH / 2 + 5;
      g.appendChild(el("polygon", {
        points: `${cx},${cy - hh} ${cx + hw},${cy} ${cx},${cy + hh} ${cx - hw},${cy}`,
        fill: "var(--panel)", stroke: col, "stroke-width": 1.5}));
    }
    const t = el("text", {x: cx, y: cy - 1, "text-anchor": "middle", class: "name",
      "font-size": "12"}); t.textContent = c.name; g.appendChild(t);
    const sub = el("text", {x: cx, y: cy + 13, "text-anchor": "middle", class: "types",
      "font-size": "10"});
    sub.textContent = (c.topic ? "topic" : "service") + (c.hz ? `  ${c.hz} Hz` : "");
    g.appendChild(sub);
    drag(g, key); svg.appendChild(g);
  });

  if (!nodeKeys.length) {
    const t = el("text", {x: 30, y: 60, class: "types"});
    t.textContent = "registry has no live endpoints — is anything running?";
    svg.appendChild(t);
  }
}
renderAll();

const st = document.getElementById("status"), stText = document.getElementById("status-text");
async function poll() {
  try {
    const r = await fetch("/api/graph", {cache:"no-store"});
    DATA = await r.json();
    if (!DRAGGING) renderAll();            // don't yank the graph out from under a drag
    st.classList.add("on"); stText.textContent = "live · " + new Date().toLocaleTimeString();
  } catch (e) { st.classList.remove("on"); stText.textContent = "reconnecting…"; }
}
poll();
setInterval(poll, POLL_MS);
</script>
</body>
</html>
"""
