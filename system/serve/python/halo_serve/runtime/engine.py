#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Serving engine: ONE GPU thread (`halo-serve-gpu`) runs every registered local workload.

One thread because Metal has no cross-process priority and concurrent MLX threads corrupted
grammar-constrained decoding. Scheduling is EDF plus:
  1. preempt at layer boundaries (`yield_points()`) when an earlier-deadline job is waiting;
  2. start a preemptible long job first while the most urgent job still has slack;
  3. a hard (`policy`) job preempts a soft one regardless of deadline.
Routine boundaries use `mx.async_eval`; a blocking sync is forced after `backlog_ms`.
Not hard real time: Metal cannot preempt a kernel.
"""
from __future__ import annotations

import collections
import heapq
import itertools
import json
import math
import os
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

mx = None                                                   # mlx.core, loaded by the first Engine


def _load_mx():
    """Import MLX lazily so this module imports on hosts without MLX."""
    global mx
    if mx is None:
        try:
            import mlx.core as _mx
            mx = _mx
        except ImportError:
            pass
    return mx


def _pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


@dataclass
class _Entry:
    name: str
    wl: Any
    deadline_ms: Optional[float]
    preemptible: bool = False
    hard: bool = False                  # hard real-time (motion policy): preempts any soft job
    costs: collections.deque = field(default_factory=lambda: collections.deque(maxlen=64))
    lat: List[float] = field(default_factory=list)
    done: int = 0
    misses: int = 0
    errors: int = 0
    last_error: str = ""


@dataclass(order=True)
class _Job:
    deadline: float
    seq: int
    release: float = field(compare=False)
    name: str = field(compare=False)
    req: Any = field(compare=False)
    fut: Future = field(compare=False)


class Engine:
    def __init__(self, backlog_ms: float = 12.0, start_guard_ms: float = 10.0):
        self.backlog_ms = backlog_ms
        self.start_guard_ms = start_guard_ms
        _load_mx()
        self._entries: Dict[str, _Entry] = {}
        self._heap: List[_Job] = []
        self._cv = threading.Condition()
        self._seq = itertools.count()
        self._cur: Optional[_Job] = None
        self._nest = 0
        self._nested_s = 0.0
        self._t_sync = 0.0
        self._busy_s = 0.0
        self._t0 = time.perf_counter()
        self._patched: set = set()
        self.gpu_tid = None                 # set by the GPU thread; must exist before start()
        self.meta: dict = {}                # per-workload labels (backends, steps) shown in stats()
        # HALO_SERVE_TRACE=1: one row per job in dump().
        self._trace = [] if os.environ.get("HALO_SERVE_TRACE") == "1" else None
        # Guaranteed share for a running soft job: a hard job preempts it only while preemptions
        # took <= (1 - soft_share) of its wall time, or when waiting would miss the hard deadline.
        # 0 = always preempt.
        self.soft_share = 0.6
        self.share_guard_ms = 30.0
        self._cur_t0 = 0.0
        self._deferred = 0
        self._thread = threading.Thread(target=self._loop, name="halo-serve-gpu", daemon=True)
        self._thread.start()
        while self.gpu_tid is None:
            time.sleep(0.001)

    def register(self, name: str, wl, deadline_ms: float = None):
        """Register a workload; it is LOADED on the GPU thread."""
        if name in self._entries:
            raise ValueError(f"workload {name!r} already registered")
        e = _Entry(name, wl, deadline_ms, hard=getattr(wl, "kind", "") == "policy")
        self._entries[name] = e
        try:
            self.call_on_gpu(wl.load).result()
            classes = wl.yield_points() or []
            for cls in classes:
                self._hook(cls)
        except BaseException:
            self._entries.pop(name, None)          # so a retry is possible
            raise
        e.preemptible = bool(classes)
        return e

    def call_on_gpu(self, fn: Callable, *a) -> Future:
        """Run an arbitrary callable on the GPU thread, ahead of any job."""
        fut: Future = Future()
        with self._cv:
            heapq.heappush(self._heap, _Job(-math.inf, next(self._seq), 0.0, "__call__",
                                            (fn, a), fut))
            self._cv.notify_all()
        return fut

    def submit(self, name: str, req: Any, deadline_ms: float = None) -> Future:
        """Enqueue one job. Returns a Future of the workload's Result."""
        e = self._entries[name]
        rel = time.perf_counter()
        dl_ms = deadline_ms or e.deadline_ms
        dl = rel + (dl_ms / 1e3 if dl_ms else 3600.0)
        fut: Future = Future()
        with self._cv:
            heapq.heappush(self._heap, _Job(dl, next(self._seq), rel, name, req, fut))
            self._cv.notify_all()
        return fut

    def _loop(self):
        self.gpu_tid = threading.get_ident()
        while True:
            with self._cv:
                job = self._pick_locked()
                if job is None:
                    self._cv.wait(timeout=0.05)
                    continue
            self._execute(job)

    def _pick_locked(self) -> Optional[_Job]:
        now = time.perf_counter()
        if not self._heap:
            return None
        ready = self._heap
        edf = min(ready)
        hard = [j for j in ready if j.name != "__call__" and self._entries[j.name].hard]
        if hard and edf.name != "__call__" and not self._entries[edf.name].hard:
            # rule 3; rule 2 below may still start a soft job while the hard one has slack
            edf = min(hard)
        choice = edf
        if edf.name != "__call__" and not self._entries[edf.name].preemptible:
            pre = [j for j in ready if j.name != "__call__" and self._entries[j.name].preemptible]
            if pre:
                est = _pct(list(self._entries[edf.name].costs), 0.9) or 0.0
                slack = edf.deadline - now - est
                if slack > self.start_guard_ms / 1e3:
                    choice = min(pre)          # rule 2: start the long job first
        self._heap.remove(choice)
        heapq.heapify(self._heap)
        return choice

    def _execute(self, job: _Job):
        if job.name == "__call__":
            fn, a = job.req
            try:
                job.fut.set_result(fn(*a))
            except BaseException as ex:                         # noqa: BLE001
                job.fut.set_exception(ex)
            return
        e = self._entries[job.name]
        outer_cur, outer_nested, outer_t0 = self._cur, self._nested_s, self._cur_t0
        self._cur, self._nested_s = job, 0.0
        t0 = time.perf_counter()
        self._cur_t0 = t0
        try:
            res = e.wl.run(job.req)
            err = None
        except BaseException as ex:                             # noqa: BLE001
            res, err = None, ex
        t1 = time.perf_counter()
        own = (t1 - t0) - self._nested_s                        # exclusive of preemptions
        self._cur, self._nested_s = outer_cur, outer_nested + (t1 - t0 if outer_cur else 0.0)
        self._cur_t0 = outer_t0
        self._busy_s += own
        e.costs.append(own)
        e.done += 1
        lat = (t1 - job.release) * 1e3
        e.lat.append(lat)
        if e.deadline_ms and lat > e.deadline_ms:
            e.misses += 1
        if self._trace is not None:
            self._trace.append((job.name, job.release - self._t0, t0 - self._t0, t1 - self._t0,
                                own, self._nest > 0, outer_cur.name if outer_cur else ""))
        if err is not None:
            e.errors += 1
            e.last_error = f"{type(err).__name__}: {err}"[:300]
            job.fut.set_exception(err)
        else:
            job.fut.set_result(res)

    def _hook(self, cls: type):
        if cls in self._patched:
            return
        orig = cls.__call__
        engine = self

        def patched(this, *a, **kw):
            out = orig(this, *a, **kw)
            engine._boundary(out)
            return out

        cls.__call__ = patched
        self._patched.add(cls)

    def _boundary(self, out):
        # Only a preemptible job on the GPU thread yields, never while nested: workloads may
        # share a layer class, and re-entering would deadlock.
        if (mx is None or threading.get_ident() != self.gpu_tid
                or self._cur is None or self._nest > 0
                or not self._entries[self._cur.name].preemptible):
            return
        now = time.perf_counter()
        urgent = None
        cur_hard = self._entries[self._cur.name].hard
        with self._cv:
            cands = [j for j in self._heap if j.name != "__call__" and j.release <= now
                     and (j.deadline < self._cur.deadline
                          or (self._entries[j.name].hard and not cur_hard))]
            if cands:
                urgent = min(cands)
                if (not cur_hard and self.soft_share > 0 and self._entries[urgent.name].hard
                        and not self._share_allows(urgent, now)):
                    urgent = None
                    self._deferred += 1
                else:
                    self._heap.remove(urgent)
                    heapq.heapify(self._heap)
        if urgent is None:
            if now - self._t_sync >= self.backlog_ms / 1e3:
                mx.eval(out); self._t_sync = time.perf_counter()
            else:
                mx.async_eval(out)
            return
        mx.eval(out)                                            # drain, then preempt
        self._nest += 1
        try:
            self._execute(urgent)
        finally:
            self._nest -= 1
            self._t_sync = time.perf_counter()

    def _share_allows(self, urgent, now) -> bool:
        """May hard job `urgent` preempt the running soft job now (see soft_share)?"""
        elapsed = now - self._cur_t0
        if self._nested_s <= (1.0 - self.soft_share) * elapsed:
            return True
        est = _pct(list(self._entries[urgent.name].costs), 0.5) or 0.0
        # margin for the next boundary, one orchestrator step away
        return urgent.deadline - now - est <= self.share_guard_ms / 1e3

    def stats(self) -> dict:
        wall = time.perf_counter() - self._t0
        out = dict(wall_s=wall, gpu_busy_frac=self._busy_s / wall if wall else 0.0,
                   soft_share=self.soft_share, preemptions_deferred=self._deferred,
                   workloads={})
        for e in self._entries.values():
            out["workloads"][e.name] = dict(
                kind=getattr(e.wl, "kind", "?"), preemptible=e.preemptible,
                deadline_ms=e.deadline_ms, done=e.done, misses=e.misses,
                errors=e.errors, last_error=e.last_error,
                lat_p50=_pct(e.lat, .5), lat_p95=_pct(e.lat, .95),
                lat_max=max(e.lat) if e.lat else None,
                cost_p50=_pct([c * 1e3 for c in e.costs], .5),
                achieved_hz=e.done / wall if wall else 0.0)
            # labels never override a measured field
            for k, v in self.meta.get(e.name, {}).items():
                out["workloads"][e.name].setdefault(k, v)
        return out

    def reset_stats(self):
        for e in self._entries.values():
            e.lat.clear(); e.done = e.misses = e.errors = 0
        self._busy_s = 0.0; self._t0 = time.perf_counter(); self._deferred = 0

    def dump(self, path: str):
        with open(path, "w") as f:
            st = self.stats()
            if self._trace is not None:
                st["trace_cols"] = ["name", "release", "start", "end", "own_s", "nested", "preempted"]
                st["trace"] = list(self._trace)
            json.dump(st, f, indent=2, default=str)


_GLOBAL: Optional[Engine] = None
_GLOCK = threading.Lock()


def global_engine() -> Engine:
    """Process-wide engine, so every model in a process shares ONE GPU thread."""
    global _GLOBAL
    with _GLOCK:
        if _GLOBAL is None:
            _GLOBAL = Engine()
        return _GLOBAL
