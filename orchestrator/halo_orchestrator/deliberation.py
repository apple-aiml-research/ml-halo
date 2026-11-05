#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Lock-free plan snapshot (`PlanRef`) and the T1 plan-authoring worker (`StaticDeliberator`)."""
from __future__ import annotations

import threading
import time
import traceback
from typing import Any, Callable, Optional

from .plan import Plan


class PlanRef:
    """The live `Plan` snapshot: lock-free `get()`, atomic `set()` by single attribute rebind.

    The tick loop must never take a lock held by the worker; an attribute rebind is atomic under
    the GIL and `Plan` is frozen, so readers always see a complete snapshot.
    """

    def __init__(self, initial: Plan):
        self._plan = initial

    def get(self) -> Plan:
        """Lock-free read of the current snapshot."""
        return self._plan

    def set(self, plan: Plan) -> None:
        """Atomically install a new snapshot (single attribute rebind — no lock)."""
        self._plan = plan


class StaticDeliberator:
    """Background thread running `deliberate_fn(prev_plan, payload) -> Plan` off the critical
    path and installing each result into `plan_ref`; an exception is logged and the plan kept."""

    def __init__(self, plan_ref: PlanRef, deliberate_fn: Callable[[Plan, Any], Plan],
                 logger: Callable[[str], None]):
        self._plan_ref = plan_ref
        self._deliberate_fn = deliberate_fn
        self._log = logger

        # Guards only the pending slot; never held by the tick's read path or across deliberate_fn.
        self._lock = threading.Lock()
        self._pending: Any = None

        self._wake = threading.Event()
        self._in_progress_since: Optional[float] = None

        threading.Thread(target=self._run, name="deliberation-worker", daemon=True).start()

    def request(self, payload: Any) -> None:
        """Non-blocking enqueue into a single slot: pending requests coalesce, latest payload wins."""
        with self._lock:
            self._pending = payload
        self._wake.set()

    @property
    def in_progress_since(self) -> Optional[float]:
        """Wall-clock start of the current `deliberate_fn` call, or `None` when idle."""
        return self._in_progress_since

    def _take_pending(self) -> Any:
        with self._lock:
            payload, self._pending = self._pending, None
            return payload

    def _run(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            # Re-polling the slot (not the Event) catches requests that arrive mid-invocation.
            while (payload := self._take_pending()) is not None:
                self._invoke(payload)

    def _invoke(self, payload: Any) -> None:
        self._in_progress_since = time.time()
        try:
            prev = self._plan_ref.get()
            new_plan = self._deliberate_fn(prev, payload)
            self._plan_ref.set(new_plan)
        except Exception:  # noqa: BLE001 — a failed attempt must not kill the worker
            self._log("deliberation worker error (keeping previous plan snapshot):\n"
                      + traceback.format_exc())
        finally:
            self._in_progress_since = None


__all__ = ["PlanRef", "StaticDeliberator"]
