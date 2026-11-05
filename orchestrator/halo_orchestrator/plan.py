#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Immutable, versioned mission-plan snapshot.

Every transition is pure (returns a new Plan), so a worker thread can install a new plan with one
atomic reference swap while the tick loop reads a stable snapshot without a lock.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Step:
    id: str
    subtask: str
    title: str
    status: str = "pending"          # pending | in_progress | completed | failed | skipped | cancelled

    def snapshot(self) -> dict:
        """Step's slice of the JSON dict."""
        return {
            "id": self.id,
            "subtask": self.subtask,
            "title": self.title,
            "status": self.status,
        }


@dataclass(frozen=True)
class Plan:
    run_id: str
    version: int
    revision: int
    stamp: float
    mission: str
    current: int
    done: bool
    steps: tuple[Step, ...] = ()
    # T1's answer to the operator; the only output for a conversational request with no steps.
    reply: str = ""

    @property
    def authored(self) -> bool:
        """True once T1 has authored this plan (version 0 is reserved for the placeholder).

        Provenance, not a skill gate. The tick loop uses `authored and not steps` to hold without a
        T2 call: T1 answered and chose no action, unlike an unauthored empty plan.
        """
        return self.version > 0

    def close_current(self) -> "Plan":
        """Mark the current step `completed` and open the next one."""
        return self._close_current("completed")

    def fail_current(self) -> "Plan":
        """Like `close_current`, but mark the current step `failed`."""
        return self._close_current("failed")

    def _close_current(self, status: str) -> "Plan":
        if not self.steps:
            return self
        now = time.time()
        idx = self.current
        steps = list(self.steps)
        steps[idx] = replace(steps[idx], status=status)
        next_idx = idx + 1 if idx + 1 < len(steps) else idx
        if next_idx != idx and steps[next_idx].status == "pending":
            steps[next_idx] = replace(steps[next_idx], status="in_progress")
        return replace(self, steps=tuple(steps), current=next_idx,
                        version=self.version + 1, revision=self.revision + 1, stamp=now)

    def close_all(self) -> "Plan":
        """Close every non-terminal step and mark the plan `done`."""
        now = time.time()
        steps = tuple(
            replace(s, status="completed")
            if s.status not in ("completed", "failed", "skipped") else s
            for s in self.steps
        )
        return replace(self, steps=steps, done=True,
                        version=self.version + 1, revision=self.revision + 1, stamp=now)

    def stop_unfinished(self) -> "Plan":
        """Cancel every unfinished step (new operator instruction), so the orchestrator sees the old
        plan as over rather than a step still in progress to carry on with."""
        if not self.steps:
            return self
        now = time.time()
        steps = tuple(
            replace(s, status="cancelled")
            if s.status in ("pending", "in_progress") else s
            for s in self.steps)
        # NOT `done`: done=True would latch "mission complete" before the new plan lands.
        return replace(self, steps=steps,
                       version=self.version + 1, revision=self.revision + 1, stamp=now)

    def reopen(self, idx: int) -> "Plan":
        """Move the pointer to `idx` and reopen that step; progress need not be monotonic."""
        now = time.time()
        steps = list(self.steps)
        steps[idx] = replace(steps[idx], status="in_progress")
        return replace(self, steps=tuple(steps), current=idx,
                        version=self.version + 1, revision=self.revision + 1, stamp=now)

    def insert_step(self, idx: int, step: "Step") -> "Plan":
        """Insert `step` at `idx` as a resumable detour that runs first.

        The step previously at `idx` goes back to pending, so a later `close_current` resumes it.
        Past the end, the step is appended."""
        now = time.time()
        steps = list(self.steps)
        if 0 <= idx < len(steps) and steps[idx].status == "in_progress":
            steps[idx] = replace(steps[idx], status="pending")
        step = replace(step, status="in_progress")
        steps.insert(min(idx, len(steps)), step)
        return replace(self, steps=tuple(steps), current=min(idx, len(steps) - 1),
                        version=self.version + 1, revision=self.revision + 1, stamp=now)

    def snapshot(self) -> dict:
        """JSON dict published on g1/plan."""
        return {
            "run_id": self.run_id,
            "version": self.version,
            "revision": self.revision,
            "stamp": self.stamp,
            "mission": self.mission,
            "authored": self.authored,
            "reply": self.reply,
            "current": self.current,
            "done": self.done,
            "steps": [s.snapshot() for s in self.steps],
        }


def unauthored(run_id: str, mission: str = "", steps: tuple[Step, ...] = ()) -> Plan:
    """Placeholder Plan before T1 has authored anything (version 0)."""
    return Plan(run_id=run_id, version=0, revision=0, stamp=time.time(),
                mission=mission,
                current=0, done=False, steps=steps)
