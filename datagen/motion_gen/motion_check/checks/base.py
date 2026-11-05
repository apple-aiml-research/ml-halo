#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""MotionQualityChecker — the design unit of the checker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from motion_gen.motion_check.config import CheckConfig
from motion_gen.motion_check.episode import MotionEpisode
from motion_gen.motion_check.schema import SOFT, CheckResult, Status

if TYPE_CHECKING:  # avoid importing kinematics (mujoco) at runtime
    from motion_gen.motion_check.kinematics import Tracks


@dataclass
class CheckContext:
    """Everything a checker needs, computed once per episode."""

    episode: MotionEpisode
    cfg: CheckConfig
    tracks: Optional["Tracks"] = None  # None when MuJoCo tracks weren't built

    @property
    def fps(self) -> float:
        return self.episode.fps


class MotionQualityChecker:
    """Base class for one atomic quality check."""

    name: str = ""  # unique id (also the CheckResult name)
    severity: str = SOFT  # HARD blocks acceptance; SOFT lowers score
    tasks: tuple[str, ...] | None = None  # None = every task; else only these task_types
    description: str = ""  # one-line human description of what it checks
    needs_tracks: bool = False  # auto-SKIP when ctx.tracks is None

    def applies(self, task: str) -> bool:
        return self.tasks is None or task in self.tasks

    @property
    def group(self) -> str:
        """Registry-facing check id (== ``name``). Kept as a distinct accessor so"""
        return self.name

    def check(self, ctx: CheckContext) -> CheckResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def result(
        self,
        *,
        passed: bool,
        score: float,
        value: Any,
        status: Status | None = None,
        time: list[list[int]] | None = None,
        comment: str = "",
        trace: list[str] | None = None,
        meta: dict | None = None,
    ) -> CheckResult:
        if status is None:
            status = Status.PASS if passed else Status.FAIL
        return CheckResult(
            name=self.name,
            status=status,
            value=value,
            passed=passed,
            score=score,
            time=time,
            comment=comment,
            severity=self.severity,
            trace=trace or [],
            meta=meta or {},
        )

    def skip(self, reason: str) -> CheckResult:
        return CheckResult(
            name=self.name,
            status=Status.SKIP,
            value=None,
            passed=True,
            score=1.0,
            comment=reason,
            severity=self.severity,
            trace=[f"SKIP: {reason}"],
        )


REGISTRY: list[MotionQualityChecker] = []


def register(cls):
    """Class decorator: instantiate the checker and add it to the registry."""
    REGISTRY.append(cls())
    return cls


def get_checkers(task: str | None = None) -> list[MotionQualityChecker]:
    return [c for c in REGISTRY if task is None or c.applies(task)]


def run_checks(ctx: CheckContext, checkers: list["MotionQualityChecker"] | None = None) -> list[CheckResult]:
    """Run the given checkers (default: every registered one applicable to the"""
    results: list[CheckResult] = []
    task = ctx.episode.task_type
    if checkers is None:
        checkers = [c for c in REGISTRY if c.applies(task)]
    for chk in checkers:
        if chk.needs_tracks and ctx.tracks is None:
            results.append(chk.skip("no MuJoCo tracks (run with mujoco env)"))
            continue
        try:
            results.append(chk.check(ctx))
        except Exception as e:  # noqa: BLE001
            print(f"[motion_check] checker {chk.name!r} errored: {e!r}", flush=True)
            results.append(
                CheckResult(
                    name=chk.name,
                    status=Status.SKIP,
                    value=None,
                    passed=True,
                    score=1.0,
                    comment=f"checker errored: {e!r}",
                    severity=chk.severity,
                    trace=[f"ERROR: {e!r}"],
                    meta={"error": repr(e)},
                )
            )
    overrides = getattr(ctx.cfg, "severity_overrides", None) or {}
    for r in results:
        if r.name in overrides:
            r.severity = overrides[r.name]
    return results
