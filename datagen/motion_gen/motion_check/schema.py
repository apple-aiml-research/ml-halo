#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Result schema for the motion quality checker."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence


class Status(str, Enum):
    """Per-check outcome. ``str`` mixin → serializes to its value in JSON."""

    PASS = "pass"  # behaviour is clean
    WARN = "warn"  # borderline / soft-degraded but not a hard failure
    FAIL = "fail"  # undesired behaviour detected
    SKIP = "skip"  # check not applicable / needed data unavailable


HARD = "hard"
SOFT = "soft"


def mask_to_intervals(mask: Sequence[bool]) -> list[list[int]]:
    """Contiguous ``True`` runs in a boolean mask → ``[[start, end], ...]``"""
    intervals: list[list[int]] = []
    start: int | None = None
    n = len(mask)
    for i in range(n):
        if mask[i] and start is None:
            start = i
        elif not mask[i] and start is not None:
            intervals.append([start, i - 1])
            start = None
    if start is not None:
        intervals.append([start, n - 1])
    return intervals


def frames_to_intervals(frames: Sequence[int]) -> list[list[int]]:
    """Sorted/unsorted frame indices → contiguous ``[[start, end], ...]``."""
    fs = sorted(set(int(f) for f in frames))
    intervals: list[list[int]] = []
    for f in fs:
        if intervals and f == intervals[-1][1] + 1:
            intervals[-1][1] = f
        else:
            intervals.append([f, f])
    return intervals


def _jsonable(obj: Any) -> Any:
    """Coerce numpy scalars/arrays and enums to JSON-native types."""
    if isinstance(obj, Status):
        return obj.value
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except (ValueError, TypeError):
            pass
    if hasattr(obj, "tolist"):  # numpy array
        return obj.tolist()
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


@dataclass
class CheckResult:
    """One check's outcome."""

    name: str
    status: Status
    value: Any  # measured quantity ("check_value")
    passed: bool  # boolean flag (True = acceptable)
    score: float = 1.0  # [0, 1], 1 = perfect
    time: list[list[int]] | None = None  # occurrence intervals [[s, e], ...]
    comment: str = ""  # one-line human summary / feedback
    severity: str = SOFT  # HARD | SOFT
    trace: list[str] = field(default_factory=list)  # justification trace (reasoning steps)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.status, str) and not isinstance(self.status, Status):
            self.status = Status(self.status)
        self.score = float(max(0.0, min(1.0, self.score)))

    def compact(self) -> list:
        """The exact ``[value, flag, score, time, comment]`` list the pipeline"""
        return [_jsonable(self.value), bool(self.passed), round(float(self.score), 4), self.time, self.comment]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "value": _jsonable(self.value),
            "passed": bool(self.passed),
            "score": round(float(self.score), 4),
            "time": self.time,
            "comment": self.comment,
            "severity": self.severity,
            "trace": list(self.trace),
            "meta": _jsonable(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CheckResult":
        return cls(
            name=d["name"],
            status=Status(d["status"]),
            value=d.get("value"),
            passed=bool(d["passed"]),
            score=float(d.get("score", 1.0)),
            time=d.get("time"),
            comment=d.get("comment", ""),
            severity=d.get("severity", SOFT),
            trace=list(d.get("trace", [])),
            meta=d.get("meta", {}) or {},
        )


@dataclass
class Verdict:
    """The final accept/reject decision for one episode."""

    accept: bool
    score: float  # overall [0, 1]
    blocking: list[str] = field(default_factory=list)  # failed HARD check names
    feedback: list[str] = field(default_factory=list)  # regeneration hints

    def to_dict(self) -> dict:
        return {
            "accept": bool(self.accept),
            "score": round(float(self.score), 4),
            "blocking": list(self.blocking),
            "feedback": list(self.feedback),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Verdict":
        return cls(
            accept=bool(d["accept"]),
            score=float(d.get("score", 0.0)),
            blocking=list(d.get("blocking", [])),
            feedback=list(d.get("feedback", [])),
        )


@dataclass
class MotionQualityReport:
    """All checks for one episode, plus the aggregated verdict."""

    episode_id: str
    task_type: str
    n_frames: int
    fps: float
    results: list[CheckResult] = field(default_factory=list)
    verdict: Verdict | None = None
    checker_version: str = ""
    checker_id: str = ""
    extra: dict = field(default_factory=dict)

    def compact(self) -> dict[str, list]:
        """The requested ``{check_item: [value, flag, score, time, comment]}``."""
        return {r.name: r.compact() for r in self.results}

    def failed(self, include_soft: bool = True) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and (include_soft or r.severity == HARD)]

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "task_type": self.task_type,
            "checker_version": self.checker_version,
            "checker_id": self.checker_id,
            "n_frames": int(self.n_frames),
            "fps": float(self.fps),
            "verdict": self.verdict.to_dict() if self.verdict else None,
            "checks": {r.name: r.to_dict() for r in self.results},
            "compact": self.compact(),
            "extra": _jsonable(self.extra),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str, indent: int = 2) -> None:
        with open(path, "w") as f:
            f.write(self.to_json(indent=indent))

    @classmethod
    def from_dict(cls, d: dict) -> "MotionQualityReport":
        checks = d.get("checks", {})
        results = [CheckResult.from_dict(v) for v in checks.values()]
        verdict = Verdict.from_dict(d["verdict"]) if d.get("verdict") else None
        return cls(
            episode_id=d["episode_id"],
            task_type=d["task_type"],
            n_frames=int(d.get("n_frames", 0)),
            fps=float(d.get("fps", 30.0)),
            results=results,
            verdict=verdict,
            checker_version=d.get("checker_version", ""),
            checker_id=d.get("checker_id", ""),
            extra=d.get("extra", {}) or {},
        )
