#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Verdict aggregation: turn a list of ``CheckResult`` into an accept/reject"""

from __future__ import annotations

from motion_gen.motion_check.config import CheckConfig
from motion_gen.motion_check.schema import HARD, SOFT, CheckResult, Status, Verdict


class RuleJudge:
    def __init__(self, cfg: CheckConfig | None = None) -> None:
        self.cfg = cfg or CheckConfig()

    def decide(self, results: list[CheckResult]) -> Verdict:
        scored = [r for r in results if r.status != Status.SKIP]

        blocking = [r.name for r in scored if self._eff_sev(r) == HARD and not r.passed]

        if scored:
            num = sum(self._w(r.name) * r.score for r in scored)
            den = sum(self._w(r.name) for r in scored) or 1.0
            score = num / den
        else:
            score = 1.0

        accept = (len(blocking) == 0) and (score >= self.cfg.accept_min_score)

        feedback: list[str] = []
        for r in sorted(scored, key=_feedback_rank):
            if r.passed and r.status != Status.WARN:
                continue
            if r.passed and r.status == Status.WARN:
                line = f"[warn] {r.name}: {r.comment}"
            else:
                tag = "BLOCK" if self._eff_sev(r) == HARD else "soft"
                line = f"[{tag}] {r.name}: {r.comment}"
            hint = self.cfg.hints.get(r.name)
            if hint:
                line += f"  -> {hint}"
            feedback.append(line)

        return Verdict(accept=accept, score=round(float(score), 4), blocking=blocking, feedback=feedback)

    def _w(self, name: str) -> float:
        return float(self.cfg.weights.get(name, 1.0))

    def _eff_sev(self, r: CheckResult):
        """Effective severity at decision time, honoring cfg.severity_overrides"""
        ov = (getattr(self.cfg, "severity_overrides", None) or {}).get(r.name)
        if ov == "soft":
            return SOFT
        if ov == "hard":
            return HARD
        return r.severity


def _feedback_rank(r: CheckResult) -> tuple:
    if not r.passed and r.severity == HARD:
        pri = 0
    elif not r.passed:
        pri = 1
    elif r.status == Status.WARN:
        pri = 2
    else:
        pri = 3
    return (pri, r.name)
