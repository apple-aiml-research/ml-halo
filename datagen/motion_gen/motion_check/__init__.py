#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Motion quality checker for Kimodo-generated G1 motion."""

from motion_gen.motion_check.checks.base import MotionQualityChecker, register, run_checks
from motion_gen.motion_check.config import CheckConfig
from motion_gen.motion_check.episode import MotionEpisode
from motion_gen.motion_check.evaluator import MotionQualityEvaluator
from motion_gen.motion_check.judge import RuleJudge
from motion_gen.motion_check.schema import (
    HARD,
    SOFT,
    CheckResult,
    MotionQualityReport,
    Status,
    Verdict,
    frames_to_intervals,
    mask_to_intervals,
)

__all__ = [
    "MotionQualityEvaluator",
    "MotionQualityChecker",
    "register",
    "run_checks",
    "CheckConfig",
    "MotionEpisode",
    "RuleJudge",
    "CheckResult",
    "MotionQualityReport",
    "Verdict",
    "Status",
    "HARD",
    "SOFT",
    "frames_to_intervals",
    "mask_to_intervals",
]
