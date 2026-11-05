#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""MotionQualityEvaluator — the top-level container."""

from __future__ import annotations

from motion_gen.motion_check.config import CheckConfig
from motion_gen.motion_check.episode import MotionEpisode
from motion_gen.motion_check.judge import RuleJudge
from motion_gen.motion_check.schema import MotionQualityReport


class MotionQualityEvaluator:
    def __init__(self, cfg=None, checkers=None, judge=None, use_mujoco=True):
        import motion_gen.motion_check.checks  # noqa: F401 — populate the registry
        from motion_gen.motion_check.checks.base import REGISTRY

        self.cfg: CheckConfig = cfg or CheckConfig()
        self.checkers = list(checkers) if checkers is not None else list(REGISTRY)
        self.judge = judge or RuleJudge(self.cfg)
        self.use_mujoco = use_mujoco

    @classmethod
    def for_task(
        cls,
        task: str | None,
        cfg: CheckConfig | None = None,
        use_mujoco: bool = True,
        config_dir: str | None = None,
        tier: str | None = None,
    ) -> "MotionQualityEvaluator":
        """Build an evaluator holding the checkers applicable to ``task`` with"""
        import motion_gen.motion_check.checks  # noqa: F401
        from motion_gen.motion_check.checks.base import get_checkers

        cfg = cfg or CheckConfig.for_task(task, config_dir=config_dir, tier=tier)
        return cls(cfg=cfg, checkers=get_checkers(task), judge=RuleJudge(cfg), use_mujoco=use_mujoco)

    def evaluate(self, episode: MotionEpisode) -> MotionQualityReport:
        from motion_gen.motion_check.checks.base import CheckContext, run_checks

        tracks = None
        if self.use_mujoco:
            try:
                from motion_gen.motion_check.kinematics import build_tracks

                tracks = build_tracks(episode, self.cfg)
            except Exception as e:  # noqa: BLE001 — geometric checkers self-skip w/o tracks
                print(f"[motion_check] MuJoCo tracks unavailable ({e!r}); array-only checkers", flush=True)

        ctx = CheckContext(episode=episode, cfg=self.cfg, tracks=tracks)
        results = run_checks(ctx, self.checkers)

        verdict = self.judge.decide(results)
        from motion_gen.motion_check.version import CHECKER_VERSION, checker_id

        return MotionQualityReport(
            episode_id=episode.episode_id,
            task_type=episode.task_type,
            n_frames=episode.T,
            fps=episode.fps,
            results=results,
            verdict=verdict,
            checker_version=CHECKER_VERSION,
            checker_id=checker_id(self.cfg, self.checkers),
            extra={"quality_tier": self.cfg.quality_tier} if self.cfg.quality_tier else {},
        )
