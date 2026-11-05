#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""End-to-end runner + CLI for the motion quality checker."""

from __future__ import annotations

import argparse
import glob
import os

from motion_gen.motion_check.config import CheckConfig
from motion_gen.motion_check.episode import MotionEpisode
from motion_gen.motion_check.schema import HARD, MotionQualityReport, Status


def run_episode(
    episode: MotionEpisode, cfg: CheckConfig | None = None, use_mujoco: bool = True, tier: str | None = None
) -> MotionQualityReport:
    """Run the full check battery + judge on one episode (wraps ``MotionQualityEvaluator``)."""
    from motion_gen.motion_check.evaluator import MotionQualityEvaluator

    cfg = cfg or CheckConfig.for_task(episode.task_type, tier=tier)
    evaluator = MotionQualityEvaluator.for_task(episode.task_type, cfg=cfg, use_mujoco=use_mujoco)
    return evaluator.evaluate(episode)


def discover_episodes(dataset_dir: str) -> list[tuple[str, str | None]]:
    """Return ``[(path, tag)]`` for every episode under ``dataset_dir``."""
    dataset_dir = os.path.expanduser(dataset_dir)
    out: list[tuple[str, str | None]] = []
    tagged = sorted(glob.glob(os.path.join(dataset_dir, "*_qpos.npy")))
    for p in tagged:
        tag = os.path.basename(p)[: -len("_qpos.npy")]
        out.append((dataset_dir, tag))
    if not out:
        for p in sorted(glob.glob(os.path.join(dataset_dir, "*", "g1_qpos.npy"))):
            out.append((os.path.dirname(p), None))
    return out


def run_dataset(
    dataset_dir: str,
    task: str | None = None,
    cfg: CheckConfig | None = None,
    limit: int | None = None,
    use_mujoco: bool = True,
    tier: str | None = None,
) -> list[MotionQualityReport]:
    eps = discover_episodes(dataset_dir)
    if limit:
        eps = eps[:limit]
    reports: list[MotionQualityReport] = []
    for path, tag in eps:
        ep = MotionEpisode.load(path, tag=tag)
        if task:
            ep.task_type = task
        ecfg = cfg or CheckConfig.for_task(ep.task_type, tier=tier)
        reports.append(run_episode(ep, ecfg, use_mujoco=use_mujoco))
    return reports


_MARK = {Status.PASS: "PASS", Status.WARN: "WARN", Status.FAIL: "FAIL", Status.SKIP: "skip"}


def _shown_status(r) -> Status:
    """Status as printed: a failed soft check only lowers the score, so it shows as WARN."""
    if r.status == Status.FAIL and r.severity != HARD:
        return Status.WARN
    return r.status


def print_report(report: MotionQualityReport, show_trace: bool = False) -> None:
    v = report.verdict
    head = f"{report.episode_id}  [{report.task_type}]  T={report.n_frames}"
    print("=" * 72)
    print(head)
    if report.checker_id:
        print(report.checker_id)
    print("-" * 72)
    for r in report.results:
        t = "" if not r.time else f"  @{r.time[:4]}{'…' if len(r.time) > 4 else ''}"
        shown = _shown_status(r)
        print(f"  [{_MARK[shown]}] {r.name:22s} val={_fmt(r.value):>10}  score={r.score:.2f}{t}")
        if r.comment and shown in (Status.FAIL, Status.WARN):
            print(f"         {r.comment}")
        if r.trace and (show_trace or shown in (Status.FAIL, Status.WARN)):
            for step in r.trace:
                print(f"           · {step}")
    print("-" * 72)
    if v:
        print(
            f"VERDICT: {'ACCEPT ✅' if v.accept else 'REJECT ❌'}  "
            f"score={v.score:.3f}" + (f"  blocking={v.blocking}" if v.blocking else "")
        )
    print("=" * 72)


def summarize(reports: list[MotionQualityReport]) -> None:
    n = len(reports)
    if not n:
        print("no episodes found")
        return
    n_accept = sum(1 for r in reports if r.verdict and r.verdict.accept)
    print("\n" + "#" * 72)
    print(f"DATASET SUMMARY: {n_accept}/{n} accepted ({100.0 * n_accept / n:.0f}%)")
    counts: dict[str, list[int]] = {}
    for rep in reports:
        for r in rep.results:
            c = counts.setdefault(r.name, [0, 0, 0])  # fail (hard), warn (incl. soft fails), skip
            shown = _shown_status(r)
            if shown == Status.FAIL:
                c[0] += 1
            elif shown == Status.WARN:
                c[1] += 1
            elif shown == Status.SKIP:
                c[2] += 1
    print(f"{'check':24s} {'fail':>6} {'warn':>6} {'skip':>6}  of {n}")
    for name in sorted(counts):
        f, w, s = counts[name]
        print(f"{name:24s} {f:>6} {w:>6} {s:>6}")
    print("#" * 72)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, dict):
        return "{…}"
    return str(v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Motion quality checker")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--episode", help="episode dir or qpos .npy file")
    g.add_argument("--dataset", help="dir of episodes to batch-check")
    ap.add_argument("--tag", default=None, help="episode tag within --episode dir (e.g. ep0)")
    ap.add_argument("--task", default=None, help="override task_type")
    ap.add_argument(
        "--tier",
        default=None,
        choices=["strict", "industry", "academic"],
        help="strictness tier: strict=strictest (reject-more), academic=kindest",
    )
    ap.add_argument("--config-dir", default=None, help="motion_check yaml overrides dir")
    ap.add_argument("--limit", type=int, default=None, help="max episodes to check (--dataset)")
    ap.add_argument("--out", default=None, help="write report/summary JSON here")
    ap.add_argument("--no-mujoco", action="store_true", help="skip MuJoCo track building")
    ap.add_argument("--fps", type=float, default=30.0, help="frame rate of the qpos (--episode)")
    ap.add_argument(
        "--trace", action="store_true", help="print the justification trace for every checker (not just flagged ones)"
    )
    args = ap.parse_args(argv)

    use_mujoco = not args.no_mujoco
    if args.episode:
        ep = MotionEpisode.load(args.episode, tag=args.tag, fps=args.fps)
        if args.task:
            ep.task_type = args.task
        cfg = CheckConfig.for_task(ep.task_type, config_dir=args.config_dir, tier=args.tier)
        report = run_episode(ep, cfg, use_mujoco=use_mujoco)
        print_report(report, show_trace=args.trace)
        if args.out:
            report.save(args.out)
            print(f"wrote {args.out}")
        return 0 if (report.verdict and report.verdict.accept) else 1

    cfg = CheckConfig.for_task(args.task, config_dir=args.config_dir, tier=args.tier) if args.task else None
    reports = run_dataset(
        args.dataset, task=args.task, cfg=cfg, limit=args.limit, use_mujoco=use_mujoco, tier=args.tier
    )
    for rep in reports:
        print_report(rep, show_trace=args.trace)
    summarize(reports)
    if args.out:
        import json

        with open(args.out, "w") as f:
            json.dump([r.to_dict() for r in reports], f, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
