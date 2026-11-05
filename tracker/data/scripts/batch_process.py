#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Batch CSV → NPZ converter for the motion-tracking data pipeline.

Walks ``--input-dir`` for ``*.csv`` motion clips and runs
``extract_sequences.py`` on them, ``--chunk_size`` clips per Isaac Sim launch, writing the
resulting ``<name>.npz`` files into ``--output-dir``. Defaults route raw clips
in ``data/`` to ``data/g1_processed/robot_only/converted/``, a dataset
folder of the training README's layout. Unrecognized
flags (e.g. ``--headless``, ``--output_fps``) are forwarded to
``extract_sequences.py``.

Examples:
    # Convert every CSV in data/ (defaults)
    python batch_process.py --headless

    # Convert a specific subset
    python batch_process.py --motion_names walk1_subject1 dance2_subject1 --headless

    # Skip clips already present in the output directory
    python batch_process.py --skip-existing --headless
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "g1_processed" / "robot_only" / "converted"
EXTRACT_SCRIPT = Path(__file__).resolve().parent / "extract_sequences.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
        help=f"Directory of input .csv motion clips (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Where the .npz files are written (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--motion_names", nargs="+", default=None,
        help="Motion base names (without extension). Default: all CSVs in --input-dir.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip motions whose output NPZ already exists.",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=50,
        help="Clips replayed per Isaac Sim launch (the launch costs far more than one replay; default 50).",
    )
    parser.add_argument(
        "--timeout_s", type=float, default=120.0,
        help="Per-clip time budget; a launch is killed after chunk_size * timeout_s + 300 s (default 120).",
    )
    # Everything else (e.g. --headless) is forwarded verbatim to extract_sequences.py.
    args, extra = parser.parse_known_args()

    if not args.input_dir.is_dir():
        parser.error(f"--input-dir does not exist: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.motion_names:
        names = sorted(args.motion_names)
    else:
        names = sorted(p.stem for p in args.input_dir.glob("*.csv"))
    if not names:
        print(f"[batch_process] No CSV motion clips found in {args.input_dir}")
        return 0

    print(f"[batch_process] {len(names)} motion(s) -> {args.output_dir}")
    todo: list[tuple[str, Path, Path]] = []
    for name in names:
        csv_path = args.input_dir / f"{name}.csv"
        npz_path = args.output_dir / f"{name}.npz"
        if not csv_path.exists():
            print(f"[batch_process] SKIP {name} (no CSV at {csv_path})")
            continue
        if args.skip_existing and npz_path.exists():
            print(f"[batch_process] SKIP {name} (output exists)")
            continue
        todo.append((name, csv_path, npz_path))

    failures: list[str] = []
    chunk_size = max(1, args.chunk_size)
    for start in range(0, len(todo), chunk_size):
        chunk = todo[start:start + chunk_size]
        for _, _, npz_path in chunk:
            npz_path.unlink(missing_ok=True)  # a stale output must not count as converted
        cmd = [
            sys.executable, str(EXTRACT_SCRIPT),
            "--input", *[str(c) for _, c, _ in chunk],
            "--output", *[str(n) for _, _, n in chunk],
            *extra,
        ]
        print(f"[batch_process] clips {start + 1}-{start + len(chunk)} of {len(todo)}")
        try:
            rc = subprocess.run(cmd, timeout=len(chunk) * args.timeout_s + 300.0).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        missing = [name for name, _, npz_path in chunk if not npz_path.exists()]
        if rc != 0 or missing:
            print(f"[batch_process] chunk exit {rc}; {len(missing)} clip(s) not written: {missing}")
            failures.extend(missing)

    if failures:
        print(f"[batch_process] {len(failures)} failed: {failures}")
        return 1
    print("[batch_process] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())