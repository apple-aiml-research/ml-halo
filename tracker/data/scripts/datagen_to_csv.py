#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Convert HALO datagen sit-on-chair episodes into tracker source CSVs, bound to the lab sofa.

datagen (``../../../datagen``) writes walk -> turn -> sit episodes on a chair at 30 fps. The tracker
trains sit clips on one fixed terrain object, the lab sofa (``assets/objects/labsofa/labsofa.urdf``), whose mesh is
datagen's ``arm_chair`` mesh. Each episode is therefore moved by one rigid planar transform (yaw about Z +
XY translation, heights kept) that puts its chair exactly onto the sofa pose of the URDF, and written as a CSV
``labsofa_<tag>.csv`` that ``extract_sequences.py`` / ``batch_process.py`` turn into training NPZs.

Inputs (either):
  --parquet DIR a rendered parquet dataset (``data/episode_NNNNNN.parquet`` + ``episode_specs/episode_NNNNNN.json``):
                pelvis pose ``observation.body_pos[0]`` / ``observation.body_quat[0]`` (wxyz) and
                ``observation.joint_pos``, in the scene frame, lifted to the spec's ``floor_z``.
  --specs DIR   motion-stage episodes (``**/g1_qpos.npy`` [T, 36] = xyz, quat wxyz, 29 joints, next to
                ``episode_spec.json`` and optionally ``meta.json``); placed builds work too.

The chair pose is the spec's ``placements`` entry with ``role == "target"`` (``pos_xyz``, ``quat_wxyz``); the
datagen chair mesh is xy-centred with its base at z = 0, like ``labsofa.obj``, so its pose maps onto the URDF
joint pose of the sofa directly. Joints are already in the deployment / CSV order of ``extract_sequences.py``.

Output CSV row: ``x, y, z, qx, qy, qz, qw, joint_0 ... joint_28`` at 30 fps, plus ``manifest.json`` (one record per
episode: source, chair asset, frames, applied yaw).

Usage (from tracker/):
    python data/scripts/datagen_to_csv.py --specs <datagen prefix>/specs --accepted_only --out data/datagen_csv/labsofa
    python data/scripts/batch_process.py --input-dir data/datagen_csv/labsofa \
        --output-dir data/g1_processed/robot-terrain/labsofa --headless
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

FPS = 30
TRACKER_ROOT = Path(__file__).resolve().parents[2]
SOFA_URDF = TRACKER_ROOT.parent / "assets" / "objects" / "labsofa" / "labsofa.urdf"


# -- quaternions (wxyz) -----------------------------------------------------------------------------------------------

def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def _yaw_of(q_wxyz) -> float:
    w, x, y, z = (float(v) for v in q_wxyz)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# -- the lab sofa pose ------------------------------------------------------------------------------------------------

def sofa_pose(urdf: Path = SOFA_URDF) -> tuple[np.ndarray, float]:
    """(xy, yaw) of the sofa in the env frame, from the URDF's fixed world joint.

    The mesh is Y-up; the joint's ``rpy = (pi/2, 0, yaw)`` (fixed-axis x, y, z) is the Y-up -> Z-up turn followed
    by a yaw about Z, the same convention datagen uses for its chairs.
    """
    joint = next(j for j in ET.parse(urdf).getroot().iter("joint") if j.find("parent").get("link") == "world")
    origin = joint.find("origin")
    xyz = [float(v) for v in origin.get("xyz").split()]
    roll, pitch, yaw = (float(v) for v in origin.get("rpy").split())
    if abs(roll - math.pi / 2) > 1e-3 or abs(pitch) > 1e-3:
        raise ValueError(f"{urdf}: expected rpy (pi/2, 0, yaw) on the world joint, got {(roll, pitch, yaw)}")
    return np.array(xyz[:2]), yaw


# -- episode readers --------------------------------------------------------------------------------------------------

def _chair(spec: dict) -> tuple[np.ndarray, float, str]:
    target = [p for p in spec.get("placements", []) if p.get("role") == "target"]
    if len(target) != 1:
        raise ValueError(f"expected one placement with role 'target', found {len(target)}")
    t = target[0]
    asset = Path(t.get("asset_path", t.get("name", ""))).stem
    return np.asarray(t["pos_xyz"][:2], dtype=np.float64), _yaw_of(t["quat_wxyz"]), asset


def read_parquet(root: Path):
    """Yield (tag, qpos [T, 36] xyz + wxyz + joints with floor_z removed, spec, extra) per parquet episode."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("--parquet needs pyarrow: pip install pyarrow")
    files = sorted(glob.glob(str(root / "data" / "episode_*.parquet")))
    if not files:
        sys.exit(f"no data/episode_*.parquet under {root}")
    for f in files:
        ep = re.search(r"episode_(\d+)\.parquet$", f).group(1)
        spec_file = root / "episode_specs" / f"episode_{ep}.json"
        if not spec_file.is_file():
            print(f"[skip] episode {ep}: no {spec_file.name}")
            continue
        spec = json.loads(spec_file.read_text())
        table = pq.read_table(f, columns=["observation.body_pos", "observation.body_quat", "observation.joint_pos"])
        cols = table.to_pydict()
        pos = np.array([np.asarray(b[0], dtype=np.float64) for b in cols["observation.body_pos"]])
        quat = np.array([np.asarray(b[0], dtype=np.float64) for b in cols["observation.body_quat"]])
        joints = np.array(cols["observation.joint_pos"], dtype=np.float64)
        pos[:, 2] -= float(spec.get("floor_z", 0.0))
        yield f"datagen_ep{ep}", np.concatenate([pos, quat, joints], axis=1), spec, {"source": f}


def read_specs(root: Path):
    """Yield (tag, qpos [T, 36], spec, extra) per motion-stage episode directory."""
    files = sorted(glob.glob(str(root / "**" / "g1_qpos.npy"), recursive=True))
    if not files:
        sys.exit(f"no **/g1_qpos.npy under {root}")
    for f in files:
        d = Path(f).parent
        spec_file = d / "episode_spec.json"
        if not spec_file.is_file():
            print(f"[skip] {d}: no episode_spec.json")
            continue
        spec = json.loads(spec_file.read_text())
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").is_file() else {}
        qpos = np.load(f).astype(np.float64)
        qpos[:, 2] -= float(spec.get("floor_z", 0.0))
        tag = "datagen_" + "_".join(d.relative_to(root).parts) if d != root else "datagen_" + d.name
        yield re.sub(r"[^A-Za-z0-9_]+", "_", tag), qpos, spec, {"source": str(d),
                                                                  "accepted": meta.get("final_gen_accepted")}


# -- conversion -------------------------------------------------------------------------------------------------------

def to_sofa_frame(qpos: np.ndarray, chair_xy: np.ndarray, chair_yaw: float, sofa_xy: np.ndarray,
                  sofa_yaw: float) -> tuple[np.ndarray, float]:
    """Rigidly move ``qpos`` (xyz, quat wxyz, joints) so the chair lands on the sofa. Returns (qpos, dyaw)."""
    dyaw = _wrap(sofa_yaw - chair_yaw)
    c, s = math.cos(dyaw), math.sin(dyaw)
    rot = np.array([[c, -s], [s, c]])
    out = qpos.copy()
    out[:, :2] = (qpos[:, :2] - chair_xy) @ rot.T + sofa_xy
    qz = np.array([math.cos(dyaw / 2), 0.0, 0.0, math.sin(dyaw / 2)])
    q = _quat_mul(np.broadcast_to(qz, qpos[:, 3:7].shape), qpos[:, 3:7])
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q *= np.where(q[:, :1] < 0.0, -1.0, 1.0)
    out[:, 3:7] = q
    return out, dyaw


def to_csv_rows(qpos: np.ndarray, hold_frames: int) -> np.ndarray:
    """[T, 36] xyz + wxyz + joints -> CSV rows xyz + xyzw + joints, with ``hold_frames`` copies of the last row."""
    rows = np.concatenate([qpos[:, :3], qpos[:, 4:7], qpos[:, 3:4], qpos[:, 7:]], axis=1)
    if hold_frames > 0:
        rows = np.concatenate([rows, np.repeat(rows[-1:], hold_frames, axis=0)], axis=0)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--parquet", type=Path, help="parquet dataset root (data/, episode_specs/)")
    src.add_argument("--specs", type=Path, help="motion-stage episodes root (**/g1_qpos.npy + episode_spec.json)")
    ap.add_argument("--out", type=Path, required=True, help="output folder for labsofa_*.csv and manifest.json")
    ap.add_argument("--hold_s", type=float, default=1.0,
                    help="seconds of seated hold appended (the episodes end on the sit, without a hold)")
    ap.add_argument("--accepted_only", action="store_true",
                    help="--specs: keep only episodes whose meta.json has final_gen_accepted = true")
    ap.add_argument("--chairs", nargs="+", default=None,
                    help="keep only these chair assets (e.g. arm_chair); default: all")
    ap.add_argument("--limit", type=int, default=None, help="convert at most this many episodes")
    args = ap.parse_args()
    if args.accepted_only and args.parquet:
        ap.error("--accepted_only needs --specs: the rendered dataset has no meta.json gate verdict")

    sofa_xy, sofa_yaw = sofa_pose()
    args.out.mkdir(parents=True, exist_ok=True)
    reader = read_parquet(args.parquet) if args.parquet else read_specs(args.specs)
    hold = max(0, round(args.hold_s * FPS))
    manifest, n_skip = [], 0
    for tag, qpos, spec, extra in reader:
        if args.limit is not None and len(manifest) >= args.limit:
            break
        if spec.get("task_type", "sit") != "sit":
            print(f"[skip] {tag}: task_type {spec.get('task_type')!r} (only sit episodes bind to the sofa)")
            n_skip += 1
            continue
        if args.accepted_only and extra.get("accepted") is not True:
            why = "no meta.json" if extra.get("accepted") is None else "final_gen_accepted = false"
            print(f"[skip] {tag}: {why} (--accepted_only)")
            n_skip += 1
            continue
        if qpos.ndim != 2 or qpos.shape[1] != 36 or len(qpos) < 2 or not np.isfinite(qpos).all():
            print(f"[skip] {tag}: bad qpos {qpos.shape}")
            n_skip += 1
            continue
        chair_xy, chair_yaw, asset = _chair(spec)
        if args.chairs and asset not in args.chairs:
            n_skip += 1
            continue
        moved, dyaw = to_sofa_frame(qpos, chair_xy, chair_yaw, sofa_xy, sofa_yaw)
        name = f"labsofa_{tag}.csv"
        np.savetxt(args.out / name, to_csv_rows(moved, hold), delimiter=",", fmt="%.7f")
        end = moved[-1, :2] - sofa_xy
        manifest.append({"csv": name, "source": extra["source"], "chair": asset, "frames": len(qpos),
                         "hold_frames": hold, "dyaw_rad": round(dyaw, 6), "accepted": extra.get("accepted"),
                         "end_pelvis_from_sofa_m": round(float(np.hypot(*end)), 3)})
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[datagen_to_csv] wrote {len(manifest)} CSVs to {args.out} ({n_skip} skipped); sofa pose "
          f"xy=({sofa_xy[0]:.4f}, {sofa_xy[1]:.4f}) yaw={sofa_yaw:.4f} rad")
    return 0 if manifest else 1


if __name__ == "__main__":
    sys.exit(main())
