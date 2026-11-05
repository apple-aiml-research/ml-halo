#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Observation-variant support shared by the G1 deploy entry points (MuJoCo, real, web).

The tracker variants differ ONLY by their ``observation:`` spec, so nothing here selects a variant by name:
everything is derived from the loaded spec.

    proprio student  configs/g1_v2.yaml        obs (434) + policy_temporal (30,166)
    depth student    configs/g1_v2_depth.yaml  + student_depth (1,48,84) + a depth source

WHY A PREFLIGHT THAT FAILS. A depth variant degrades SILENTLY when its dependency is missing: without a depth source
the builder still emits a far plane, which looks like a working run with a policy acting on an empty scene. So each
check reports what it found and, by default, refuses to continue, with an explicit ``--allow_*`` flag for the cases
where running degraded is the intent.
"""

from __future__ import annotations

import sys
from typing import Any, Optional

# The depth encode clip (near_m, far_m) the tracker trains with:
# training/.../g1/motion_tracking_scene_distill_cfg.py::DEPTH_CLIP. A wrong clip rescales every pixel while the tensor
# shapes and the ONNX signature still match, so nothing downstream can detect it.
EXPECTED_DEPTH_CLIP: tuple[float, float] = (0.2, 2.5)


def image_group(spec) -> Optional[Any]:
    """The spec's image (depth) group, or None. One place to ask 'does this policy need depth?'."""
    for g in spec.groups.values():
        if g.layout == "image":
            return g
    return None


def check_odom(spec, backend, timeout: float = 2.0) -> bool:
    """``anchor_mode: odom`` needs live odometry (rt/odommodestate). True = safe to proceed.

    Run before Start so a missing odometry stream fails while the robot is still safe. Backends without
    an odometry age (simulation) pass.
    """
    if not hasattr(backend, "wait_for_odom"):
        return True
    if backend.wait_for_odom(timeout=timeout):
        return True
    print(f"[variant] ERROR: anchor_mode=odom but no odometry on rt/odommodestate within {timeout:.1f} s "
          f"(age {backend.odom_age_s():.3f} s). The motion anchor would have no frame. Check that the "
          f"robot's state estimator is publishing.", file=sys.stderr)
    return False


def describe(spec, *, depth_hold_steps: int) -> str:
    """One-line variant summary for the startup log — the fastest way to spot a wrong config.

    ``depth_hold_steps`` is the resolved hold (``observation_builder.depth_hold_steps``). Rates assume the 50 Hz
    control loop.
    """
    parts = []
    for name, g in spec.groups.items():
        if g.layout == "image":
            rate = f"@{50.0 / depth_hold_steps:g}Hz hold{depth_hold_steps} (declared decim{g.obs_decim})"
            parts.append(f"{name}(image 1x{g.height}x{g.width} clip{list(g.clip)} {rate})")
        else:
            parts.append(f"{name}({g.layout},{len(g.terms)} terms)")
    return (f"anchor={spec.anchor_mode} history={spec.history_length} "
            f"groups=[{', '.join(parts)}]" + (" consumes=[depth]" if image_group(spec) is not None else ""))


# IsaacLab "world"-convention camera frame (+X forward, +Z up) -> MuJoCo/OpenGL camera frame (-Z forward, +Y up),
# applied by RIGHT multiplication: q_mujoco_body = q_wxyz_world (x) THIS.
_WORLD_CONV_TO_MUJOCO_CAM = (0.5, 0.5, -0.5, -0.5)


def depth_camera_rig(blk: dict) -> dict:
    """The config's ``depth_camera:`` block as a resolved rig.

    Converts ``quat_wxyz_world`` (how training states the mount, ``convention="world"``) into the MuJoCo body-frame
    quaternion the renderer needs.
    """
    w1, x1, y1, z1 = (float(v) for v in blk["quat_wxyz_world"])
    w2, x2, y2, z2 = _WORLD_CONV_TO_MUJOCO_CAM
    return {
        "name": blk.get("render_camera", "depth_cam"),
        "parent_body": blk.get("parent_body", "torso_link"),
        "pos": [float(v) for v in blk["pos"]],
        "quat_wxyz": [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2],
        "fovy": float(blk.get("fovy", 59.26)),
    }


def compile_scene_with_depth_camera(xml_path: str, depth_camera: dict):
    """Compile ``xml_path`` with the config's depth camera injected. Returns ``(model, data, rig)``.

    The training camera is not part of the robot XML: it is described only by the config's ``depth_camera:`` block
    and injected here, so it survives scene changes.
    """
    import mujoco

    rig = depth_camera_rig(depth_camera)
    spec = mujoco.MjSpec.from_file(xml_path)
    parent = spec.body(rig["parent_body"])  # returns None (does not raise) for a missing body
    if parent is None:
        raise ValueError(f"depth_camera.parent_body '{rig['parent_body']}' not found in {xml_path}")
    cam = parent.add_camera()
    cam.name = rig["name"]
    cam.pos = rig["pos"]
    cam.quat = rig["quat_wxyz"]
    cam.fovy = rig["fovy"]
    cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
    model = spec.compile()
    return model, mujoco.MjData(model), rig


def make_mujoco_depth_source(spec, model, data, camera: str):
    """MuJoCo depth source sized and clipped FROM THE SPEC. None if the policy needs no depth."""
    g = image_group(spec)
    if g is None:
        return None
    from depth_sources import MuJoCoDepthSource
    return MuJoCoDepthSource(model, data, camera, (g.height, g.width), float(g.clip[1]))


def make_realsense_depth_source(spec):
    """In-process RealSense depth source sized/clipped from the spec. None if the policy needs no depth.

    In-process means the camera must be attached to the machine running the controller.
    """
    g = image_group(spec)
    if g is None:
        return None
    from depth_sources import RealSenseDepthSource
    return RealSenseDepthSource((g.height, g.width), float(g.clip[1]))


def check_depth_clip(spec) -> list[str]:
    """Problems if the config's depth clip is not the clip the tracker trains with (``EXPECTED_DEPTH_CLIP``)."""
    g = image_group(spec)
    if g is None:
        return []
    got = (round(float(g.clip[0]), 6), round(float(g.clip[1]), 6))
    if got == EXPECTED_DEPTH_CLIP:
        return []
    return [f"depth clip is {got[0]}/{got[1]} m but the tracker trains at {EXPECTED_DEPTH_CLIP[0]}/"
            f"{EXPECTED_DEPTH_CLIP[1]} m (DEPTH_CLIP in motion_tracking_scene_distill_cfg.py). Every pixel would be "
            f"mis-scaled and nothing downstream can detect it. Fix `observation.groups.{g.name}.clip`, or pass "
            f"--allow_depth_clip_mismatch for a checkpoint trained with a different clip."]


def setup_real(config, backend, *, allow_no_depth: bool = False, allow_depth_clip_mismatch: bool = False):
    """Resolve the spec, bring up the depth source (if the policy needs one) and preflight — for the REAL robot.

    Returns ``(spec, depth_source)``, or ``None`` if the preflight refused (the caller should clean up the backend
    and exit non-zero). ``depth_source`` is None for variants that need no depth. Also checks live odometry for
    ``anchor_mode: odom`` (``check_odom``). Shared by every real entry point so the checks cannot drift apart.
    """
    from utils.core.observation_builder import ObservationSpec
    spec = ObservationSpec.from_dict(config.observation)

    depth_source = None
    if image_group(spec) is not None:
        try:
            depth_source = make_realsense_depth_source(spec)
            ok, meta = depth_source.start()
            print(f"[INFO] RealSense depth: {('OK ' + str(meta)) if ok else ('FAILED ' + str(meta))}")
            if not ok:
                depth_source.close()
                depth_source = None
        except Exception as exc:                          # noqa: BLE001
            print(f"[WARN] depth source unavailable: {exc}", file=sys.stderr)
            depth_source = None

    ok = preflight(spec, backend=backend, depth_source=depth_source, allow_no_depth=allow_no_depth,
                   allow_depth_clip_mismatch=allow_depth_clip_mismatch)
    ok = check_odom(spec, backend) and ok
    if not ok:
        if depth_source is not None:
            depth_source.close()
        return None
    return spec, depth_source


def _check_depth_dependency(spec, depth_source, allow_no_depth: bool, read_frame: bool = True) -> bool:
    """student_depth needs a live source of the right geometry and far plane. True = safe to proceed.

    ``read_frame=False`` checks the source's declared geometry only. Use it for the MuJoCo source when the control
    loop runs on another thread: its GL renderer is pinned to the first thread that renders.
    """
    g = image_group(spec)
    if g is None:
        return True
    if depth_source is None:
        print(f"[variant] ERROR: this policy has an image group ({g.name}) but no depth source was created. The "
              f"builder would emit a constant FAR plane and the policy would act on an empty scene while appearing "
              f"to run. Pass --allow_no_depth to accept that deliberately.", file=sys.stderr)
        return allow_no_depth

    if not read_frame:
        out_hw = tuple(getattr(depth_source, "out_hw", (g.height, g.width)))
        src_far = float(getattr(depth_source, "far_m", g.clip[1]))
        if out_hw != (g.height, g.width) or src_far < float(g.clip[1]) - 1e-6:
            print(f"[variant] ERROR: depth source geometry {out_hw} / far {src_far} m does not match the spec "
                  f"({g.height}, {g.width}) / clip far {g.clip[1]} m.", file=sys.stderr)
            return allow_no_depth
        print(f"[variant] student_depth: source {out_hw}, far {src_far} m, clip {list(g.clip)}")
        return True

    ok = True
    try:
        frame = depth_source.get_depth()
        far = float(g.clip[1])
        print(f"[variant] student_depth: source live, frame {frame.shape} "
              f"range [{frame.min():.3f}, {frame.max():.3f}] m, clip {list(g.clip)}")
        # Shape: the builder cannot encode a frame of another size.
        if tuple(frame.shape[-2:]) != (g.height, g.width):
            print(f"[variant] ERROR: depth source returns {tuple(frame.shape[-2:])} but the spec wants "
                  f"({g.height}, {g.width}).", file=sys.stderr)
            ok = allow_no_depth
        # The source fills invalid / no-return pixels with its own far_m and the encode clips afterwards: a far_m
        # above the clip is harmless, one below encodes every invalid pixel too near.
        src_far = getattr(depth_source, "far_m", None)
        if src_far is not None and float(src_far) < far - 1e-6:
            print(f"[variant] ERROR: depth source far_m={float(src_far)} m is BELOW the spec's clip far {far} m.",
                  file=sys.stderr)
            ok = allow_no_depth
        if frame.min() == frame.max() and abs(float(frame.min()) - far) < 1e-6:
            print(f"[variant] WARNING: the first depth frame is a uniform FAR plane ({far} m) — the camera may "
                  f"not be streaming yet.", file=sys.stderr)
    except Exception as exc:                     # noqa: BLE001
        print(f"[variant] ERROR: depth source failed on first read: {exc}", file=sys.stderr)
        ok = allow_no_depth
    return ok


def preflight(spec, *, backend, depth_source=None, allow_no_depth: bool = False,
              allow_depth_clip_mismatch: bool = False, read_depth_frame: bool = True) -> bool:
    """Report the variant and verify its dependencies. False = do not proceed.

    Called by every entry point before the control loop starts, so a missing dependency surfaces while the robot
    is still safe rather than as degraded behaviour mid-run.
    """
    from utils.core.observation_builder import depth_hold_steps
    ok = True
    hold = depth_hold_steps(spec, bool(getattr(backend, "IS_SIMULATION", False)))
    print(f"[variant] {describe(spec, depth_hold_steps=hold)}")

    for msg in check_depth_clip(spec):
        print(f"[variant] {'WARNING' if allow_depth_clip_mismatch else 'ERROR'}: {msg}", file=sys.stderr)
        ok = ok and allow_depth_clip_mismatch

    # Runs even after a clip error, so a failing run reports every problem in one pass.
    depth_ok = _check_depth_dependency(spec, depth_source, allow_no_depth, read_frame=read_depth_frame)
    ok = ok and depth_ok

    if not ok:
        print("[variant] preflight FAILED — refusing to start. Re-run with the matching --allow_* flag if running "
              "degraded is intended.", file=sys.stderr)
    return ok
