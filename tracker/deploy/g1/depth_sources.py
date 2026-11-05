#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""In-process depth sources for the depth student's `student_depth` observation group.

The sources read the sensor (or render the MuJoCo camera) directly into the array the
ObservationBuilder consumes, in the controller's process. The RealSense source therefore needs the
camera attached to the computer running the controller. To get frames off-box for visualization or
recording, publish them from a separate low-rate thread, never from the control path.

CONTRACT (identical for every source, and identical to training):
  * `get_depth()` returns a contiguous ``(H, W)`` float32 array in **METERS**, raw — NOT encoded.
    The clip+log1p encode happens once, in the ObservationBuilder's image group, so the recording /
    viz path keeps true metric depth and there is exactly one place the encode can be wrong.
  * Downsampling uses a VALID-ONLY (mask-weighted) mean, and the FAR side is clamped BEFORE any
    aggregation. Both are load-bearing, and the ObservationBuilder's clip runs AFTER the resize so it
    can undo neither:
      - filling holes with far and then box-averaging biases every output pixel FARTHER with local
        hole density (about +0.3 m on a 0.5 m wall at a typical D435 dropout of ~15 %);
      - letting a saturated 65.54 m reading into a mean turns that same 0.5 m wall into 1.14 m at the
        ~101:1 downsample ratio from 848x480 to 84x48.
    Only a block with NO valid pixel reports far. Invalid = non-finite or ``<= 0`` (the RealSense
    "no return").
  * ``far_m`` MUST equal the far clip the checkpoint trained with (``DEPTH_CLIP`` in
    ``motion_tracking_scene_distill_cfg.py``, 2.5 m). Feeding a different far value shifts every
    invalid pixel (3.0 m instead of 2.5 m moves it by ~12 % of the encoded input range). It is
    required, not defaulted.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np


def _require_cv2():
    """OpenCV, which the valid-only area mean needs; raises a clear ImportError without it."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "depth downsampling needs OpenCV (pip install opencv-python-headless): the valid-only "
            "area mean has no exact numpy equivalent for the camera's resolution, and a nearest-pixel "
            "fallback would feed the policy a differently filtered image") from exc
    return cv2


def sanitize_depth(depth_m: np.ndarray, out_hw: tuple[int, int], far_m: float) -> np.ndarray:
    """Raw metric depth -> contiguous (H, W) float32 at ``out_hw``, in meters.

    Far-clamps, then downsamples with a VALID-ONLY (mask-weighted) mean. Shared by every source so
    the aggregation cannot drift between them; see the module docstring for why both steps matter.
    Downsampling needs OpenCV; without it this raises ImportError (a frame already at ``out_hw``
    needs no resize and works without it).
    """
    d = np.asarray(depth_m, dtype=np.float32)
    if d.ndim != 2:
        raise ValueError(f"depth must be (H, W); got {d.shape}")
    d = d.copy()                                    # never mutate the caller's / renderer's buffer
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    far = float(far_m)

    # VALID mask: finite and a real return. The D435 reports "no return" as 0, and its stated valid
    # range tops out around 65.54 m, so anything non-finite or <= 0 carries no depth information.
    valid = np.isfinite(d) & (d > 1e-6)
    # Clamp the FAR side of the VALID pixels before any aggregation: an out-of-range reading is real
    # data, just saturated, and letting 65.54 m into a mean wrecks the block (a 0.5 m wall reads
    # 1.14 m at the real 101:1 ratio). The NEAR side is left to the ObservationBuilder's clip_min, so
    # there is exactly one place that value lives.
    # Zero the INVALID pixels while clamping, rather than np.minimum in place: NaN survives
    # np.minimum (np.minimum(nan, 2.5) is nan) and would then leak into the weighted sum below as
    # nan * 0.0 == nan, poisoning the whole output pixel. (The C++ mirror skips invalid pixels.)
    d = np.where(valid, np.minimum(d, far), 0.0).astype(np.float32)

    if (d.shape[0], d.shape[1]) == (out_h, out_w):
        d[~valid] = far                             # no aggregation: invalid simply reads far
        return np.ascontiguousarray(d, dtype=np.float32)

    # DOWNSAMPLE WITH A VALID-ONLY (mask-weighted) MEAN (see the module docstring). Training rendered
    # 84x48 natively and its pixel dropout set WHOLE output pixels to far, rather than mixing a far
    # fraction into every one. Averaging only the measured pixels removes the hole-density bias,
    # invents no geometry (unlike hole filling) and adds no lag (unlike a temporal filter).
    # Only a block with NO valid pixel reports far.
    vf = valid.astype(np.float32)
    cv2 = _require_cv2()
    num = cv2.resize(d * vf, (out_w, out_h), interpolation=cv2.INTER_AREA)
    den = cv2.resize(vf, (out_w, out_h), interpolation=cv2.INTER_AREA)
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), far).astype(np.float32)
    return np.ascontiguousarray(out, dtype=np.float32)


class _BaseDepthSource:
    """Common far-plane fallback and frame count."""

    def __init__(self, out_hw: tuple[int, int], far_m: float):
        # far_m fills every invalid/no-return pixel, so far_m=0 renders "obstacle pressed against
        # the lens" across the frame (see far_plane()), and a negative value is not a distance.
        if far_m is None:
            raise ValueError("far_m is required and must equal the clip far your checkpoint trained "
                             "with (2.5 m for the HALO depth student)")
        far_m = float(far_m)
        if not np.isfinite(far_m) or far_m <= 0.0:
            raise ValueError(f"far_m must be a positive distance in meters, got {far_m}")
        self.out_hw = (int(out_hw[0]), int(out_hw[1]))
        self.far_m = far_m
        self._far = np.full(self.out_hw, self.far_m, dtype=np.float32)
        self._frames = 0

    def far_plane(self) -> np.ndarray:
        """Saturated far frame — what a source returns before its first real frame.

        Deliberately NOT zeros: 0 m would read as an obstacle pressed against the lens, which is the
        most dangerous possible hallucination. Far = 'nothing in view', matching the sensor's own
        no-return convention and training's inf -> far rule."""
        return self._far

    def is_fresh(self) -> bool:
        """True once the source has produced a frame (synchronous sources never go stale)."""
        return self._frames > 0


class MuJoCoDepthSource(_BaseDepthSource):
    """Renders depth from a camera in the live MuJoCo model — for `deploy_mujoco.py` eval.

    Uses MuJoCo's own depth buffer (already metric, already linearised), so there is no
    near/far un-projection to get wrong. The renderer is created once and reused.

    ``camera`` must be the name of a camera in the loaded XML whose pose/fovy matches the training
    TiledCamera; otherwise the policy sees a geometrically different scene and the comparison to sim
    is meaningless.
    """

    def __init__(self, model, data, camera: str, out_hw: tuple[int, int], far_m: float):
        super().__init__(out_hw, far_m)
        import mujoco
        self._mj = mujoco
        self.model, self.data, self.camera = model, data, camera
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if self.cam_id < 0:
            names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                     for i in range(model.ncam)]
            raise ValueError(f"camera {camera!r} not in the model; have {names}")
        # Render at the obs size: the training camera rendered 84x48 natively, with no downsample.
        # LAZY, AND PINNED TO THE FIRST THREAD THAT RENDERS. A mujoco.Renderer holds an OpenGL context,
        # and on macOS that context is thread-affine: constructing it here (usually the main thread) and
        # then calling render() from a control thread BLOCKS FOREVER — no exception, no output, the
        # control loop simply never completes its first tick, which is indistinguishable from a policy
        # that does nothing. Creating it on first get_depth() puts it on the thread that actually
        # renders; see _ensure_renderer for why migration is refused rather than retried.
        self._renderer = None
        self._owner_thread = None

    def get_depth(self) -> np.ndarray:
        self._ensure_renderer()
        self._renderer.update_scene(self.data, camera=self.cam_id)
        raw = self._renderer.render()               # (h, w) float32 METERS, MuJoCo depth buffer
        self._frames += 1
        return sanitize_depth(raw, self.out_hw, self.far_m)

    def _ensure_renderer(self) -> None:
        """Create the GL renderer on the first calling thread, and PIN it there.

        A `mujoco.Renderer` owns an OpenGL context. On macOS that context is thread-affine, and it
        cannot be migrated: measured behaviour is that using it from a second thread HANGS with no
        exception and no output (the control loop simply never finishes its first tick, which looks
        exactly like a policy that does nothing). Closing and recreating on the new thread appears to
        work once and then hangs on the thread after that — so recreation is not a fix either.

        Hence: pin, and RAISE if anyone else calls. A loud error naming the fix beats a silent hang.
        The rule for callers is one renderer per thread, created by the thread that will use it:
          * the depth source is owned by whichever thread runs the control loop;
          * a scene/viz renderer must be a SEPARATE object owned by the thread that serves frames.
        """
        me = threading.get_ident()
        if self._renderer is None:
            h, w = self.out_hw
            # No multisampling, as in the C++ backend: some Linux / EGL drivers resolve an MSAA depth buffer by
            # averaging samples, which blends foreground and background depth along every silhouette edge.
            self.model.vis.quality.offsamples = 0
            self._renderer = self._mj.Renderer(self.model, height=h, width=w)
            self._renderer.enable_depth_rendering()
            self._owner_thread = me
            return
        if self._owner_thread != me:
            raise RuntimeError(
                f"MuJoCoDepthSource.get_depth() called from thread {me} but its GL context is owned "
                f"by thread {self._owner_thread}. A mujoco.Renderer context cannot move between "
                f"threads — doing so hangs silently on macOS. Render depth only from the thread that "
                f"runs the control loop, and give any viz renderer its own separate object.")

    def close(self) -> None:
        """Release the GL renderer (G1Controller.cleanup calls this on the control thread).

        From any other thread it does nothing: the context is thread-affine (see _ensure_renderer).
        """
        if self._renderer is None or self._owner_thread != threading.get_ident():
            return
        try:
            self._renderer.close()
        except Exception:
            pass
        self._renderer = None


class RealSenseDepthSource(_BaseDepthSource):
    """Reads a RealSense D4xx DIRECTLY in this process — no server, no socket.

    A background thread owns the pipeline and publishes the newest sanitized frame under a lock; the
    control loop's `get_depth()` is a bounded read of that one array. The camera runs at its own rate
    (30 Hz typical) independent of the 50 Hz control loop, so a slow or stalled camera can never
    block control — it degrades to the last frame and then, past `stale_s`, to the far plane.

    STALENESS -> FAR, not last-frame-forever. A frozen depth image is worse than no image: the policy
    confidently steers around geometry that is no longer there. Past `stale_s` we return the far
    plane, and `is_fresh()` reports False.
    """

    FPS = 30
    NATIVE_HW = (480, 848)

    def __init__(self, out_hw: tuple[int, int], far_m: float, stale_s: float = 0.5):
        super().__init__(out_hw, far_m)
        if self.NATIVE_HW != self.out_hw:
            _require_cv2()                # fail at start-up, not silently inside the capture thread
        self.stale_s = float(stale_s)
        self._latest: Optional[np.ndarray] = None
        self._last_t = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pipe = None
        self._scale = 1.0e-3          # depth_scale, read from the device at start()

    def start(self, wait_timeout: float = 5.0) -> tuple[bool, dict]:
        import pyrealsense2 as rs
        cfg = rs.config()
        h, w = self.NATIVE_HW
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, self.FPS)
        self._pipe = rs.pipeline()
        profile = self._pipe.start(cfg)
        self._scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        self._thread = threading.Thread(target=self._loop, name="realsense-depth", daemon=True)
        self._thread.start()
        t0 = time.time()
        while time.time() - t0 < wait_timeout:
            with self._lock:
                if self._latest is not None:
                    return True, {"depth_scale": self._scale, "native_hw": self.NATIVE_HW,
                                  "fps": self.FPS, "out_hw": self.out_hw, "far_m": self.far_m}
            time.sleep(0.02)
        return False, {"error": f"no depth frame within {wait_timeout}s"}

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                frames = self._pipe.wait_for_frames(timeout_ms=1000)
                df = frames.get_depth_frame()
                if not df:
                    continue
                # Z16 -> meters via the device's own scale. asanyarray gives a view into the SDK
                # buffer, so sanitize_depth's copy is what keeps us safe after the frame is released.
                raw = np.asanyarray(df.get_data()).astype(np.float32) * self._scale
                frame = sanitize_depth(raw, self.out_hw, self.far_m)
                with self._lock:
                    self._latest = frame
                    self._frames += 1
                    self._last_t = time.time()
            except Exception:
                time.sleep(0.01)

    def get_depth(self) -> np.ndarray:
        with self._lock:
            frame, last_t = self._latest, self._last_t
        if frame is None or (time.time() - last_t) > self.stale_s:
            return self.far_plane()
        return frame

    def is_fresh(self) -> bool:
        """True while the newest frame is younger than ``stale_s`` (get_depth returns it, not the far plane)."""
        with self._lock:
            return self._latest is not None and (time.time() - self._last_t) <= self.stale_s

    def close(self) -> None:
        """Stop the capture thread and the pipeline. Safe to call more than once."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._pipe is not None:
            try:
                self._pipe.stop()
            except Exception:
                pass
            self._pipe = None

