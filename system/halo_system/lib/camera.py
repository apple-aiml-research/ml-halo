#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""OAK camera capture for the generator node (real robot)."""
from __future__ import annotations

import threading
import time


class OAKCameraStream:
    """Luxonis OAK (DepthAI) camera capture.

    OAK devices are Intel Myriad-X VPUs (e.g. OAK-1, USB id ``03e7:2485``), NOT
    UVC webcams, so we pull frames over the DepthAI v3 pipeline instead. A
    background thread fills ``get_frame()`` with RGB ``uint8`` frames,
    center-cropped to the output aspect and resized to
    ``out_width x out_height`` (default 1280x800 = 16:10, the same frame the
    sim's oak1 render produces).

    ``oak_camera_index`` selects which on-device camera socket to read, indexing
    into ``device.getConnectedCameras()`` sorted by name (0 -> CAM_A on OAK-1).
    The OAK emits its full-resolution native mode, which we then crop + resize.
    """

    # Sensor the policy was trained for: OAK-1 W = OV9782 (1280x800 global shutter, wide FOV).
    # The plain OAK-1 has an IMX378 (12 MP, narrow FOV) and looks nothing like the training images.
    EXPECTED_SENSOR = "OV9782"

    def __init__(self, oak_camera_index=0, out_width=1280, out_height=800, warmup_frames=10):
        self.oak_camera_index = int(oak_camera_index)
        self.out_width = out_width
        self.out_height = out_height
        # OAK auto-exposure ramps over the first several frames; drop them so
        # the main loop never sees an underexposed observation.
        self.warmup_frames = max(0, int(warmup_frames))
        self._frame = None
        self._frame_time = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        # Set on first frame OR on a startup error, so start() can report
        # open failures synchronously.
        self._started = threading.Event()
        self._error = None

    def _check_sensor(self, device, socket):
        """Warn loudly when the sensor on `socket` is not EXPECTED_SENSOR (wrong OAK plugged in)."""
        try:
            feats = {str(f.socket): f for f in device.getConnectedCameraFeatures()}
        except Exception as exc:  # noqa: BLE001 -- no feature API: skip the check
            print(f"[WARN] OAK camera {self.oak_camera_index}: could not read the sensor name "
                  f"({type(exc).__name__}: {exc}); sensor check skipped", flush=True)
            return
        f = feats.get(str(socket))
        if f is None:
            return
        name = str(f.sensorName)
        if name.upper() == self.EXPECTED_SENSOR:
            print(f"[INFO] OAK camera {self.oak_camera_index} ({socket}): sensor {name} "
                  f"{f.width}x{f.height}", flush=True)
            return
        msg = (f"OAK camera {self.oak_camera_index} ({socket}) has sensor {name} {f.width}x{f.height}, "
               f"but the policy expects {self.EXPECTED_SENSOR} (OAK-1 W). Wrong camera plugged in?")
        print("[WARN] " + "!" * 70 + "\n[WARN] " + msg + "\n[WARN] " + "!" * 70, flush=True)

    def start(self, ready_timeout=15.0):
        """Open the OAK device in a background thread.

        The DepthAI device can only be held by one handle, so we open it
        inside the read thread. We then block up to
        ``ready_timeout`` for either the first frame or a startup error, and
        re-raise the error here.
        """
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        self._started.wait(timeout=ready_timeout)
        if self._error is not None:
            self._running = False
            raise RuntimeError(f"Cannot open OAK camera "
                               f"{self.oak_camera_index}: {self._error}")

    @staticmethod
    def _center_crop_to_aspect(frame, target):
        """Center-crop ``frame`` to ``target`` aspect ratio. No-op if already matches."""
        h, w = frame.shape[:2]
        if h <= 0:
            return frame
        cur = w / h
        if abs(cur - target) < 0.01:
            return frame
        if cur > target:
            new_w = int(round(h * target))
            x0 = (w - new_w) // 2
            return frame[:, x0:x0 + new_w]
        new_h = int(round(w / target))
        y0 = (h - new_h) // 2
        return frame[y0:y0 + new_h, :]

    def _read_loop(self):
        import cv2
        import numpy as np
        try:
            import depthai as dai
        except ImportError as exc:
            self._error = (f"depthai not installed ({exc}). OAK cameras need "
                           f"the DepthAI SDK: `pip install depthai`.")
            self._started.set()
            return
        try:
            device = dai.Device()
            with dai.Pipeline(device) as pipeline:
                sockets = sorted(device.getConnectedCameras(), key=str)
                if not sockets:
                    raise RuntimeError("OAK device reports no connected cameras")
                if self.oak_camera_index >= len(sockets):
                    raise RuntimeError(
                        f"oak_camera_index={self.oak_camera_index} out of range; "
                        f"device has {len(sockets)} camera(s): "
                        f"{[str(s) for s in sockets]}")
                socket = sockets[self.oak_camera_index]
                self._check_sensor(device, socket)
                cam = pipeline.create(dai.node.Camera).build(socket)
                # FRESHEST FRAME ONLY. DepthAI's default output queue is a FIFO of 16 that
                # drops its OLDEST on overflow (createOutputQueue(maxSize=16, blocking=False)),
                # and queue.get() returns the OLDEST entry. So if this thread ever falls behind
                # the camera -- it shares the GIL with inference, the g1/obs JPEG encode and
                # state polling -- the backlog never clears and the generator keeps seeing an
                # image up to 16 frames (~0.5 s @30 fps) old. maxSize=1 keeps one frame on the
                # device side, and the drain below discards anything older on this side.
                # SCALE ON THE CAMERA, not on the host. requestFullResolutionOutput() streams the
                # sensor's full mode and we crop+resized here: measured on the robot, frames were
                # ~275 ms old (p50) at ~25 fps with ZERO stale frames on this side -- the backlog
                # was inside the device/USB, which could not move full-size frames at 30 fps.
                # requestOutput(size) makes the ISP do the same centre crop (resizeMode CROP is
                # the default) and scale, so only out_width x out_height crosses the link.
                _out = None
                if self.out_width > 0 and self.out_height > 0:
                    # FRAME RATE FROM THE USB LINK. Measured on a USB 2.0 link (usb=HIGH): 1280x800
                    # @30 fps exceeds what the link carries -> 25 fps and frames ~380 ms old;
                    # @15 fps -> 15 fps and ~63 ms. On USB 3 (SUPER) 30 fps is fine. The generator
                    # plans every ~0.15-0.25 s, so 15 fps loses nothing.
                    try:
                        _usb = str(device.getUsbSpeed().name)
                    except Exception:  # noqa: BLE001
                        _usb = "?"
                    _fps = 30.0 if _usb in ("SUPER", "SUPER_PLUS") else 15.0
                    if _usb not in ("SUPER", "SUPER_PLUS"):
                        print(f"[WARN] OAK on a USB {_usb} link (not USB 3): streaming "
                              f"{_fps:g} fps so frames stay fresh -- use a USB 3 port/cable for 30",
                              flush=True)
                    try:
                        _out = cam.requestOutput((int(self.out_width), int(self.out_height)),
                                                 fps=_fps)
                        _mode = f"ISP-scaled {self.out_width}x{self.out_height} @{_fps:g}fps (usb {_usb})"
                    except Exception as _exc:  # noqa: BLE001 -- SDK without that signature
                        print(f"[WARN] OAK requestOutput({self.out_width}x{self.out_height}) "
                              f"failed ({type(_exc).__name__}: {_exc}); using full resolution")
                        _out = None
                if _out is None:
                    _out = cam.requestFullResolutionOutput()
                    _mode = "full-res"
                queue = _out.createOutputQueue(maxSize=1, blocking=False)
                pipeline.start()
                print(f"[INFO] OAK camera {self.oak_camera_index} ({socket}): "
                      f"streaming {_mode}"
                      + (f" -> {self.out_width}x{self.out_height}"
                         if self.out_width > 0 else ""))
                # Device opened and pipeline running — unblock start() now.
                # get_frame() stays None until warmup completes below.
                self._started.set()
                ok_count = 0
                first_ok_logged = False
                _lat, _drop, _t_rep = [], 0, time.time()
                while self._running and pipeline.isRunning():
                    pkt = queue.get()
                    if pkt is None:
                        continue
                    # Drain: if more frames arrived meanwhile, keep only the newest.
                    for _newer in (queue.tryGetAll() or []):
                        if _newer is not None:
                            pkt = _newer
                            _drop += 1
                    frame = pkt.getCvFrame()  # BGR uint8, like cv2.read()
                    if frame is None:
                        continue
                    # True frame age: capture time (host-synced) vs now, same DepthAI clock.
                    try:
                        _lat.append((dai.Clock.now() - pkt.getTimestamp()).total_seconds())
                    except Exception:  # noqa: BLE001 -- diagnostics must never stop capture
                        pass
                    ok_count += 1
                    if ok_count <= self.warmup_frames:
                        # Device is open and delivering — report ready even
                        # while we burn warmup frames.
                        self._started.set()
                        continue
                    if not first_ok_logged:
                        h, w = frame.shape[:2]
                        print(f"[INFO] OAK camera {self.oak_camera_index}: first "
                              f"usable frame ({w}x{h}) after {ok_count} reads "
                              f"({self.warmup_frames}-frame warmup); reading…")
                        first_ok_logged = True
                    # Crop to the OUTPUT aspect (16:10 for the 1280x800 default = the sim's oak
                    # render), not a fixed 16:9: the OAK fisheye is 16:10 natively, so a 16:9
                    # crop threw away ~10% of its vertical FOV.
                    _aspect = (self.out_width / self.out_height
                               if self.out_width > 0 and self.out_height > 0 else 16 / 10)
                    frame = self._center_crop_to_aspect(frame, _aspect)
                    if (self.out_width > 0 and self.out_height > 0
                            and (frame.shape[1], frame.shape[0]) != (self.out_width, self.out_height)):
                        # INTER_AREA only for downscaling; the OAK's native mode may be SMALLER
                        # than the requested output, and upscaling wants INTER_LINEAR. Skipped
                        # entirely when the native mode already IS the output (OV9782 1280x800).
                        _down = frame.shape[1] >= self.out_width
                        frame = cv2.resize(frame, (self.out_width, self.out_height),
                                           interpolation=cv2.INTER_AREA if _down
                                           else cv2.INTER_LINEAR)
                    with self._lock:
                        self._frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self._frame_time = time.time()
                    # Every 10 s: how old the frames handed to the generator really are, and how
                    # many stale ones the drain threw away (non-zero = this thread fell behind).
                    if time.time() - _t_rep >= 10.0 and _lat:
                        _a = np.asarray(_lat) * 1e3
                        print(f"[cam] frame age capture->ready p50 {np.median(_a):.0f} ms  "
                              f"p95 {np.percentile(_a, 95):.0f} ms  max {_a.max():.0f} ms  "
                              f"| {len(_a)} frames, {_drop} stale dropped", flush=True)
                        _lat, _drop, _t_rep = [], 0, time.time()
                    self._started.set()
        except Exception as exc:  # noqa: BLE001 — surface via start()/logs
            if self._error is None:
                self._error = str(exc)
            print(f"[WARN] OAK camera {self.oak_camera_index}: read loop "
                  f"stopped: {exc}")
        finally:
            self._running = False
            self._started.set()

    def get_frame(self, max_age=1.0):
        with self._lock:
            if self._frame is None:
                return None
            if (time.time() - self._frame_time) > max_age:
                return None
            return self._frame.copy()

    def stop(self):
        self._running = False
        # The read thread owns the DepthAI device via a context manager; joining
        # lets it exit its blocking queue.get() (next frame ~33ms) and close the
        # device cleanly before we return.
        if self._thread is not None:
            self._thread.join(timeout=2.0)
