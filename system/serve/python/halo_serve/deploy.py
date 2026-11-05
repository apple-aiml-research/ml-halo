#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Deploy entry point: put an already-loaded generator model under the serving engine.

    from halo_serve.deploy import attach
    attach(motion_model)            # returns the backend actually used: "mlx" | "torch"

Registers the model with `global_engine()` and redirects `model.predict` so every call is a job
on the one GPU thread (same signature and return value). Env knobs:
  HALO_SERVE_STATS           JSON path; engine stats written every 20 jobs and at exit
"""
from __future__ import annotations

import atexit
import itertools
import os

from halo_serve.api.registry import create
from halo_serve.runtime.engine import global_engine

_ids = itertools.count()


def attach(model, name: str = "motion_gen") -> str:
    if getattr(model, "_halo_serve_attached", None):
        return model._halo_serve_attached
    eng = global_engine()
    wname = f"{name}:{next(_ids)}"                # count.__next__ is atomic under the GIL
    wl = create("halo_generator", model=model)
    # Without a deadline EDF would rank a co-served orchestrator call (~45 s) ahead of every plan.
    eng.register(wname, wl, deadline_ms=333.0)
    wl._predict = model.predict                 # the accelerated predict; the job runs this
    stats_path = os.environ.get("HALO_SERVE_STATS")
    n = itertools.count(1)

    def predict(rgb, prompt, prev_chunk=None):
        req = {"rgb": rgb, "prompt": prompt, "prev_chunk": prev_chunk}
        out = eng.submit(wname, req).result().output
        if stats_path and next(n) % 20 == 0:
            try:                                # telemetry must never fail a motion plan
                eng.dump(stats_path)
            except Exception as e:              # noqa: BLE001
                print(f"[halo_serve] stats dump failed: {type(e).__name__}: {e}", flush=True)
        return out

    model.predict = predict
    model._halo_serve_attached = wl.backend
    if stats_path:
        atexit.register(lambda: eng.dump(stats_path))
    eng.meta[wname] = {"dit": wl.backend,
                       "encoder": getattr(model, "_halo_serve_vlm_backend", "torch"),
                       "steps": int(model.config.num_inference_steps)}
    # Serve this engine to the orchestrator process when an orchestrator tier uses the local model.
    ep = os.environ.get("HALO_SERVE_ENDPOINT")
    if ep:
        from halo_serve.runtime.service import start_service
        import halo_serve.orchestrator  # noqa: F401  registers the "chat" op
        start_service(eng, ep)
    print(f"[halo_serve] attached {wname}: dit={wl.backend} "
          f"encoder={eng.meta[wname]['encoder']} steps={eng.meta[wname]['steps']} "
          f"gpu-thread={eng.gpu_tid}", flush=True)
    return wl.backend


def preload_orchestrator() -> None:
    """Load + warm HALO_SERVE_PRELOAD_ORCHESTRATOR (if set) in the calling thread."""
    pre = os.environ.get("HALO_SERVE_PRELOAD_ORCHESTRATOR")
    if not pre:
        return
    import time
    import halo_serve.orchestrator as _orchestrator
    t0 = time.time()
    _orchestrator._workload_for(pre)
    print(f"[halo_serve] orchestrator {pre} preloaded in {time.time() - t0:.1f}s "
          f"(after the generator warm-up)", flush=True)
