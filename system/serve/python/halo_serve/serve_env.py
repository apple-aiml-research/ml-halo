#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Serving switch shared by run_sim.py and run_real.py: env overlay for the generator process.

Config -- the task YAML (`halo_serve: true`); absent = off:

    halo_serve: true

Stats go to `<run_dir>/halo_serve_stats.json`. This module must import without MLX installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

def serve_env(mission_raw, run_dir) -> Dict[str, str]:
    """Env overlay for the generator process from the task YAML `halo_serve:` (absent = off)."""
    if mission_raw is None:
        return {"HALO_SERVE": "0"}
    if not isinstance(mission_raw, bool):
        raise ValueError(f"mission `halo_serve:` must be a bool, got {type(mission_raw).__name__}")
    on = bool(mission_raw)
    if not on:
        return {"HALO_SERVE": "0"}
    try:
        import halo_serve.deploy  # noqa: F401 -- fail here rather than in the child
        import mlx.core, mlx_vlm  # noqa: F401,E401 -- the fast path runs on MLX
    except Exception as e:                                        # noqa: BLE001
        print(f"[halo_serve] requested but not importable ({type(e).__name__}: {e}) -- "
              f"generator runs stock torch.", flush=True)
        return {"HALO_SERVE": "0"}
    env = {"HALO_SERVE": "1",
           "HALO_SERVE_STATS": str(Path(run_dir).expanduser().resolve() / "halo_serve_stats.json")}
    print("[halo_serve] ON for the generator: " + " ".join(f"{k}={v}" for k, v in env.items()),
          flush=True)
    return env


# The orchestrator runs its local model on the generator's engine (one GPU owner) via a shared endpoint.

def service_env(orchestrator_cmd, run_dir) -> Dict[str, str]:
    """Env for both the generator (serves the engine) and the orchestrator (connects to it)."""
    from halo_serve.orchestrator import DEFAULT_MODEL
    from halo_serve.runtime.service import default_endpoint
    try:
        import mlx.core, mlx_vlm  # noqa: F401,E401
    except ImportError as e:
        raise SystemExit(f"[halo_serve] the orchestrator's local model needs MLX ({e}); on Apple Silicon "
                         f"run: pip install -e 'system/serve/python[mlx]'") from e
    ep = default_endpoint(Path(run_dir).expanduser().resolve())
    model = (orchestrator_cmd[orchestrator_cmd.index("--model") + 1] if "--model" in orchestrator_cmd
             else DEFAULT_MODEL)
    env = {"HALO_SERVE": "1", "HALO_SERVE_ENDPOINT": ep, "HALO_SERVE_PRELOAD_ORCHESTRATOR": model}
    print(f"[halo_serve] orchestrator co-served through {ep} "
          f"(orchestrator stays its own process; model {model})", flush=True)
    return env
