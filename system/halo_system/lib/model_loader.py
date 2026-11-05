#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Motion-generator model loading (generator glue) for the deploy nodes.

The other shared generator-side helpers live alongside this module in
``halo_system.lib`` (inference: motion_plan_to_qpos / generate_motion_plan /
tracker_qpos_to_model_chunk; camera: OAKCameraStream). This file holds the one
piece that depends on the generator package.

The generator is NOT necessarily installed as a package in the deployment env,
so the repo root (which holds ``generator/``) is added to sys.path here. This
file lives at ``<repo>/system/halo_system/lib/model_loader.py`` ->
``parents[3]`` == the repo root.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

# --- generator bootstrap (must precede the model-package imports) ----------
# Prefer an ALREADY-importable generator (e.g. an editable install) so every
# node runs the same model code. Only fall back to the in-repo checkout when
# the generator isn't installed in the env.
_HALO_ROOT = Path(__file__).resolve().parents[3]          # repo root
if importlib.util.find_spec("generator") is None and str(_HALO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HALO_ROOT))

# Checkpoint tensors that belong to auxiliary heads the deploy path never runs (a
# task-progress head trained alongside the motion model). Tolerated -- and listed -- on
# load; ANY OTHER mismatch means the config does not describe the trained model and the
# load is refused.
_AUX_PREFIXES = ("progress_task_head.", "progress_head.", "_normalizer.")


def _load_model_strict(checkpoint: str, config, device):
    """Build with model_config_from (every model field copied by name, live VLM) and refuse any
    checkpoint/config mismatch other than frozen-VLM weights and known auxiliary heads."""
    import torch
    import generator
    # The model resolves its normalizer with a top-level `from data.transforms import ...`;
    # put the package folder on sys.path for that. Appended, not prepended, so nothing
    # already importable is shadowed.
    _pkg = os.path.dirname(generator.__file__)
    if _pkg not in sys.path:
        sys.path.append(_pkg)
    from generator.models.motion_generator import MotionGenerator, model_config_from
    t0 = time.perf_counter()
    model = MotionGenerator(model_config_from(config))
    t1 = time.perf_counter()
    # Memory-mapped on the CPU, copied once into the (CPU) model, then ONE move to the
    # device. map_location=device would put a second full copy on the GPU first and briefly
    # hold ~3 copies of a large fine-tuned VLM.
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    t2 = time.perf_counter()
    sd = {k.removeprefix("module."): v for k, v in ck["model_state_dict"].items()}
    res = model.load_state_dict(sd, strict=False)
    del sd
    ck["model_state_dict"] = None              # drop the mapped tensors before the device copy
    t3 = time.perf_counter()
    unexpected = [k for k in res.unexpected_keys if not k.startswith(_AUX_PREFIXES)]
    missing = [k for k in res.missing_keys if not k.startswith(("vlm_encoder.",) + _AUX_PREFIXES)]
    if unexpected or missing:
        raise RuntimeError(
            f"{Path(checkpoint).name} does not match {getattr(config, '_path', 'its config')}:\n"
            f"  in checkpoint but not in model: {unexpected[:10]}\n"
            f"  in model but not in checkpoint: {missing[:10]}\n"
            "Use the config of the run that produced this checkpoint.")
    aux = sorted({k.split(".")[0] for k in res.unexpected_keys if k.startswith(_AUX_PREFIXES[:2])})
    model.to(device)
    model.eval()
    print(f"[timing] model load: build {t1 - t0:.1f}s (incl. pretrained VLM), checkpoint read "
          f"{t2 - t1:.1f}s, load_state_dict {t3 - t2:.1f}s, to({device}) "
          f"{time.perf_counter() - t3:.1f}s", flush=True)
    m = config.model
    print(f"Loaded {Path(checkpoint).name}: epoch {ck.get('epoch', '?')} "
          f"(best_val_loss={ck.get('best_val_loss', 'N/A')}), "
          f"cond_pos_embed={getattr(m, 'cond_pos_embed', False)}"
          + (f", ignored aux heads {aux}" if aux else ""), flush=True)
    return model


def load_generator_model(config_path: str, checkpoint: str, device: str):
    """Load the motion-generator model + image transform from a config + ckpt.

    Imports the generator lazily (heavy: torch + the model package) so this
    module stays importable without them.

    Returns ``(motion_model, image_transform, config)``.
    """
    from halo_system.lib.assets import resolve_asset
    config_path = resolve_asset(str(config_path), required="generator config")
    checkpoint = resolve_asset(str(checkpoint), required="generator checkpoint")

    from generator.configs.config import load_config
    from generator.data.transforms import get_vlm_image_transform

    config = load_config(config_path)
    motion_model = _load_model_strict(checkpoint, config, device)
    # Optional: serve this model through halo_serve (one GPU-owning thread, MLX DiT,
    # measured deadlines). Unset = unchanged behaviour; any failure falls back to torch.
    if os.environ.get("HALO_SERVE", "0") == "1":
        try:
            from halo_serve.deploy import attach
            # Lets the server cache a fine-tuned VLM's converted MLX weights per checkpoint
            # file (conversion runs once, later starts reuse it).
            motion_model._halo_serve_checkpoint = str(checkpoint)
            _t = time.perf_counter()
            attach(motion_model, name=Path(checkpoint).stem)
            print(f"[timing] halo_serve attach (MLX DiT + VLM): {time.perf_counter() - _t:.1f}s",
                  flush=True)
        except Exception as e:  # never break the robot over an accelerator
            print(f"[halo_serve] attach failed, running torch: {type(e).__name__}: {e}",
                  flush=True)
    image_transform = get_vlm_image_transform(tuple(config.data.image_size))
    return motion_model, image_transform, config
