#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Accelerate a loaded motion generator with the MLX DiT (and MLX Qwen encoder for VLM models).

    from halo_serve.integrations.halo_system import accelerate
    accelerate(model)            # returns "mlx" or "torch" (what it actually selected)

Only the inpaint path (DiT + prev-chunk prefix) is taken over; any other call goes to the
original `predict`. Falls back to torch if MLX is unavailable.
Equivalence contract: same torch-seeded noise, same conditioning, same tail (`squeeze(0)` only
for unbatched input, then `normalizer.denormalize`).
"""
from __future__ import annotations

import logging
import os
import types

import torch

log = logging.getLogger(__name__)


def _conditioning(model, rgb, language):
    """The conditioning block of `MotionGenerator.predict`, verbatim (every model is Qwen + DiT)."""
    enc = getattr(model, "_encode_vlm", None)                # set by accelerate()
    image_feat, text_feat, text_mask = (enc or model.vlm_encoder)(rgb, language)
    conditioning, mask = model._vlm_tokens(image_feat, text_feat, text_mask)
    if mask is not None and bool(mask.any()):
        # The accelerated DiT has no padding-mask input.
        raise RuntimeError("padded text tokens are not supported by the accelerated DiT")
    return model._encode_conditioning(conditioning)


class _TorchOutProcessor:
    """Force a HF processor's outputs back to torch tensors.

    After mlx-vlm has loaded one model, processors for later models ignore `return_tensors="pt"`.
    """

    def __init__(self, proc):
        self._p = proc

    def __call__(self, *a, **kw):
        import numpy as np
        out = self._p(*a, **kw)
        for k in list(out.keys()):
            v = out[k]
            if v is not None and not isinstance(v, torch.Tensor) and hasattr(v, "shape"):
                out[k] = torch.from_numpy(np.array(v))
        return out

    def __getattr__(self, name):
        return getattr(self._p, name)


def accelerate(model) -> str:
    """Install the MLX DiT on `model` (Apple silicon). Never raises: falls back to torch."""
    try:
        if getattr(model, "dit_decoder", None) is None:
            raise RuntimeError("not a DiT model")
        from halo_serve.backends.dit_decoder_mlx import MLXInpaintDiTDecoder
        mlx_dit = MLXInpaintDiTDecoder(model.dit_decoder)
    except Exception as e:                                        # noqa: BLE001
        log.warning("halo_serve: MLX DiT unavailable (%s); staying on torch", e)
        return "torch"

    # MLXVLMEncoder wraps the model's own torch VLMEncoder (same processor and tokenisation).
    vlm_backend = "torch"
    try:
        from halo_serve.backends.vlm_encoder_mlx import MLXVLMEncoder
        enc = MLXVLMEncoder(model.vlm_encoder)
        _load_finetuned_vlm(enc, model)
        enc.processor = _TorchOutProcessor(enc.processor)
        # The generator always conditions on every image token and every text token.
        vlm_backend = "mlx-fast-full-textall"
        model._encode_vlm = enc.forward          # _conditioning() prefers this
    except Exception as e:                                    # noqa: BLE001
        log.warning("halo_serve: MLX VLM encoder unavailable (%s); encoder stays torch", e)
    model._halo_serve_vlm_backend = vlm_backend

    orig_predict = model.predict

    def predict(self, rgb, language, prev_chunk=None):
        if not (self.inpaint_prev_chunk and prev_chunk is not None):
            return orig_predict(rgb, language, prev_chunk=prev_chunk)
        single_sample = rgb.dim() == 3
        if single_sample:
            rgb = rgb.unsqueeze(0)
        if isinstance(language, str):
            language = [language]
        B, P = rgb.shape[0], self.prev_chunk_frames
        prefix = prev_chunk.to(rgb.device)
        if prefix.dim() == 2:
            prefix = prefix.unsqueeze(0).expand(B, -1, -1)
        with torch.no_grad():
            conditioning = _conditioning(self, rgb, language)
            trajectory = torch.randn(B, self.config.trajectory_length, self.config.output_dim,
                                     device=rgb.device)            # same seeding as torch
            trajectory[:, :P] = prefix
            trajectory = mlx_dit.inpaint_euler(trajectory, conditioning,
                                               self.config.num_inference_steps, prefix[:, :P], P)
        if single_sample:
            trajectory = trajectory.squeeze(0)
        if self.normalizer is not None:
            trajectory = self.normalizer.denormalize(trajectory)
        return trajectory

    model.predict = types.MethodType(predict, model)
    log.info("halo_serve: MLX inpaint DiT installed (%d steps, P=%d)",
             model.config.num_inference_steps, model.prev_chunk_frames)
    return "mlx"


def _finetuned_vlm_cache_path(model):
    """Disk cache path (~/.cache/halo/mlx_vlm) for converted fine-tuned MLX VLM weights, keyed by
    checkpoint path/size/mtime."""
    ck = getattr(model, "_halo_serve_checkpoint", None)
    if not ck:
        return None
    try:
        import hashlib
        from pathlib import Path
        p = Path(ck).resolve()
        st = p.stat()
        key = hashlib.sha1(f"{p}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]
        return Path.home() / ".cache" / "halo" / "mlx_vlm" / f"{p.stem}.{key}.safetensors"
    except OSError:
        return None


def _load_finetuned_vlm(enc, model) -> None:
    """Copy a fine-tuned Qwen's weights into the MLX encoder (which loads stock HF weights).

    Raises (-> torch encoder) if the converted weights do not cover the MLX model."""
    if not model.vlm_encoder.vision_trainable:
        return
    import mlx.core as mx
    from mlx.utils import tree_flatten
    torch_sd = model.vlm_encoder.model.state_dict()

    cache = _finetuned_vlm_cache_path(model)

    def _copy():
        mm = enc._model
        have = dict(tree_flatten(mm.parameters()))
        hit = None
        if cache is not None and cache.exists():
            # Shapes are re-checked; any mismatch falls back to converting.
            try:
                got = mx.load(str(cache))
                if set(got) == set(have) and all(
                        tuple(got[k].shape) == tuple(have[k].shape) for k in have):
                    hit = got
                    print(f"[halo_serve] fine-tuned VLM: MLX weights from cache {cache.name}",
                          flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[halo_serve] fine-tuned VLM cache unreadable ({e}); converting", flush=True)
        if hit is None:
            w = {k: mx.array(v.detach().float().cpu().numpy()) for k, v in torch_sd.items()}
            if hasattr(mm, "sanitize"):
                w = mm.sanitize(w)
            hit = {k: v.astype(have[k].dtype) for k, v in w.items()
                   if k in have and tuple(v.shape) == tuple(have[k].shape)}
            if cache is not None and len(hit) == len(have):
                try:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    tmp = cache.with_name(cache.stem + ".tmp.safetensors")
                    mx.save_safetensors(str(tmp), hit)
                    os.replace(tmp, cache)
                    print(f"[halo_serve] fine-tuned VLM: MLX weights cached -> {cache}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[halo_serve] fine-tuned VLM cache not written ({e})", flush=True)
        mm.load_weights(list(hit.items()), strict=False)
        mx.eval(mm.parameters())
        return len(have), len(hit), [k for k in have if k not in hit][:5]
    n_have, n_hit, miss = enc._run(_copy)
    print(f"[halo_serve] fine-tuned VLM -> MLX: {n_hit}/{n_have} tensors from the checkpoint", flush=True)
    if n_hit < n_have:
        raise RuntimeError(f"fine-tuned VLM weights cover only {n_hit}/{n_have} MLX tensors "
                           f"(e.g. missing {miss}); keeping the torch encoder")
