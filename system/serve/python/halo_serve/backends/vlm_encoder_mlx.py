#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""MLX (mlx-vlm) backend for the VLM feature extractor.

Re-runs the torch ``VLMEncoder`` forward through mlx-vlm's Metal-native Qwen3.5 (~2.5x faster at
batch 1; the forward is dispatch bound on MPS). The torch encoder's processor is reused, so
preprocessing and tokenisation are identical; only the model compute differs.

    forward(images [B,3,H,W] in [0,1], texts: List[str])
        -> (image_features [B, gh*gw, D] fp32 = every image token,
            text_features [B, T, D] fp32 = every token after <|vision_end|>,
            text_mask [B, T] all False)

The vision tower is compiled once per image grid (fused bf16, slight numeric shift) and the
M-RoPE position ids are cached per (prompt, grid) token layout (exact). bf16 kernels differ from
torch-MPS, so features are not bit-identical (cosine ~0.998). Quantized checkpoints are rejected.

Threading: mlx-vlm GPU state is bound to the thread that creates it, and predict() is called from
per-generation threads, so construction AND every forward run on one private executor thread.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Tuple

import numpy as np
import torch
import mlx.core as mx
from mlx_vlm import load as _mlx_vlm_load
from torchvision.transforms.functional import to_pil_image

logger = logging.getLogger(__name__)


class MLXVLMEncoder:
    """mlx-vlm re-implementation of ``VLMEncoder``'s feature-extraction forward."""

    def __init__(self, torch_encoder) -> None:
        if not mx.metal.is_available():
            raise RuntimeError("MLX Metal is unavailable.")
        self._mx_dtype = {torch.bfloat16: mx.bfloat16, torch.float16: mx.float16,
                          torch.float32: mx.float32}
        self.processor = torch_encoder.processor
        self.dtype: torch.dtype = torch_encoder.dtype
        tok = self.processor.tokenizer
        self._vision_start_id = tok.convert_tokens_to_ids("<|vision_start|>")
        self._vision_end_id = tok.convert_tokens_to_ids("<|vision_end|>")
        if self._vision_start_id is None or self._vision_end_id is None:
            raise ValueError("tokenizer is missing <|vision_start|>/<|vision_end|> tokens")
        if self.dtype not in self._mx_dtype:
            raise ValueError(f"unsupported VLM dtype {self.dtype}")

        from huggingface_hub import snapshot_download
        path = snapshot_download(torch_encoder.model_id, local_files_only=True)
        # Chat-template strings depend only on the prompt text (the image is a
        # placeholder), so for a fixed prompt they are constant — cache per text.
        self._chat_cache: dict = {}
        self._vis_graphs: dict = {}
        self._rope_cache: dict = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-vlm")
        self._executor.submit(self._build, path).result()

    # ------------------------------------------------------------------ #
    def _build(self, path: str) -> None:
        """Load the mlx-vlm model ON the executor thread; reject quantized weights."""
        self._model, _ = _mlx_vlm_load(path)
        emb = self._model.language_model.model.embed_tokens
        if hasattr(emb, "bits"):  # quantized Embedding/Linear expose `.bits`
            raise RuntimeError(
                "mlx-vlm loaded a QUANTIZED model; this path requires full-rank weights.")
        if emb.weight.dtype not in (mx.bfloat16, mx.float16, mx.float32):
            raise RuntimeError(f"unexpected VLM weight dtype {emb.weight.dtype}")
        logger.info("mlx-vlm VLM backend loaded (dtype=%s).", emb.weight.dtype)

    def _run(self, fn: Callable, *args):
        return self._executor.submit(fn, *args).result()

    # ------------------------------------------------------------------ #
    def forward(self, images: torch.Tensor, texts: List[str]
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """See module docstring. Dispatches onto the private MLX thread."""
        return self._run(self._forward_impl, images, texts)

    def _vision(self, pv, grid_key):
        g = self._vis_graphs.get(grid_key)
        if g is None:
            m = self._model
            grid = mx.array([list(grid_key)], dtype=mx.int32)
            g = mx.compile(lambda p: m.vision_tower(p, grid)[0])
            self._vis_graphs[grid_key] = g
        return g(pv)

    def _forward_impl(self, images: torch.Tensor, texts) -> tuple:
        if images.dim() != 4 or len(texts) != images.shape[0]:
            raise ValueError(f"images {tuple(images.shape)} vs {len(texts)} texts")
        m = self._model
        dtype = self._mx_dtype[self.dtype]
        img_f: List[np.ndarray] = []
        txt_f: List[np.ndarray] = []
        # B==1 is the deployment case; B>1 runs each sample independently.
        for b in range(images.shape[0]):
            inp, i0, i1 = self._prepare(images[b], texts[b])
            ids_np = inp["input_ids"].numpy().astype(np.int32)
            grid_np = inp["image_grid_thw"].numpy().astype(np.int32)
            ids = mx.array(ids_np)
            pv = mx.array(inp["pixel_values"].float().numpy()).astype(dtype)
            vis = self._vision(pv, tuple(grid_np.ravel().tolist()))
            emb = m.language_model.model.embed_tokens(ids)
            emb, _ = m.merge_input_ids_with_image_features(
                vis, emb, ids, m.config.image_token_index, m.config.video_token_index)
            key = (ids_np.tobytes(), grid_np.tobytes())
            pr = self._rope_cache.get(key)
            if pr is None:
                pr = m.language_model.get_rope_index(ids, mx.array(grid_np), None, None)
                mx.eval(*pr)
                self._rope_cache[key] = pr
            h = m.language_model(ids, inputs_embeds=emb, position_ids=pr[0], rope_deltas=pr[1],
                                 mask=None, return_hidden=True, skip_logits=True).hidden_states[-1]
            im = h[0, i0:i1].astype(mx.float32)                              # [gh*gw, D]
            tx = h[0, i1 + 1:].astype(mx.float32)                            # [T, D]
            mx.eval(im, tx)
            img_f.append(np.array(im, copy=True))
            txt_f.append(np.array(tx, copy=True))
        dev = images.device
        if len({t.shape[0] for t in txt_f}) != 1:
            raise ValueError("different text lengths in one batch")
        img = torch.from_numpy(np.stack(img_f)).to(dev)
        txt = torch.from_numpy(np.stack(txt_f)).to(dev)                      # [B, T, D]
        return img, txt, torch.zeros(txt.shape[:2], dtype=torch.bool, device=dev)

    def _prepare(self, image_chw: torch.Tensor, text: str):
        """Replicate VLMEncoder.forward preprocessing EXACTLY; return (inputs, i0, i1)
        where [i0:i1) is the image-token span in the token sequence."""
        pil = to_pil_image(image_chw.detach().to("cpu", torch.float32).clamp(0, 1))
        chat = self._chat_cache.get(text)
        if chat is None:
            chat = self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image", "image": pil},
                                              {"type": "text", "text": text}]}],
                tokenize=False, add_generation_prompt=False)
            self._chat_cache[text] = chat
        inp = self.processor(text=[chat], images=[pil], return_tensors="pt", padding=True)
        ids = inp["input_ids"][0].tolist()
        try:
            i0 = ids.index(self._vision_start_id) + 1
            i1 = ids.index(self._vision_end_id)
        except ValueError:
            raise ValueError("tokenized input has no <|vision_start|>..<|vision_end|> span")
        if i1 <= i0:
            raise ValueError(f"invalid image-token span [{i0}, {i1})")
        return inp, i0, i1
