#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Adapter: any mlx-vlm vision-language model as a reasoner.

    create("mlx_vlm", model_path="mlx-community/Qwen3.6-35B-A3B-4bit", max_tokens=96)

Request: {"messages": [OpenAI-style chat], "schema": dict|None, "max_tokens": int}. Greedy.
A schema makes decoding grammar-constrained (rebuilt per call: the matcher is stateful).
Decoding reuses the cached pre-image prompt prefix (`backends.prefix_cache`).
Yield points are the model's own layer classes, so the engine can preempt between layers.
"""
from __future__ import annotations

import contextlib
import os
import sys
from typing import Any

from halo_serve.api.registry import register
from halo_serve.api.workload import Result, Workload, discover_layer_classes


@contextlib.contextmanager
def _quiet_fds():
    # LLGuidance dumps per-token parser state from native code to fd 1/2.
    dn = os.open(os.devnull, os.O_WRONLY); s = (os.dup(1), os.dup(2))
    try:
        sys.stdout.flush(); sys.stderr.flush(); os.dup2(dn, 1); os.dup2(dn, 2); yield
    finally:
        os.dup2(s[0], 1); os.dup2(s[1], 2)
        for f in (*s, dn): os.close(f)


@register("mlx_vlm")
class MlxVlmReasoner(Workload):
    kind = "reasoner"

    def __init__(self, model_path: str, max_tokens: int = 256):
        self.model_path, self.max_tokens = model_path, max_tokens
        self.model = self.processor = self._apc = None

    def load(self) -> None:
        import mlx.core as mx
        from mlx_vlm import load
        from halo_serve.backends.prefix_cache import PrefixCache
        mx.set_wired_limit(int(24e9))
        mx.set_cache_limit(int(2e9))
        self.model, self.processor = load(self.model_path)
        mx.eval(self.model.parameters())
        # mlx-vlm's own APC needs a text-only suffix; this reuses the pre-image prefix.
        self._apc = PrefixCache(self.model, self.processor)

    def run(self, req: Any) -> Result:
        fmt, imgs = self._from_messages(req["messages"])
        procs = None
        if req.get("schema"):
            from mlx_vlm.structured import build_json_schema_logits_processor
            procs = [build_json_schema_logits_processor(self.processor.tokenizer, req["schema"])]
        with _quiet_fds():
            o = self._apc.generate(fmt, imgs, req.get("max_tokens", self.max_tokens), procs)
        return Result(o["text"], {k: v for k, v in o.items() if k != "text"})

    def _from_messages(self, messages):
        """OpenAI-style chat messages -> (templated prompt, images in order). Thinking is off."""
        import base64
        import io
        from PIL import Image
        conv, imgs = [], []
        for m in messages:
            c = m.get("content")
            if isinstance(c, str):
                conv.append({"role": m["role"], "content": c})
                continue
            parts = []
            for p in c or []:
                if p.get("type") == "text":
                    parts.append({"type": "text", "text": p["text"]})
                elif p.get("type") == "image_url":
                    url = p["image_url"]["url"]
                    raw = base64.b64decode(url.split(",", 1)[1])
                    imgs.append(Image.open(io.BytesIO(raw)).convert("RGB"))
                    parts.append({"type": "image"})
            if m["role"] == "system":           # templates want a plain-string system turn
                parts = "\n".join(p["text"] for p in parts if p["type"] == "text")
            conv.append({"role": m["role"], "content": parts})
        tmpl = getattr(self.processor, "apply_chat_template", None) or \
            self.processor.tokenizer.apply_chat_template
        fmt = tmpl(conv, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return fmt, imgs

    def yield_points(self):
        return discover_layer_classes(self.model)
