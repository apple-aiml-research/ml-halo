#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Prefix cache for mlx-vlm models: reuse the prompt state BEFORE the first image.

mlx-vlm's own caches need a text-only suffix; an orchestrator prompt is system prompt, then image.
Reusable prefixes are learned (longest common token prefix with recent prompts, cut before the
first media token) and kept as a few LRU snapshots. On a hit only the suffix is prefilled, with
M-RoPE positions from the FULL prompt. Outputs may differ in the last bits (prefill chunking).
"""
from __future__ import annotations

from typing import List


class PrefixCache:
    MIN_PREFIX, SLOTS, HISTORY, MIN_GAIN = 64, 6, 8, 64

    def __init__(self, model, processor):
        self.model, self.processor = model, processor
        cfg = model.config
        self.media_ids = {getattr(cfg, k, None) for k in (
            "image_token_index", "image_token_id", "video_token_index", "video_token_id",
            "vision_start_token_id")} - {None}
        tc = getattr(cfg, "text_config", None)
        if tc is not None:
            self.media_ids |= {getattr(tc, k, None) for k in ("vision_start_token_id",)} - {None}
        self.snaps: list = []                 # [(key token tuple, snapshot)], most recent last
        self.recent: list = []                # last prompts' ids, to learn shared prefixes

    @staticmethod
    def _copy(st):
        """A real copy: cache states are views onto buffers the model updates in place."""
        import mlx.core as mx
        if st is None:
            return None
        if isinstance(st, (list, tuple)):
            return type(st)(PrefixCache._copy(x) for x in st)
        c = mx.array(st) * 1          # new buffer
        mx.eval(c)
        return c

    @classmethod
    def _save(cls, cache) -> list:
        return [(cls._copy(c.state), getattr(c, "meta_state", None)) for c in cache]

    def _restore(self, snap):
        from mlx_vlm.models import cache as C
        fresh = C.make_prompt_cache(self.model.language_model)
        for c, (st, meta) in zip(fresh, snap):
            c.state = self._copy(st)  # the hit mutates it; the snapshot must stay pristine
            if meta is not None:
                c.meta_state = meta
        return fresh

    def _touch(self, sn):
        """LRU: move the snapshot just used to the back."""
        self.snaps = [x for x in self.snaps if x is not sn] + [sn]

    def generate(self, prompt: str, images: List, max_tokens: int, logits_processors=None):
        import mlx.core as mx
        from mlx_vlm.generate.ar import generate_step
        from mlx_vlm.generate.dispatch import _prepare_generation_inputs
        from mlx_vlm.models import cache as C

        model, lm = self.model, self.model.language_model
        ids, pv, mask, kw = _prepare_generation_inputs(model, self.processor, prompt,
                                                       images or None, None, None, {})
        idl = ids[0].tolist()
        media = next((i for i, t in enumerate(idl) if t in self.media_ids), len(idl) - 1)

        def lcp(a):
            n = min(media, len(a))
            return next((i for i in range(n) if idl[i] != a[i]), n)

        # Best snapshot = the longest one that is a prefix of this prompt.
        best = max((sn for sn in self.snaps if len(sn[0]) <= media
                    and tuple(idl[:len(sn[0])]) == sn[0]), key=lambda sn: len(sn[0]), default=None)
        have = len(best[0]) if best else 0
        # Snapshot both the LONGEST shared prefix and a STABLE one (shortest prefix shared with a
        # similar prompt): the user part drifts a little every call.
        lcps = [x for x in (lcp(a) for a in self.recent) if x >= self.MIN_PREFIX]
        longest = max(lcps, default=0)
        stable = min((x for x in lcps if x >= longest // 2), default=0)
        want = sorted({b for b in (stable, longest) if b > have + self.MIN_GAIN})
        self.recent = (self.recent + [idl])[-self.HISTORY:]
        pos, rd = lm.get_rope_index(ids, kw.get("image_grid_thw"), kw.get("video_grid_thw"), mask)
        common = dict(max_tokens=max_tokens, logits_processors=logits_processors,
                      sampler=lambda lp: mx.argmax(lp, axis=-1), **kw)
        learned = {}
        hit = best is not None
        base = have if hit else 0

        def keep(n, c):                             # n = tokens processed IN THIS CALL
            learned[base + n] = self._save(c)

        ck = [b - base for b in want] if want else None
        if hit:
            self._touch(best)
            common.update(position_ids=pos, rope_deltas=rd)       # FULL positions: lm slices
            steps = generate_step(ids[:, have:], model, pv, None, # them by absolute offset
                                  prompt_cache=self._restore(best[1]),
                                  prompt_cache_checkpoint=keep if ck else None,
                                  prompt_cache_checkpoint_lengths=ck, **common)
        else:
            steps = generate_step(ids, model, pv, mask, prompt_cache=C.make_prompt_cache(lm),
                                  prompt_cache_checkpoint=keep if ck else None,
                                  prompt_cache_checkpoint_lengths=ck, **common)

        tok = self.processor.tokenizer
        eos = model.config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {tok.eos_token_id}
        out = []
        for t, _ in steps:
            if t in eos:
                break
            out.append(t)
        for n, st in learned.items():               # learned for free during this call's prefill
            key = tuple(idl[:n])
            self.snaps = [sn for sn in self.snaps if sn[0] != key][-(self.SLOTS - 1):] + \
                [(key, st)]
        return dict(text=tok.decode(out), prompt_tokens=len(idl),
                    reused_tokens=base, generation_tokens=len(out),
                    truncated=len(out) >= max_tokens)
