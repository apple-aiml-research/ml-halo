#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
Vision-Language Model (VLM) encoder using Qwen3.5.

A single multimodal backbone that processes the RGB frame and the language
command together and returns image and text features for the motion decoder.

Supports:
- Qwen3.5-2B (hidden_dim=2048)
- Qwen3.5-4B (hidden_dim=2560)
"""

import torch
import torch.nn as nn
from typing import List
from torchvision.transforms.functional import to_pil_image


# Model ID lookup
VLM_MODEL_IDS = {
    "qwen3.5-2b": "Qwen/Qwen3.5-2B",
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
}

# Output dimension lookup
VLM_ENCODER_DIMS = {
    "qwen3.5-2b": 2048,
    "qwen3.5-4b": 2560,
}


def get_vlm_encoder_dim(vlm_backbone: str) -> int:
    """Get output dimension for a VLM backbone."""
    if vlm_backbone not in VLM_ENCODER_DIMS:
        raise ValueError(
            f"Unknown VLM backbone: {vlm_backbone}. "
            f"Available: {list(VLM_ENCODER_DIMS.keys())}"
        )
    return VLM_ENCODER_DIMS[vlm_backbone]


class VLMEncoder(nn.Module):
    """
    Unified vision-language encoder using Qwen3.5.

    Processes RGB images and text together through a pretrained multimodal
    model and returns every image token and every instruction token of the last
    hidden layer.

    The VLM weights are frozen by default and excluded from checkpoints.
    """

    def __init__(
        self,
        model_name: str,
        frozen: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        vision_trainable: bool = False,
        lm_trainable: bool = False,
        gpu_preprocess: bool = False,
    ):
        """
        Initialize VLMEncoder.

        Args:
            model_name: Short name (e.g. "qwen3.5-2b") or full HuggingFace ID
            frozen: Freeze VLM weights (recommended)
            dtype: Model dtype (bfloat16 recommended for memory efficiency)
            vision_trainable: unfreeze the VISION TOWER (all of its layers); the
                language model stays frozen unless lm_trainable is also set.
            lm_trainable: also unfreeze the language model (requires
                vision_trainable).
            gpu_preprocess: run the HF image processor on the GPU tensors and
                reuse cached prompt tokens per task string, instead of .cpu() ->
                PIL -> processor -> .to(device) plus a .tolist() span scan on
                every call. Same ops, same inputs; checked against the CPU path
                on the first batch (_check_fast).
        """
        super().__init__()

        if lm_trainable and not vision_trainable:
            raise ValueError("lm_trainable=True requires vision_trainable=True "
                             "(training the LM while the tower feeding it is "
                             "frozen is not a configuration we mean to support)")
        self.frozen = frozen
        self.dtype = dtype
        self.vision_trainable = vision_trainable
        self.lm_trainable = lm_trainable
        self.gpu_preprocess = gpu_preprocess
        self._prompt_cache = {}
        self._fast_checked = False

        # Resolve model ID
        if model_name in VLM_MODEL_IDS:
            self.model_id = VLM_MODEL_IDS[model_name]
        else:
            self.model_id = model_name

        # Load model and processor
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # Which linear-attention implementation the Qwen3.5 Gated DeltaNet layers
        # got. Without flash-linear-attention, transformers falls back to a torch
        # loop (63 Python iterations per layer per chunk) whose backward was ~70%
        # of all backward GPU ops in one profile, leaving the GPU idle.
        for m in self.model.modules():
            fn = getattr(m, "chunk_gated_delta_rule", None)
            if fn is not None:
                conv = getattr(m, "causal_conv1d_fn", None)
                try:
                    import triton
                    tri = f"; triton {triton.__version__} ({triton.__file__})"
                except Exception:
                    tri = "; triton not importable"
                print(f"[VLMEncoder] linear attention: chunk_gated_delta_rule from "
                      f"{getattr(fn, '__module__', fn)}; causal_conv1d "
                      f"{'from ' + conv.__module__ if conv is not None else 'torch fallback'}"
                      f"{tri}", flush=True)
                break

        # Read output dimension from model config
        # Multimodal models have nested configs (text_config / vision_config)
        if hasattr(self.model.config, "text_config"):
            self.output_dim = self.model.config.text_config.hidden_size
        else:
            self.output_dim = self.model.config.hidden_size

        if frozen:
            self.model.requires_grad_(False)
            self.model.eval()

        self._visual = self._find_visual()
        if vision_trainable and self._visual is None:
            raise RuntimeError("vision_trainable=True but no vision tower found")
        # Anything trainable moves to fp32: AdamW on bf16 parameters has no fp32
        # master copy, so small updates to pretrained weights are rounded away
        # (bf16 keeps 8 mantissa bits).
        if lm_trainable:
            self.model.to(torch.float32)
            self.model.requires_grad_(True)
            self.dtype = torch.float32          # _prepare_inputs casts to this
            what = "ENTIRE VLM (vision tower + language model)"
        elif vision_trainable:
            self._visual.to(torch.float32)
            self._visual.requires_grad_(True)
            what = "vision tower only (language model stays frozen)"
        if vision_trainable:
            n = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"[VLMEncoder] TRAINABLE in fp32: {n:,} params — {what}", flush=True)

    def _find_visual(self):
        """The vision tower submodule, or None."""
        return getattr(self.model, "visual", None) or getattr(
            getattr(self.model, "model", None), "visual", None)

    def trainable_vlm_parameters(self):
        """Trainable VLM parameters (the vision tower, plus the language model
        when lm_trainable), for a dedicated low-LR optimizer group."""
        if not self.vision_trainable:
            return []
        src = self.model if self.lm_trainable else self._visual
        return [p for p in src.parameters() if p.requires_grad]

    def _prepare_inputs(self, images: torch.Tensor, texts: List[str]) -> dict:
        """Tensors + task strings -> the processor kwargs the VLM consumes.

        The prompt construction defines the conditioning the trained models
        expect: do not change it.
        """
        B = images.shape[0]
        device = next(self.model.parameters()).device

        # Convert tensors to PIL images (processor expects PIL)
        pil_images = [to_pil_image(images[i].cpu().clamp(0, 1)) for i in range(B)]

        # Build chat-template messages for each sample
        batch_messages = []
        for i in range(B):
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_images[i]},
                    {"type": "text", "text": texts[i]},
                ],
            }]
            batch_messages.append(messages)

        # Apply chat template and process
        batch_texts = [
            self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            for msgs in batch_messages
        ]
        inputs = self.processor(
            text=batch_texts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
        )

        # Move inputs to model device and dtype
        return {
            k: v.to(device=device, dtype=self.dtype)
            if v.dtype.is_floating_point
            else v.to(device=device)
            for k, v in inputs.items()
        }

    def _image_spans(self, input_ids: torch.Tensor) -> List[tuple]:
        """Per-sample [start, end) index range of the image tokens.

        The span is what sits between <|vision_start|> and <|vision_end|>, i.e.
        the <|image_pad|> positions the vision tower's tokens were spliced into.
        """
        tokenizer = self.processor.tokenizer
        vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")

        spans = []
        for b in range(input_ids.shape[0]):
            ids = input_ids[b].tolist()
            img_start, img_end = None, None
            for idx, tid in enumerate(ids):
                if tid == vision_start_id:
                    img_start = idx + 1
                if tid == vision_end_id:
                    img_end = idx
                    break
            spans.append((img_start, img_end))
        return spans

    def _prepare(self, images: torch.Tensor, texts: List[str]):
        """-> (model inputs, per-sample image spans, image_grid_thw of sample 0).

        Spans and grid come back as Python ints. On the gpu_preprocess path they
        come from the prompt cache, so nothing here waits on the GPU.
        """
        if self.gpu_preprocess:
            fast = self._prepare_fast(images, texts)
            if not self._fast_checked:
                self._check_fast(images, texts, fast)
                self._fast_checked = True
            return fast
        inputs = self._prepare_inputs(images, texts)
        grid = inputs.get("image_grid_thw")
        grid = tuple(int(x) for x in grid[0]) if grid is not None else None
        return inputs, self._image_spans(inputs["input_ids"]), grid

    def _prompt(self, text: str, height: int, width: int) -> dict:
        """Processor output for one (task string, image size), minus the pixels.

        The prompt only depends on the text and the image grid (which fixes how
        many <|image_pad|> tokens are spliced in), so it is computed once with a
        blank image of the right size.
        """
        key = (text, height, width)
        hit = self._prompt_cache.get(key)
        if hit is None:
            blank = to_pil_image(torch.zeros(3, height, width))
            chat = self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image", "image": blank},
                                              {"type": "text", "text": text}]}],
                tokenize=False, add_generation_prompt=False)
            ref = self.processor(text=[chat], images=[blank],
                                 return_tensors="pt", padding=True)
            seq = {}
            for k, v in ref.items():
                if k in ("pixel_values", "image_grid_thw"):
                    continue
                if v.ndim != 2 or v.shape[0] != 1:
                    raise RuntimeError(f"unexpected processor output {k} {tuple(v.shape)}")
                seq[k] = v[0]
            hit = {
                "seq": seq,
                "span": self._image_spans(ref["input_ids"])[0],
                "grid": tuple(int(x) for x in ref["image_grid_thw"][0]),
            }
            self._prompt_cache[key] = hit
        return hit

    def _prepare_fast(self, images: torch.Tensor, texts: List[str]):
        """_prepare_inputs on the GPU: same processor ops, no host round trip."""
        B, _, H, W = images.shape
        device = next(self.model.parameters()).device

        # Same uint8 quantisation to_pil_image applies (mul(255).byte(), which
        # truncates), then the HF image processor on the GPU tensors: resize,
        # rescale/normalize and patchify are the same torch ops the PIL path
        # runs, just without the .cpu() and the PIL round trip.
        u8 = images.to(device).clamp(0, 1).mul(255).to(torch.uint8)
        pix = self.processor.image_processor(images=list(u8.unbind(0)),
                                             return_tensors="pt")

        # Prompt tokens: cached per task string, padded like processor(padding=True).
        prompts = [self._prompt(t, H, W) for t in texts]
        lens = [len(p["seq"]["input_ids"]) for p in prompts]
        L = max(lens)
        left = self.processor.tokenizer.padding_side == "left"
        pad_id = self.processor.tokenizer.pad_token_id
        out = {}
        for k in prompts[0]["seq"]:
            fill = pad_id if k == "input_ids" else 0
            t = torch.full((B, L), fill, dtype=prompts[0]["seq"][k].dtype)
            for b, p in enumerate(prompts):
                n = lens[b]
                if left:
                    t[b, L - n:] = p["seq"][k]
                else:
                    t[b, :n] = p["seq"][k]
            out[k] = t
        spans = []
        for b, p in enumerate(prompts):
            off = L - lens[b] if left else 0
            s, e = p["span"]
            spans.append((s + off, e + off))
        out["pixel_values"] = pix["pixel_values"]
        out["image_grid_thw"] = pix["image_grid_thw"]

        # Plain (blocking) copies: the token tensors are freshly built pageable
        # CPU temporaries. On CUDA a non_blocking copy from pageable memory is
        # synchronous anyway, but on MPS it is truly async and the source can be
        # freed/reused mid-copy -- input_ids/attention_mask arrived corrupted
        # (caught by _check_fast on MPS). They are tiny.
        inputs = {
            k: v.to(device=device, dtype=self.dtype)
            if v.dtype.is_floating_point
            else v.to(device=device)
            for k, v in out.items()
        }
        return inputs, spans, prompts[0]["grid"]

    def _check_fast(self, images, texts, fast):
        """Compare the GPU path with the CPU reference once, on a real batch.

        Guards against a transformers version whose image processor silently
        moves tensors to the CPU, or handles tensors differently from PIL. Token
        keys must match exactly. Pixels do NOT match bit for bit: the CPU uint8
        bicubic resize (224 -> 256) is a fixed-point kernel, the CUDA one works
        in float, so high-contrast edges differ by a few uint8 levels. Measured
        on the first batch of a 40-rank run: max|d| 0.023-0.047 (3-6
        levels of 2/255 after the [-1, 1] normalisation), mean|d| ~0.00095
        (~0.12 level). A real bug (patch order, normalisation, wrong grid) is
        orders of magnitude larger, so the bounds are 0.1 max / 2e-3 mean.
        """
        inputs, spans, grid = fast
        ref = self._prepare_inputs(images, texts)
        ref_spans = self._image_spans(ref["input_ids"])
        ref_grid = tuple(int(x) for x in ref["image_grid_thw"][0])
        problems = []
        if set(ref) != set(inputs):
            problems.append(f"keys {sorted(inputs)} vs reference {sorted(ref)}")
        for k in set(ref) & set(inputs):
            a, r = inputs[k], ref[k]
            if a.shape != r.shape or a.dtype != r.dtype or a.device != r.device:
                problems.append(f"{k}: {tuple(a.shape)} {a.dtype} {a.device} vs "
                                f"{tuple(r.shape)} {r.dtype} {r.device}")
            elif k != "pixel_values" and not torch.equal(a, r):
                problems.append(f"{k}: values differ")
        if spans != ref_spans or grid != ref_grid:
            problems.append(f"spans/grid {spans[:2]} {grid} vs {ref_spans[:2]} {ref_grid}")
        stats = ""
        if not problems:
            d = (inputs["pixel_values"].float() - ref["pixel_values"].float()).abs()
            stats = f"pixel max|d| {d.max().item():.4g} mean|d| {d.mean().item():.3g}"
            if d.max().item() > 0.1 or d.mean().item() > 2e-3:
                problems.append(stats)
        if problems:
            raise RuntimeError("[VLMEncoder] gpu_preprocess does not match the CPU "
                               "path: " + "; ".join(problems))
        print(f"[VLMEncoder] gpu_preprocess matches the CPU path on a batch of "
              f"{len(texts)}: tokens/spans/grid identical, {stats}", flush=True)

    def forward(self, images: torch.Tensor, texts: List[str]) -> tuple:
        """
        Encode images and text together through the VLM.

        Args:
            images: [B, 3, H, W] RGB tensors in [0, 1] range
            texts: List of B text strings

        Returns:
            (image_tokens, text_tokens, text_mask), from the last hidden layer:
            - image_tokens: [B, N, output_dim] float32, every image token (the
              merged grid, row-major)
            - text_tokens: [B, T, output_dim] float32, every token after
              <|vision_end|> through the last real one (task words, <|im_end|>,
              trailing newline), zero-padded to the longest prompt in the batch
            - text_mask: [B, T] key-padding mask, True = pad
        """
        B = images.shape[0]
        device = next(self.model.parameters()).device
        inputs, spans, _ = self._prepare(images, texts)

        # no_grad only when NOTHING needs gradients; with a trainable tower the
        # graph must be kept even though self.frozen is True for the LM.
        if self.frozen and not self.vision_trainable:
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
        else:
            outputs = self.model(**inputs, output_hidden_states=True)
        hs = outputs.hidden_states[-1]                  # [B, seq_len, output_dim]

        if len({e - s for s, e in spans}) != 1:
            raise RuntimeError(f"ragged image spans in one batch: {spans}")
        image_tokens = torch.stack([hs[b, s:e] for b, (s, e) in enumerate(spans)])

        # Text tokens: ragged across task strings, so padded with a key-padding
        # mask. The .tolist() is a host sync.
        last = (inputs["attention_mask"].sum(dim=1) - 1).tolist()
        starts = [e + 1 for _, e in spans]              # e = index of <|vision_end|>
        lens = [l - s + 1 for s, l in zip(starts, last)]
        T = max(lens)
        text_idx = torch.zeros(B, T, dtype=torch.long)
        text_mask = torch.ones(B, T, dtype=torch.bool)
        for b in range(B):
            text_idx[b, :lens[b]] = torch.arange(starts[b], last[b] + 1)
            text_mask[b, :lens[b]] = False
        text_idx, text_mask = text_idx.to(device), text_mask.to(device)
        text_tokens = hs[torch.arange(B, device=device)[:, None], text_idx].float()
        text_tokens = text_tokens.masked_fill(text_mask[..., None], 0.0)
        return image_tokens.float(), text_tokens, text_mask

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        """Frozen VLM weights come from HuggingFace, so they stay out of the
        checkpoint. TRAINABLE ones must be saved — returning {} unconditionally
        would discard the fine-tune at every save.

        Must honour `destination`: when the parent MotionGenerator saves, it
        passes one shared dict down and ignores the return value. An earlier
        version called super().state_dict(destination=...) and filtered only its
        own return value, so the full frozen LM (6.1 GB) leaked into every
        checkpoint with a trainable tower. Build into a private dict, filter,
        then publish.
        """
        if not self.vision_trainable:
            return destination if destination is not None else {}
        full = super().state_dict(prefix=prefix, keep_vars=keep_vars)
        train = self._trainable_names()
        kept = {k: v for k, v in full.items() if k[len(prefix):] in train}
        if destination is None:
            from collections import OrderedDict
            return OrderedDict(kept)
        destination.update(kept)
        return destination

    def _trainable_names(self):
        """Names (encoder-relative) of parameters that require grad."""
        return {n for n, p in self.named_parameters() if p.requires_grad}

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Load only the vision tower, and only when it is trainable.

        Loads straight into the tower submodule. Calling self.load_state_dict()
        here would re-enter this hook and recurse until the stack blew.
        """
        if not self.vision_trainable:
            return
        train = self._trainable_names()
        sub = {}
        for k, v in state_dict.items():
            tail = k[len(prefix):] if prefix and k.startswith(prefix) else k
            if tail in train:
                sub[tail] = v
        if not sub:
            print("[VLMEncoder] WARNING trainable VLM but the checkpoint has no "
                  "VLM weights; keeping the pretrained ones", flush=True)
            return
        # Assign directly: calling self.load_state_dict() here would re-enter this
        # hook and recurse until the stack blew.
        own = dict(self.named_parameters())
        n = 0
        with torch.no_grad():
            for k, v in sub.items():
                if k in own and own[k].shape == v.shape:
                    own[k].copy_(v)
                    n += 1
        print(f"[VLMEncoder] loaded {n}/{len(sub)} trainable VLM tensors from "
              f"checkpoint", flush=True)

    def train(self, mode: bool = True):
        """Keep the frozen language model in eval; let a trainable tower train."""
        if self.frozen and not self.vision_trainable:
            super().train(False)
            return self
        super().train(False)                 # whole VLM to eval (LM frozen)
        if self.vision_trainable and self._visual is not None:
            self._visual.train(mode)         # tower follows the requested mode
        return self
