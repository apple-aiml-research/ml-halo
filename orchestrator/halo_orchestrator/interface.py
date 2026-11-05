#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""VLM client for the orchestrator: a local model on the halo_serve engine (structured output).

The engine is reached over a local socket and the motion generator can preempt it. The schema is
enforced as a decoding grammar. Requests and responses use the chat-completions message format.
"""
from __future__ import annotations
import base64, io, json
from typing import Any, Optional


# Ego frames are downscaled to fit (W, H) before they go to the VLM.
DEFAULT_MAX_PX = (448, 280)

DEFAULT_SERVE_MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"


def _data_url(img: Any, quality: int = 88) -> str:
    """`data:image/jpeg;base64,...` of an RGB array, downscaled to fit DEFAULT_MAX_PX."""
    from PIL import Image  # lazy
    pil = Image.fromarray(img).convert("RGB")
    mw, mh = DEFAULT_MAX_PX
    if pil.width > mw or pil.height > mh:
        pil.thumbnail((mw, mh), Image.Resampling.LANCZOS)
    buf = io.BytesIO(); pil.save(buf, format="JPEG", quality=quality)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


class VLMInterface:
    def __init__(self, model: Optional[str] = None,
                 timeout_s: float = 45.0, timeout_s_long: float = 90.0):
        self.timeout_s = float(timeout_s)
        # For plan authoring (longer output, cold-start warm-up); must stay above timeout_s.
        self.timeout_s_long = float(timeout_s_long)
        # `model` is a repo id or local path.
        self.model = model or DEFAULT_SERVE_MODEL

    @staticmethod
    def user_content(text: str, image: Any = None, frames: Optional[list] = None) -> list:
        """Build a chat-style user `content` list: a text part + image part(s)."""
        content = [{"type": "text", "text": text}]
        # Truthiness on purpose: an empty frames list falls back to `image`.
        seq = frames if frames else ([("", image)] if image is not None else [])
        for label, img in seq:
            if label:
                content.append({"type": "text", "text": label})
            content.append({"type": "image_url",
                            "image_url": {"url": _data_url(img)}})
        return content

    def decide_messages(self, messages: list, max_tokens: int, schema: Optional[dict] = None,
                        timeout_s: Optional[float] = None) -> dict:
        """Structured decision over a full message history.

        Returns {"data", "raw", "truncated"}. `truncated` marks a max_tokens cut (a JSON prefix
        that fails to parse)."""
        try:
            from halo_serve.orchestrator import chat
        except ImportError as e:
            raise RuntimeError(
                "the orchestrator needs halo_serve (put <repo>/system on PYTHONPATH)") from e
        out = chat(messages, schema=schema, max_tokens=int(max_tokens),
                   timeout_s=float(timeout_s or self.timeout_s), model=self.model)
        choice = out["choices"][0]
        answer = (choice["message"].get("content") or "").strip()
        return {"data": _parse_json(answer), "raw": answer,
                "truncated": choice.get("finish_reason") == "length"}


def _parse_json(answer: str):
    """json.loads of the schema-constrained answer; None if empty or unparseable."""
    try:
        return json.loads(answer) if answer else None
    except ValueError:
        return None
