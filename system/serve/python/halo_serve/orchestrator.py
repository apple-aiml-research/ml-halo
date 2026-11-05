#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""The orchestrator's local-model backend: a chat call served by the serving engine.

    from halo_serve.orchestrator import chat
    out = chat(messages, schema=..., max_tokens=512, timeout_s=45)   # OpenAI chat-completion dict

The call goes over HALO_SERVE_ENDPOINT (a Unix socket) to the GPU-owner (generator) process,
so both share one engine. The model is registered lazily on first call; the schema is enforced
by constrained decoding.
"""
from __future__ import annotations

import copy
import json
import os
import threading
import time
from concurrent.futures import Future

from halo_serve.runtime.service import register_op

DEFAULT_MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"
_lock = threading.Lock()
_names: dict = {}


def _local_path(model: str) -> str:
    """A repo id -> its cached snapshot dir when one exists (offline lookup fails without refs/main)."""
    if os.path.exists(os.path.expanduser(model)):
        return os.path.expanduser(model)
    hub = os.path.expanduser(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub"))
    snaps = os.path.join(hub, "models--" + model.replace("/", "--"), "snapshots")
    if os.path.isdir(snaps):
        for d in sorted(os.listdir(snaps)):
            if os.path.exists(os.path.join(snaps, d, "config.json")):
                return os.path.join(snaps, d)
    return model


def _workload_for(model: str) -> str:
    with _lock:
        if model in _names:
            return _names[model]
        from halo_serve.api import create
        from halo_serve.runtime.engine import global_engine
        eng = global_engine()
        name = f"orchestrator:{model.rsplit('/', 1)[-1]}"
        t0 = time.perf_counter()
        wl = create("mlx_vlm", model_path=_local_path(model), max_tokens=512)
        eng.register(name, wl)
        # Warm-up: pay kernel compile / allocator growth at load, not on the first decision.
        tw = time.perf_counter()
        eng.submit(name, {"messages": [{"role": "user", "content": "Reply with {}."}],
                          "schema": {"type": "object"}, "max_tokens": 8}).result()
        eng.meta[name] = {"role": "orchestrator", "model": model,
                          "load_s": round(tw - t0, 1),
                          "warmup_s": round(time.perf_counter() - tw, 1)}
        print(f"[halo_serve] orchestrator model {model} loaded in {time.perf_counter() - t0:.1f}s "
              f"(engine gpu-thread={eng.gpu_tid}, preemptible={eng._entries[name].preemptible})",
              flush=True)
        _names[model] = name
        return name


def _with_schema(messages, schema):
    """Append the schema to the system message: a grammar constrains tokens but does not explain fields."""
    m = copy.deepcopy(messages)
    note = ("\n\nRespond ONLY with a JSON object matching this JSON schema:\n"
            + json.dumps(schema, separators=(",", ":")))
    if m and m[0].get("role") == "system" and isinstance(m[0].get("content"), str):
        m[0]["content"] += note
    else:
        m.insert(0, {"role": "system", "content": note.strip()})
    return m


def _tight_ids(schema):
    """Restrict free-text `id` strings to identifier characters."""
    s = copy.deepcopy(schema)

    def walk(n):
        if isinstance(n, dict):
            p = n.get("properties")
            if isinstance(p, dict) and isinstance(p.get("id"), dict) \
                    and p["id"].get("type") == "string" and not {"enum", "pattern"} & set(p["id"]):
                p["id"]["pattern"] = "^[A-Za-z0-9_-]{1,32}$"
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)
    walk(s)
    return s


def _submit(args: dict):
    """Submit one chat job to this process's engine; returns a Future of an OpenAI-shaped dict."""
    from halo_serve.runtime.engine import global_engine
    schema, messages = args.get("schema"), args["messages"]
    if schema:
        schema = _tight_ids(schema)
        messages = _with_schema(messages, schema)
    name = _workload_for(args.get("model") or DEFAULT_MODEL)
    req = {"messages": messages, "schema": schema,
           "max_tokens": int(args.get("max_tokens", 512))}
    job = global_engine().submit(name, req, deadline_ms=float(args.get("timeout_s", 45)) * 1e3)
    out: Future = Future()
    job.add_done_callback(lambda f: out.set_exception(f.exception()) if f.exception()
                          else out.set_result(_as_chat_response(f.result())))
    return out


register_op("chat", _submit)


def chat(messages, schema=None, max_tokens: int = 512, timeout_s: float = 45.0,
         model: str = None) -> dict:
    args = {"messages": messages, "schema": schema, "max_tokens": int(max_tokens),
            "timeout_s": float(timeout_s), "model": model or DEFAULT_MODEL}
    ep = os.environ.get("HALO_SERVE_ENDPOINT")
    if not ep:
        raise RuntimeError("HALO_SERVE_ENDPOINT is not set: the orchestrator model is served by "
                           "the generator process (run_sim/run_real set it)")
    from halo_serve.runtime import service
    if ep not in service._started:          # another process owns the GPU: ask it
        return service.call(ep, "chat", args, timeout_s=float(timeout_s) + 5.0)
    return _submit(args).result(timeout=float(timeout_s) + 5.0)


def _as_chat_response(res) -> dict:
    info = res.info or {}
    return {"choices": [{"message": {"role": "assistant", "content": res.output},
                         "finish_reason": "length" if info.get("truncated") else "stop"}]}
