#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Model registry: adapters register under a name; configs refer to that name.

    @register("mlx_vlm")
    class MlxVlmReasoner(Workload): ...

    wl = create("mlx_vlm", model_path="...", max_tokens=96)

Built-in adapters register when `halo_serve.adapters` is imported.
"""
from __future__ import annotations

from typing import Callable, Dict, Type

_REGISTRY: Dict[str, Type] = {}


def register(name: str) -> Callable[[Type], Type]:
    def deco(cls: Type) -> Type:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"adapter name {name!r} already registered to {_REGISTRY[name]}")
        _REGISTRY[name] = cls
        return cls
    return deco


def create(name: str, **kwargs):
    _load_builtins()
    if name not in _REGISTRY:
        raise KeyError(f"unknown adapter {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def _load_builtins() -> None:
    import halo_serve.adapters  # noqa: F401
