#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""The adapter contract: any model the serving engine can run implements `Workload`.

Kinds: `policy` = short periodic jobs, tight deadline (motion generator); `reasoner` = long
jobs, loose deadline (VLM orchestrator). `yield_points()` lists layer classes the engine may
preempt between.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class Result:
    output: Any
    info: dict = field(default_factory=dict)


class Workload:
    """Base class. Subclasses override `load`, `run`, and optionally `yield_points`."""

    kind: str = "policy"           # "policy" | "reasoner"

    def load(self) -> None:
        """Build/load the model. Runs on the engine's GPU thread (MLX state is thread-bound)."""

    def run(self, req: Any) -> Result:
        """Run ONE job to completion (output materialised) and return it."""
        raise NotImplementedError

    def yield_points(self) -> List[type]:
        """Layer classes whose __call__ may be interrupted. [] = not interruptible."""
        return []


def discover_layer_classes(model) -> List[type]:
    """Default yield points: the class of every layer list at the usual attribute paths."""
    found = []
    paths = [("language_model", "model", "layers"), ("model", "layers"), ("layers",),
             ("vision_tower", "blocks"), ("vision_tower", "layers"),
             ("visual", "blocks"), ("vision_model", "layers")]
    for path in paths:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if isinstance(obj, (list, tuple)) and obj:
            cls = type(obj[0])
            if cls not in found:
                found.append(cls)
    return found
