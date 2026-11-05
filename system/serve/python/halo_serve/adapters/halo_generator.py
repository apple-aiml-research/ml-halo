#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Adapter: a motion generator (any encoder: ResNet or Qwen-VLM; DiT decoder).

    create("halo_generator", model=<already loaded MotionGenerator>)

load() moves the DiT Euler loop to MLX (`halo_serve.integrations.halo_system`).
A request is the dict `generate_motion_plan` would pass to predict():
    {"rgb": [3,H,W] tensor, "prompt": str, "prev_chunk": [P,D] tensor|None}
"""
from __future__ import annotations

from typing import Any

from halo_serve.api.registry import register
from halo_serve.api.workload import Result, Workload


@register("halo_generator")
class HaloGeneratorPolicy(Workload):
    kind = "policy"

    def __init__(self, model):
        self.model = model
        self.backend = "torch"

    def load(self) -> None:
        from halo_serve.integrations.halo_system import accelerate
        self.model.eval()
        self.backend = accelerate(self.model)

    def run(self, req: Any) -> Result:
        import torch
        y = self._predict(req["rgb"], req["prompt"], prev_chunk=req.get("prev_chunk"))
        if getattr(y, "device", None) is not None and y.device.type == "mps":
            torch.mps.synchronize()
        return Result(y, {"backend": self.backend})
