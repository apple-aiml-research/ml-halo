#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Decompose ONNX LayerNormalization nodes into primitive ops.

TensorRT < 8.6 has no native `LayerNormalization` importer (and no matching
plugin), so an opset-17+ ONNX policy that contains it fails to parse on e.g.
JetPack 5's TRT 8.5.2:

    [E] No importer registered for op: LayerNormalization ...
    [8] Assertion failed: creator && "Plugin not found ..."

This rewrites every LayerNormalization node into the equivalent subgraph
(ReduceMean / Sub / Mul / Add / Sqrt / Div), which every TRT 8.5+ parser
supports, so onnx_to_trt can build the plan. ReduceMean is emitted in the form
the model's declared opset requires: `axes` as an attribute below opset 18, as
an input from opset 18. The math matches the ONNX
LayerNormalization spec:

    mean = ReduceMean(X, axes)
    d    = X - mean
    var  = ReduceMean(d*d, axes)
    y    = d / Sqrt(var + epsilon) * Scale (+ B)

Usage:
    decompose_layernorm.py IN.onnx [OUT.onnx]
    (default OUT: <IN without .onnx>.ln.onnx)

Validate parity before trusting it (needs onnxruntime). Use random inputs: an
all-zero input makes the variance zero, so a wrong rewrite would still match.
    import onnxruntime as ort, numpy as np
    a = ort.InferenceSession("in.onnx"); b = ort.InferenceSession("out.onnx")
    rng = np.random.default_rng(0)
    feeds = {i.name: rng.standard_normal([d if isinstance(d, int) and d > 0 else 1
                                          for d in i.shape]).astype(np.float32)
             for i in a.get_inputs()}
    for x, y in zip(a.run(None, feeds), b.run(None, feeds)):
        assert np.allclose(x, y, atol=1e-5)
"""
from __future__ import annotations  # PEP 604 (int | None) on Python < 3.10

import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _normalized_axes(axis: int, rank: int | None) -> list[int]:
    # Spec: normalize over [axis .. last]. Negative axis is handled without the
    # rank; a non-negative axis needs the input rank.
    if axis < 0:
        return list(range(axis, 0))
    if rank is None:
        raise ValueError(
            f"LayerNormalization axis={axis} (non-negative) needs the input rank; "
            "re-export the model with axis=-1, or with opset<=16."
        )
    return list(range(axis, rank))


def _rank_of(graph, name: str) -> int | None:
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        if vi.name == name and vi.type.tensor_type.HasField("shape"):
            return len(vi.type.tensor_type.shape.dim)
    return None


def _default_opset(model: onnx.ModelProto) -> int:
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx"):
            return imp.version
    raise ValueError("model has no default-domain opset import")


def decompose(model: onnx.ModelProto) -> onnx.ModelProto:
    g = model.graph
    # ReduceMean takes `axes` as an attribute up to opset 17 and as an input from
    # opset 18. LayerNormalization exists from opset 17, so both forms occur.
    axes_as_input = _default_opset(model) >= 18
    new_nodes = []
    n_ln = 0
    for node in g.node:
        if node.op_type != "LayerNormalization":
            new_nodes.append(node)
            continue
        n_ln += 1
        x = node.input[0]
        scale = node.input[1]
        bias = node.input[2] if len(node.input) > 2 and node.input[2] else None
        y = node.output[0]  # keep the original output name for downstream wiring

        axis = -1
        eps = 1e-5
        for a in node.attribute:
            if a.name == "axis":
                axis = a.i
            elif a.name == "epsilon":
                eps = a.f

        axes = _normalized_axes(axis, _rank_of(g, x))
        p = f"{node.name or 'ln'}_{n_ln}"

        eps_init = numpy_helper.from_array(
            np.array(eps, dtype="float32"), name=f"{p}_eps")
        g.initializer.append(eps_init)
        if axes_as_input:
            g.initializer.append(numpy_helper.from_array(
                np.array(axes, dtype="int64"), name=f"{p}_axes"))

        def reduce_mean(inp: str, out: str, name: str):
            if axes_as_input:
                return helper.make_node("ReduceMean", [inp, f"{p}_axes"], [out], name,
                                        keepdims=1)
            return helper.make_node("ReduceMean", [inp], [out], name, axes=axes, keepdims=1)

        mean = f"{p}_mean"; d = f"{p}_d"; dd = f"{p}_dd"; var = f"{p}_var"
        vare = f"{p}_vare"; std = f"{p}_std"; norm = f"{p}_norm"
        scaled = y if bias is None else f"{p}_scaled"

        new_nodes += [
            reduce_mean(x, mean, f"{p}_rm1"),
            helper.make_node("Sub", [x, mean], [d], f"{p}_sub"),
            helper.make_node("Mul", [d, d], [dd], f"{p}_sq"),
            reduce_mean(dd, var, f"{p}_rm2"),
            helper.make_node("Add", [var, f"{p}_eps"], [vare], f"{p}_addeps"),
            helper.make_node("Sqrt", [vare], [std], f"{p}_sqrt"),
            helper.make_node("Div", [d, std], [norm], f"{p}_div"),
            helper.make_node("Mul", [norm, scale], [scaled], f"{p}_scale"),
        ]
        if bias is not None:
            new_nodes.append(helper.make_node("Add", [scaled, bias], [y], f"{p}_bias"))

    del g.node[:]
    g.node.extend(new_nodes)
    print(f"[decompose_layernorm] rewrote {n_ln} LayerNormalization node(s)")
    onnx.checker.check_model(model)
    return model


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else in_path[:-5] + ".ln.onnx" \
        if in_path.endswith(".onnx") else in_path + ".ln.onnx"
    model = onnx.load(in_path)
    model = decompose(model)
    onnx.save(model, out_path)
    print(f"[decompose_layernorm] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
