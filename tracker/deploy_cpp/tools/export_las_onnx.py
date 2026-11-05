#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Export a LiteAnyStereo checkpoint to the ONNX the depth student's stereo engine expects.

Calls the upstream exporter (``export_onnx.py::export()`` of github.com/TomTomTommi/LiteAnyStereo; clone it and put
the checkpoints from its README's "Checkpoints" section under ``checkpoints/``) and adds the two properties of the target that upstream does not set:

- **opset**: upstream hardcodes 18; TensorRT 8.5 (JetPack 5, the stock G1 Orin) parses up to 17, so ``--opset``
  overrides it (16 if a TRT build still rejects a node).
- **geometry**: the feature pyramid needs dims divisible by 32 and upstream takes ``--width/--height`` verbatim, so
  the capture size (424x240 in ``configs/depth_d435i.yaml``) is rounded up (448x256). ``StereoDepthEngine`` reads the
  padded dims from the plan and replicate-pads the capture to them.

The IO contract (inputs ``left`` / ``right`` FP32 (1,3,H,W) holding 0-255 values, ImageNet normalisation in the graph,
output ``disparity`` FP32 (1,1,H,W) in pixels) comes from upstream unchanged.

The defaults reproduce the plan the deployed depth student used (``las_424x240_orin.plan``): LAS1 with
``checkpoints/LiteAnyStereo.pth``, max disparity 192, opset 17.

    python3 tools/export_las_onnx.py --las_repo <LiteAnyStereo checkout> \\
        --restore_ckpt <LiteAnyStereo checkout>/checkpoints/LiteAnyStereo.pth --output las_448x256.onnx
    # then ON THE ORIN (a .plan is not portable across devices or TRT versions):
    ./build/release-rt-orin/onnx_to_trt --onnx las_448x256.onnx --plan las_424x240_orin.plan --weakly-typed --fp16

LAS2-S/M/L/H share the contract: ``--version las2
--model_size s`` and a matching ``--depth_plan`` at run time.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import sys
from pathlib import Path

PAD_DIVISOR = 32


def round_up(value: int, divisor: int = PAD_DIVISOR) -> int:
    return ((value + divisor - 1) // divisor) * divisor


@contextlib.contextmanager
def forced_opset(opset: int):
    """Force ``opset_version`` on every ``torch.onnx.export`` call in this block (upstream passes 18 itself)."""
    import torch

    real = torch.onnx.export

    def patched(*args, **kwargs):
        kwargs["opset_version"] = opset
        return real(*args, **kwargs)

    torch.onnx.export = patched
    try:
        yield
    finally:
        torch.onnx.export = real


def describe_onnx(path: Path) -> None:
    """Print the exported IO contract, to compare with what StereoDepthEngine requires."""
    try:
        import onnx
    except ImportError:
        print("[info] onnx not installed; skipping the IO dump")
        return
    model = onnx.load(str(path))
    initialisers = {i.name for i in model.graph.initializer}

    def shape_of(vi) -> str:
        dims = [str(d.dim_value) if d.dim_value else (d.dim_param or "?") for d in vi.type.tensor_type.shape.dim]
        elem = onnx.TensorProto.DataType.Name(vi.type.tensor_type.elem_type)
        return f"{vi.name}: {elem}[{','.join(dims)}]"

    print("\n---- exported ONNX IO ----")
    print(f"  opset : {[(o.domain or 'ai.onnx') + ':' + str(o.version) for o in model.opset_import]}")
    for vi in model.graph.input:
        if vi.name not in initialisers:
            print(f"  input  {shape_of(vi)}")
    for vi in model.graph.output:
        print(f"  output {shape_of(vi)}")
    ops = sorted({n.op_type for n in model.graph.node})
    print(f"  {len(model.graph.node)} nodes, {len(ops)} distinct op types")
    watch = [o for o in ops if o in {"LayerNormalization", "InstanceNormalization", "GridSample",
                                     "GroupNormalization", "Einsum", "ScatterND", "NonZero"}]
    if watch:
        print(f"  NOTE: ops that may need a TRT plugin or TRT >= 8.6: {', '.join(watch)}")
        print("        (the stereo engine loads the TRT plugins, which covers InstanceNormalization on TRT 8.5)")
    print("--------------------------\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--las_repo", type=Path, required=True,
                    help="LiteAnyStereo checkout (the one with export_onnx.py)")
    ap.add_argument("--version", default="las1", help="las1 | las2 (default: las1, the deployed plan)")
    ap.add_argument("--model_size", default="s", help="LAS2 size: s | m | l | h (ignored for las1)")
    ap.add_argument("--restore_ckpt", required=True, help="the .pth checkpoint")
    ap.add_argument("--width", type=int, default=424, help="capture width, rounded up to a multiple of 32")
    ap.add_argument("--height", type=int, default=240, help="capture height, rounded up to a multiple of 32")
    ap.add_argument("--max_disp", type=int, default=192, help="max disparity, baked into the graph (default: 192)")
    ap.add_argument("--opset", type=int, default=17,
                    help="ONNX opset: 17 for TRT 8.5, 16 if the TRT build rejects a node (default: 17)")
    ap.add_argument("--output", required=True, help="output .onnx path")
    ap.add_argument("--simplify", action="store_true", help="run onnxsim on the result (upstream's --simplify)")
    args = ap.parse_args()

    repo = args.las_repo.expanduser().resolve()
    if not (repo / "export_onnx.py").is_file():
        print(f"error: {repo} has no export_onnx.py; is it a LiteAnyStereo checkout?", file=sys.stderr)
        return 2
    ckpt = Path(args.restore_ckpt).expanduser().resolve()
    if not ckpt.is_file():
        print(f"error: checkpoint not found: {ckpt}", file=sys.stderr)
        return 2
    out = Path(args.output).expanduser().resolve()

    pad_w, pad_h = round_up(args.width), round_up(args.height)
    if (pad_w, pad_h) != (args.width, args.height):
        print(f"[geometry] {args.width}x{args.height} -> {pad_w}x{pad_h} (multiple of {PAD_DIVISOR})")
    size = "-" + args.model_size.upper() if args.version != "las1" else ""
    print(f"[export] {args.version}{size} @ {pad_w}x{pad_h}  max_disp={args.max_disp}  opset={args.opset}  ckpt={ckpt}")

    # Upstream resolves paths relative to the CWD and imports `core.*` by bare name: run from inside the checkout.
    sys.path.insert(0, str(repo))
    prev_cwd = Path.cwd()
    os.chdir(repo)
    try:
        export_onnx = importlib.import_module("export_onnx")
        with forced_opset(args.opset):
            export_onnx.export(version=args.version, model_size=args.model_size, restore_ckpt=str(ckpt),
                               width=pad_w, height=pad_h, max_disp=args.max_disp, output_name=str(out),
                               simplify=args.simplify)
    finally:
        os.chdir(prev_cwd)
        sys.path.remove(str(repo))

    if not out.is_file():
        print(f"error: the upstream export returned but {out} does not exist", file=sys.stderr)
        return 1
    print(f"[export] wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    describe_onnx(out)
    print("Next, on the Orin:")
    print(f"  ./build/release-rt-orin/onnx_to_trt --onnx {out.name} --plan <plan> --weakly-typed --fp16")
    return 0


if __name__ == "__main__":
    sys.exit(main())
