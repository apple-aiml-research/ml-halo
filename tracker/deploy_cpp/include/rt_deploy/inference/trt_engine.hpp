//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// TrtEngine — TensorRT backend for IInferenceEngine (TensorRT 10 API; the
// enqueueV3 path also builds against TensorRT 8.5+ on older JetPacks).
//
// Responsibilities (all set up at construction, nothing on the hot path):
//   - Deserialize the .plan file produced by tools/onnx_to_trt.
//   - Allocate pinned host buffers (cudaMallocHost) for input/output.
//   - Allocate device buffers (cudaMalloc) for input/output.
//   - Create one CUDA stream shared by all per-call ops.
//   - Resolve input/output tensor addresses once via setInputTensorAddress /
//     setOutputTensorAddress (TRT 10 API).
//
// Hot-path infer_multi():
//   memcpy(input  → pinned host)
//   either:
//     cudaGraphLaunch(captured_graph, stream)          -- fast path
//   or (when use_cuda_graph==false or capture failed):
//     cudaMemcpyAsync(pinned host → device, H2D, stream)
//     context->enqueueV3(stream)
//     cudaMemcpyAsync(device → pinned host, D2H, stream)
//   cudaStreamSynchronize(stream)
//   memcpy(pinned host → output)
//
// CUDA Graphs note: capturing the H2D + enqueueV3 + D2H sequence into a graph
// at init eliminates per-call kernel-launch overhead. For small / launch-bound
// policies on Jetson this typically halves p50 latency and tightens p999
// jitter. Buffer pointers are stable across calls so capture is safe; falls
// back transparently to the explicit sequence if capture fails.
//
// All failures during construction surface via a Result<>-returning factory.
// Operator errors on the hot path return InferenceStatus and do not throw.

#include "rt_deploy/core/expected.hpp"
#include "rt_deploy/inference/i_inference_engine.hpp"

#include <memory>
#include <string>
#include <string_view>

namespace rt_deploy::inference {

class TrtEngine;  // forward declaration — implementation hidden in .cpp

struct TrtEngineConfig {
    std::string plan_path;              // path to serialized .plan file
    bool        use_pinned_host = true; // cudaMallocHost for IO staging
    int         device_id       = 0;    // which CUDA device to bind
    bool        use_cuda_graph  = true; // capture infer sequence into a CUDA
                                         // graph at init. Set false to force
                                         // the explicit per-call submission
                                         // path (debug / unsupported plugins).
};

[[nodiscard]] core::Result<std::unique_ptr<IInferenceEngine>>
make_trt_engine(const TrtEngineConfig& cfg);

}  // namespace rt_deploy::inference
