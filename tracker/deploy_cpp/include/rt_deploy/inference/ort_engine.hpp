//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// OrtEngine — ONNX Runtime backend for IInferenceEngine.
//
// Primary use: dev-host builds where TRT/CUDA aren't available. Also the
// FP32 reference for comparing against the Python pipeline.
//
// Hot-path infer_multi() copies each input into a preallocated Ort::Value, calls
// session.Run(), and memcpys the output tensor data into the caller's span.
// No heap traffic on the hot path (session-owned tensors, preallocated).

#include "rt_deploy/core/expected.hpp"
#include "rt_deploy/inference/i_inference_engine.hpp"

#include <memory>
#include <string>

namespace rt_deploy::inference {

enum class OrtProvider : std::uint8_t {
    kCpu  = 0,
    kCuda = 1,
};

struct OrtEngineConfig {
    std::string onnx_path;
    OrtProvider provider        = OrtProvider::kCpu;
    int         intra_op_threads = 2;
    int         inter_op_threads = 1;
    int         device_id        = 0;  // CUDA device when provider == kCuda
};

[[nodiscard]] core::Result<std::unique_ptr<IInferenceEngine>>
make_ort_engine(const OrtEngineConfig& cfg);

}  // namespace rt_deploy::inference
