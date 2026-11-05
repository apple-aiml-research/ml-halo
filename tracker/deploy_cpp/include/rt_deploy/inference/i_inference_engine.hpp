//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// IInferenceEngine — thin, allocation-free abstraction for policy inference.
//
// The control loop owns one IInferenceEngine per policy and invokes
// infer_multi() on the hot path. The implementation must not allocate, log, or throw. All
// buffers and streams are set up at construction time.
//
// Two engines:
//   - TrtEngine (TensorRT, enqueueV3 API) — the RT target on Jetson Orin.
//   - OrtEngine (ONNX Runtime, CPU or CUDA) — dev-host fallback.
//
// Both are selected via cmake options RT_DEPLOY_WITH_TRT / RT_DEPLOY_WITH_ORT.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace rt_deploy::inference {

// Shape of a single IO tensor. Dynamic dims are not supported; the plan must
// be built with explicit dimensions.
struct TensorShape {
    std::vector<std::int32_t> dims;

    [[nodiscard]] std::size_t elements() const noexcept {
        std::size_t n = 1;
        for (auto d : dims) {
            n *= d > 0 ? static_cast<std::size_t>(d) : 0U;
        }
        return n;
    }
};

enum class InferenceStatus : std::uint8_t {
    kOk               = 0,
    kSizeMismatch     = 1,  // input/output span size doesn't match engine shape
    kRuntimeError     = 2,  // engine reported an error
    kNotReady         = 3,  // async infer still in flight
};

// The engine knows its input/output shapes; the caller owns the memory behind
// the spans. infer_multi() is expected to be noexcept and ≤ 1 control-period
// latency for a correctly-sized policy on the target hardware.
class IInferenceEngine {
public:
    virtual ~IInferenceEngine() = default;

    IInferenceEngine()                                   = default;
    IInferenceEngine(const IInferenceEngine&)            = delete;
    IInferenceEngine& operator=(const IInferenceEngine&) = delete;
    IInferenceEngine(IInferenceEngine&&)                 = delete;
    IInferenceEngine& operator=(IInferenceEngine&&)      = delete;

    // Hot-path-safe synchronous inference. `inputs` holds one span per engine
    // input, in the engine's declared input order (the proprio student has two:
    // `obs` + `policy_temporal`; the depth student three: + `student_depth`).
    // Returns kSizeMismatch if the input count or any span size is wrong.
    [[nodiscard]] virtual InferenceStatus
    infer_multi(std::span<const std::span<const float>> inputs,
                std::span<float>                        output) noexcept = 0;

    // Input introspection — safe to call at any time, not on the hot path.
    [[nodiscard]] virtual std::size_t      input_count() const noexcept                   = 0;
    [[nodiscard]] virtual std::size_t      input_size_at(std::size_t idx) const noexcept  = 0;
    [[nodiscard]] virtual TensorShape      input_shape_at(std::size_t idx) const          = 0;
    [[nodiscard]] virtual std::string_view input_name_at(std::size_t idx) const noexcept  = 0;
    [[nodiscard]] virtual std::string_view backend_name() const noexcept                  = 0;
};

}  // namespace rt_deploy::inference
