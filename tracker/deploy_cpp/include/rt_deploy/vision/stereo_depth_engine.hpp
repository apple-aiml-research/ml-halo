//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// IStereoDepthEngine — the on-device fusion stage: IR pair → fused metric depth.
//
// One call runs the whole GPU path with a single round trip to host memory:
//   ir_to_chw (×2)  →  TensorRT enqueueV3 (LiteAnyStereo)  →  disp_to_depth
// The disparity never touches the CPU; only the IR bytes go in and the float32
// metric depth comes out.
//
// The concrete backend is TensorRT (make_stereo_depth_engine). It reuses the
// RAII / CUDA-graph idioms of inference::TrtEngine but differs in three ways
// the policy engine doesn't need:
//   * A fixed IO contract: two image inputs (left, right) and one disparity
//     output.
//   * Unified-memory awareness — on an integrated Orin GPU all IO buffers are
//     mapped (cudaHostAllocMapped) so there is NO cudaMemcpy across the bus;
//     on a discrete dev GPU it falls back to pinned-host + cudaMalloc + async
//     H2D/D2H, just like TrtEngine.
//   * It owns the pre/post-process kernels, so its "output" is metric depth,
//     not raw disparity.
//
// Construction does all allocation; infer() is noexcept, allocation-free, and
// runs off the capture thread.

#include "rt_deploy/core/expected.hpp"
#include "rt_deploy/inference/i_inference_engine.hpp"
#include "rt_deploy/vision/depth_ops.hpp"
#include "rt_deploy/vision/stereo_frame.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace rt_deploy::vision {

struct StereoDepthEngineConfig {
    std::string plan_path;             // serialized TensorRT engine (.plan)
    int         device_id      = 0;
    bool        use_cuda_graph = true;  // capture the H2D+infer+D2H sequence

    // Network input resolution = the IR resolution fed to the engine. Must be
    // <= the engine's padded input dims and <= kMaxWidth/kMaxHeight. The engine
    // pads internally to the 32-divisible dims it was built for.
    std::int32_t width  = 424;
    std::int32_t height = 240;

    // Output (policy) resolution — the depth is reduced down to this and
    // published as float32 metres. e.g. 84×48.
    std::int32_t out_width  = 84;
    std::int32_t out_height = 48;

    // Depth post-processing (set from camera intrinsics + config).
    float fx_baseline = 0.0F;  // fx[px] * baseline[m] at the network resolution
    // Always set from VisionConfig by DepthPipeline; these defaults exist only
    // so a partially-filled config is not silently zero. 0.2/2.5 = the depth
    // student's clip, matching vision/config.hpp.
    float clip_min_m  = 0.2F;  // near clip [m]; valid depths below clamp up
    float clip_max_m  = 2.5F;  // far clip [m]; valid above + invalid/occluded map here

    // How each output cell reduces its network-resolution footprint (see
    // vision::Reduce in depth_ops.hpp). Mirror VisionConfig::downsample here —
    // the GPU kernel and the raw-D435i CPU path must use the same mode or the
    // two sources produce differently-biased observations.
    Reduce downsample = Reduce::kMin;
};

class IStereoDepthEngine {
public:
    virtual ~IStereoDepthEngine() = default;

    IStereoDepthEngine()                                     = default;
    IStereoDepthEngine(const IStereoDepthEngine&)            = delete;
    IStereoDepthEngine& operator=(const IStereoDepthEngine&) = delete;
    IStereoDepthEngine(IStereoDepthEngine&&)                 = delete;
    IStereoDepthEngine& operator=(IStereoDepthEngine&&)      = delete;

    // Run the full IR→depth path. `in` supplies left/right + the network
    // resolution; `out.depth_m` (float metres, reduced to the output res per
    // StereoDepthEngineConfig::downsample), `out.width`, `out.height` are
    // filled. The caller owns the rest of `out` (intr, seq, stamp, infer_ms).
    // noexcept, hot-path safe.
    [[nodiscard]] virtual inference::InferenceStatus infer(const StereoFrame& in,
                                                           DepthFrame& out) noexcept = 0;

    // Update fx*baseline + clip range once camera intrinsics are known. Off hot
    // path; call before the first infer() when CUDA graphs are enabled.
    virtual void set_output_params(float fx_baseline, float clip_min_m,
                                   float clip_max_m) noexcept = 0;

    [[nodiscard]] virtual std::int32_t     pad_width() const noexcept  = 0;
    [[nodiscard]] virtual std::int32_t     pad_height() const noexcept = 0;
    [[nodiscard]] virtual bool             zero_copy() const noexcept  = 0;  // integrated GPU?
    [[nodiscard]] virtual std::string_view backend_name() const noexcept = 0;
};

[[nodiscard]] core::Result<std::unique_ptr<IStereoDepthEngine>>
make_stereo_depth_engine(const StereoDepthEngineConfig& cfg);

}  // namespace rt_deploy::vision
