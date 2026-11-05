//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// RawDepthSource — the `kRaw` depth source: the D435i's own on-chip stereo
// depth, Z16 -> metres -> 84x48, straight into the g1 control loop's inbox.
//
//   [rs-depth thread @ rate_hz]
//    wait_for_frames() -> Z16 * depth_scale -> invalid(0) -> clip_max
//                      -> clamp[clip_min, clip_max] -> downsample 84x48
//                      --TripleBuffer<g1::DepthFrame>--> [g1 control loop]
//
// This is the switchable fallback behind `depth_source: raw`, and the A/B
// reference the cleaned path is judged against. It is deliberately the SAME
// sink, the SAME g1::DepthFrame contract and the SAME clamp/invalid/downsample
// rules (vision::depth_ops) as DepthPipeline, so swapping between them changes
// only the depth estimator — never the observation semantics. The control loop
// cannot tell which one is running.
//
// It needs no GPU, no TensorRT plan and no CUDA: it is the one path that still
// works when the engine will not build, which is exactly what a fallback is
// for. The projector follows `config.emitter` — note the D435i's on-chip block
// matcher genuinely BENEFITS from the dot pattern, the opposite of the learned
// matcher, so a fair raw baseline usually wants `emitter: true` even though the
// cleaned path requires it off.
//
// One thread, one blocking wait_for_frames(), no allocation after start().

#include "rt_deploy/core/expected.hpp"
#include "rt_deploy/core/triple_buffer.hpp"
#include "rt_deploy/g1/depth_frame.hpp"
#include "rt_deploy/vision/camera_intrinsics.hpp"
#include "rt_deploy/vision/config.hpp"
#include "rt_deploy/vision/depth_source.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <thread>

namespace rt_deploy::vision {

class RawDepthSource final : public IDepthSource {
public:
    // `inbox` is the control loop's depth sink and must outlive the source.
    RawDepthSource(VisionConfig config, core::TripleBuffer<g1::DepthFrame>& inbox);
    ~RawDepthSource() override;

    // Open the device, read the depth-sensor scale + intrinsics, and spawn the
    // `rs-depth` thread. Off hot path.
    [[nodiscard]] core::Result<void, std::string> start() override;

    void stop() noexcept override;

    [[nodiscard]] DepthStats       stats() const noexcept override;
    [[nodiscard]] StereoIntrinsics intrinsics() const noexcept override;
    [[nodiscard]] std::string_view name() const noexcept override { return "raw"; }

private:
    // librealsense2 headers stay confined to raw_depth_source.cpp.
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rt_deploy::vision
