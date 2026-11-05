//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// Wire/handoff buffers for the depth pipeline. Both are fixed-size, trivially
// copyable PODs so they can live as `core::TripleBuffer<T>` slots — no heap, no
// constructor work on the hot path, mirroring policy::RawMotionPlan.
//
//   StereoFrame  : capture thread → inference thread. The raw IR1/IR2 pair.
//   DepthFrame   : the stereo engine's output scratch. Fused float32 metric
//                  depth (metres) — never Z16, never millimetres. DepthPipeline
//                  copies it into the control loop's g1::DepthFrame.
//
// Capacity is sized for the D435i's largest IR mode (1280×800). The pipeline
// runs at 424×240 (see configs/depth_d435i.yaml), so only the leading
// width*height samples of each array are populated; `width`/`height` carry the
// live extent. Allocating for the max keeps the type fixed-size and the slot
// swap a pure pointer flip regardless of resolution.

#include "rt_deploy/vision/camera_intrinsics.hpp"

#include <array>
#include <cstdint>

namespace rt_deploy::vision {

// D435i IR streams top out at 1280×800. ~1.0 MP — one byte per IR pixel.
inline constexpr std::int32_t kMaxWidth  = 1280;
inline constexpr std::int32_t kMaxHeight = 800;
inline constexpr std::size_t  kMaxPixels =
    static_cast<std::size_t>(kMaxWidth) * static_cast<std::size_t>(kMaxHeight);

// The published depth is a small, fixed policy observation (e.g. 84×48). Cap
// generously so the output dims are configurable without resizing the type.
inline constexpr std::int32_t kMaxOutWidth  = 256;
inline constexpr std::int32_t kMaxOutHeight = 256;
inline constexpr std::size_t  kMaxOutPixels =
    static_cast<std::size_t>(kMaxOutWidth) * static_cast<std::size_t>(kMaxOutHeight);

// Raw rectified IR pair, emitter OFF. IR1 = left, IR2 = right (D4xx ordering).
// One channel, 8-bit, row-major, tightly packed at width*height.
struct StereoFrame {
    std::array<std::uint8_t, kMaxPixels> left{};   // IR1
    std::array<std::uint8_t, kMaxPixels> right{};  // IR2

    std::int32_t  width    = 0;
    std::int32_t  height   = 0;
    std::uint64_t seq      = 0;  // monotonically increasing capture index
    std::int64_t  stamp_ns = 0;  // CLOCK_MONOTONIC at capture (latency probe)

    [[nodiscard]] std::size_t pixels() const noexcept {
        return static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    }
};

// Fused metric depth, float32 metres, clamped to [clip_min, clip_max] (invalid
// / occluded → clip_max). Delivered at the small output resolution the policy
// consumer expects (e.g. 84×48). `intr` travels with the frame for any consumer
// that wants the source intrinsics.
struct DepthFrame {
    std::array<float, kMaxOutPixels> depth_m{};

    StereoIntrinsics intr{};
    std::int32_t     width     = 0;    // output width (e.g. 84)
    std::int32_t     height    = 0;    // output height (e.g. 48)
    std::uint64_t    seq       = 0;
    std::int64_t     stamp_ns  = 0;     // copied from the source StereoFrame
    float            infer_ms  = 0.0F;  // last inference wall time (diagnostics)

    [[nodiscard]] std::size_t pixels() const noexcept {
        return static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    }
};

}  // namespace rt_deploy::vision
