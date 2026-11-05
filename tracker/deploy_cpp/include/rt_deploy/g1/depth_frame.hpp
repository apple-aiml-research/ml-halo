//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// DepthFrame — fixed-size, trivially-copyable metric-depth frame for the
// depth-student `student_depth` observation group.
//
// Row-major HxW depth in METERS, RAW (pre-encode): the ObservationBuilder image
// group applies clip([clip_min,clip_max]) -> log1p when it consumes this. Fixed
// 48x84 at compile time so it fits a core::TripleBuffer<DepthFrame> slot (the
// wait-free hand-off from the non-RT depth producer — sim render / real RealSense
// — to the RT control loop, which only ever does a wait-free read).
//
// The producer fills `data` row-major (index = r*kCols + c), in meters, bumps
// `seq`, and stamps `stamp_ns` immediately before publish(). Consumers never
// mutate it.

#include <array>
#include <cstddef>
#include <cstdint>

namespace rt_deploy::g1 {

struct DepthFrame {
    static constexpr int         kRows  = 48;
    static constexpr int         kCols  = 84;
    static constexpr std::size_t kCount =
        static_cast<std::size_t>(kRows) * static_cast<std::size_t>(kCols);  // 4032

    std::array<float, kCount> data{};   // row-major (r*kCols + c), meters, raw
    std::int64_t              seq = 0;  // monotonic producer frame id (staleness)
    // core::MonotonicClock stamp taken by the producer just before publish().
    // Lets a consumer report the frame's AGE per tick instead of only "how many
    // ticks since seq moved" — seq staleness alone cannot separate a slow
    // producer from a dead one, and cannot be compared against the control
    // period at all.
    //
    // Same clock domain as the control loop's tick timestamp, deliberately: the
    // consumer subtracts the two directly, so a producer using CLOCK_REALTIME or
    // the camera's own hardware clock here would silently produce nonsense.
    //
    // 0 == "not stamped". Consumers MUST treat 0 as unknown rather than as an
    // epoch, because an unstamped frame would otherwise read as hours old.
    std::int64_t              stamp_ns = 0;
};

}  // namespace rt_deploy::g1
