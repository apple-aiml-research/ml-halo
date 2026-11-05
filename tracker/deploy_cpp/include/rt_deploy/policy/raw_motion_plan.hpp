//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// RawMotionPlan — the payload a transport (e.g. the ZMQ tracker nodes in
// system/cpp, or an in-process driver) pushes into the control loop's motion
// inbox.
//
// The transport copies the wire rows verbatim into `data` and fills in
// `n_rows`, `input_fps` and `flags`. No resampling, no remap, no velocity
// computation — those happen on the control thread when the plan is consumed
// (~100 µs for a 5 s / 150-frame plan).
//
// Row layout (matches the Python sender and MotionLoader's CSV convention):
//   [0:3]   base xyz position (m)
//   [3:7]   base orientation quaternion in xyzw order
//   [7:36]  29 joint positions in deployment order (radians)
//
// Storage is a fixed-size std::array so the type is trivially copyable, fits
// in a TripleBuffer slot, and never allocates on the hot path. kMaxRows is
// sized for ~60 s at 30 fps; oversized payloads are rejected at the transport
// boundary.

#include <array>
#include <cstdint>

namespace rt_deploy::policy {

// Per-chunk control flags carried in the wire header (and on the in-process
// RawMotionPlan struct). Unset bits use the documented default.
//
//   kFlagAlignYaw         : legacy "align on install" bit, set by default for
//                           wire compatibility. Every install is rebased by
//                           align_motion_to_anchor regardless; see
//                           kFlagRobotAnchor for the anchor choice.
//
//   kFlagAutoStartOffset  : when set, the tracker slices the resampled chunk
//                           by its own current `inference_tick` (= count of
//                           inferences run since the last reset, equivalent
//                           to Python `controller.counter`). Use case: RTC
//                           streaming where the chunk is conditioned on a
//                           trim that reset the tick to 0, so inference_tick
//                           captures exactly the frames advanced past during
//                           inference latency. When the flag is OFF the
//                           chunk plays from frame 0 — use this for fresh
//                           plans that aren't continuations of a prior trim.
inline constexpr std::uint32_t kFlagAlignYaw        = 1U << 0U;
inline constexpr std::uint32_t kFlagAutoStartOffset = 1U << 1U;
//
//   kFlagRobotAnchor      : install this chunk with align_anchor="robot" — i.e.
//                           rebase the chunk's post-slice frame 0 onto the
//                           robot's CURRENT odom pose (or IMU fallback) even
//                           when a previous reference exists. Default OFF: when
//                           a previous reference is loaded, the chunk is
//                           rebased onto its splice pose at this tick
//                           (align_anchor="reference"); when none is loaded,
//                           the robot pose is used as a cold-start fallback.
//                           Use case: stationary "hold" plans that re-ground
//                           the reference to the robot's actual position so
//                           accumulated reference drift doesn't appear in the
//                           obs. FLAG_ROBOT_ANCHOR in system/'s Python
//                           helpers (zmq_helper.py).
inline constexpr std::uint32_t kFlagRobotAnchor     = 1U << 2U;

// Default flag set for in-process producers (e.g. a trim round-trip).
inline constexpr std::uint32_t kFlagsDefault = kFlagAlignYaw;

struct RawMotionPlan {
    static constexpr std::int32_t kMaxRows = 1800;  // ~60 s @ 30 fps
    static constexpr std::int32_t kCols    = 36;    // base xyz(3) + quat_xyzw(4) + joints(29)
    static constexpr std::size_t  kCapacity
        = static_cast<std::size_t>(kMaxRows) * static_cast<std::size_t>(kCols);

    std::array<double, kCapacity> data{};
    std::int32_t                  n_rows    = 0;
    std::int32_t                  input_fps = 0;
    // Per-chunk behavior flags. See kFlag* above.
    std::uint32_t                 flags     = kFlagsDefault;
};

}  // namespace rt_deploy::policy
