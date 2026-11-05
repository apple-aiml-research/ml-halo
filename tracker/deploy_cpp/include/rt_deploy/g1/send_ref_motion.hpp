//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// send_ref_motion — install a streaming motion chunk into the active reference.
//
// Mirrors deploy/g1/g1_controller.py:G1Controller.send_ref_motion. The
// pipeline is resample → joint remap → slice → derivatives → anchor rebase.
// Kept out of ControlLoop::run so the swap path can run without a control
// loop.
//
// Allocation-free, noexcept, hot-path safe. The caller is responsible for:
//   - holding the only writer reference to `scratch` (the loop's
//     processed_motion_scratch_), and
//   - on success: calling observation_builder.set_motion(&scratch) and the
//     loop's tick reset + snapshot republish. The observation history buffers
//     are INTENTIONALLY left untouched across the swap — Python parity with
//     G1Controller.send_ref_motion, which never resets history.

#include "rt_deploy/g1/config.hpp"
#include "rt_deploy/math/types.hpp"
#include "rt_deploy/policy/joint_remap.hpp"
#include "rt_deploy/policy/motion_loader.hpp"
#include "rt_deploy/policy/raw_motion_plan.hpp"

namespace rt_deploy::g1 {

// Per-swap diagnostics — distance of the chunk's post-slice frame 0 from the
// robot's current state at install time. Captured by the loop into ring
// vectors and reported in print_swap_summary().
struct SwapMetrics {
    float joint_mean = 0.0F;  // mean |q_train - chunk_train|
    float joint_l2   = 0.0F;  // ||q_train - chunk_train||
    float base_xy    = 0.0F;  // ||chunk_xy - robot_xy||
    float base_z     = 0.0F;  // |chunk_z - robot_z|
};

// Why a swap was rejected.
enum class SwapRejection : std::uint8_t {
    kNone,
    kResampleFailed,        // scratch capacity too small or invalid header
    kStartOffsetExceedsLen, // auto-slice inference_tick >= chunk frame_count
    kInvalidHeader,         // n_rows outside 1..RawMotionPlan::kMaxRows
    kNonFinite,             // a NaN / inf value in the plan rows
};

struct SwapResult {
    bool          installed = false;
    SwapRejection rejection = SwapRejection::kNone;
    std::int32_t  raw_n_rows   = 0;  // for diagnostic logging on rejection
    std::int32_t  raw_input_fps = 0; // ditto
    SwapMetrics   metrics{};
};

// Install `raw` into `scratch`. On success the caller swaps the obs builder's
// motion pointer to `&scratch`, resets the inference tick, and republishes the
// active-plan snapshot — but does NOT reset the obs builder's history buffers
// (Python parity; the buffers are robot-state observations and remain valid
// across a motion change). Returns installed == false (with `rejection` set)
// when the plan is rejected; in that case `scratch` is NOT modified. That matters: after the
// first install the obs builder tracks `scratch` itself, so it is the live
// reference.
//
// `previous` is the OLD reference (observation_builder.motion()) read BEFORE
// any state in `scratch` is overwritten. May be nullptr or empty.
[[nodiscard]] SwapResult send_ref_motion(
    const policy::RawMotionPlan&         raw,
    const policy::MotionData*            previous,
    const math::RobotState<kDof>&        state,
    std::int32_t                         inference_tick,
    policy::MotionData&                  scratch,
    const policy::JointRemap<kDof>&      remap) noexcept;

}  // namespace rt_deploy::g1
