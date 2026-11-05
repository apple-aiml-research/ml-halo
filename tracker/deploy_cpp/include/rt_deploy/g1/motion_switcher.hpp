//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// MotionSwitcher — runtime swap between the operator-supplied reference motion
// and a synthesised standing-pose motion, with on-demand yaw alignment.
//
// Parity with deploy/g1/deploy_real.py:do_standup + replay_original_motion.
// In the Python flow, pressing B during CONTROL calls send_ref_motion with a
// single standing frame (yaw-aligned to the robot), and pressing A re-sends
// the originally-loaded motion from frame 0. This class provides the C++
// equivalent: a stateful holder that owns a standing MotionData and points at
// either it or the caller-supplied original. Each switch yaw-aligns the
// chosen target to the robot's current orientation so the policy doesn't see
// a discontinuity in motion_anchor_ori on the first tick after the swap.
//
// Threading: single-threaded. The switcher and the consumer (the observation
// builder) both live on the control thread; switch_*() is called from the
// loop body's flag handler immediately before the next observation build.
//
// Allocations: zero on the hot path. The standing MotionData is built once at
// construction (off the loop). yaw alignment writes through the existing
// matrices in place (Eigen same-size assignment does not realloc). The
// original motion is held by non-owning pointer so the loop's existing
// ownership model is unchanged.

#include "rt_deploy/core/expected.hpp"
#include "rt_deploy/g1/config.hpp"
#include "rt_deploy/policy/joint_remap.hpp"
#include "rt_deploy/policy/motion_loader.hpp"

#include <Eigen/Core>
#include <cstdint>
#include <utility>

namespace rt_deploy::g1 {

// Build a static standing-pose MotionData from g1::Config (training order).
//
// Uses load_motion's standing-pose code path (csv_path = empty), then applies
// the deployment→training joint remap so the matrix layout matches what the
// observation builder expects. min_output_frames pads the motion if the value
// is greater than the default (10 s × output_fps).
//
// Off the hot path. Returns an error if load_motion fails (e.g. invalid fps).
[[nodiscard]] core::Result<policy::MotionData>
make_standing_motion(const Config&                   cfg,
                      const policy::JointRemap<kDof>& remap,
                      std::int32_t                    min_output_frames = 0);

class MotionSwitcher {
public:
    // `original` is the user-supplied motion (CSV-loaded). The switcher
    // borrows it; the caller retains ownership and must keep it alive for the
    // switcher's lifetime. `original` may be null — switch_to_original() is
    // then a no-op and active() returns the standing motion only.
    MotionSwitcher(policy::MotionData* original, policy::MotionData standing) noexcept
        : original_{original}, standing_{std::move(standing)},
          active_{original_ != nullptr ? original_ : &standing_} {}

    // Yaw-align the original motion to `robot_quat_wxyz` and select it as
    // active. Returns the new active pointer (or the standing motion if
    // `original` was null at construction). Idempotent within < 1 mrad.
    policy::MotionData* switch_to_original(const Eigen::Vector4f& robot_quat_wxyz) noexcept;

    // Yaw-align the standing motion to `robot_quat_wxyz` and select it as
    // active. Returns the new active pointer.
    policy::MotionData* switch_to_standing(const Eigen::Vector4f& robot_quat_wxyz) noexcept;

    [[nodiscard]] policy::MotionData* active() const noexcept { return active_; }
    [[nodiscard]] policy::MotionData* original() const noexcept { return original_; }
    [[nodiscard]] policy::MotionData* standing() noexcept { return &standing_; }

private:
    policy::MotionData* original_{nullptr};
    policy::MotionData  standing_{};
    policy::MotionData* active_{nullptr};
};

}  // namespace rt_deploy::g1
