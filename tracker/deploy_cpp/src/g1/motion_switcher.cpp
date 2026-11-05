//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#include "rt_deploy/g1/motion_switcher.hpp"

#include "rt_deploy/policy/motion_loader.hpp"

#include <cmath>

namespace rt_deploy::g1 {

core::Result<policy::MotionData>
make_standing_motion(const Config&                   cfg,
                      const policy::JointRemap<kDof>& remap,
                      std::int32_t                    min_output_frames) {
    using ResT = core::Result<policy::MotionData>;

    policy::MotionLoaderConfig mcfg;
    mcfg.csv_path                = "";  // standing-pose path
    mcfg.input_fps               = 30;  // unused on the standing path
    mcfg.output_fps              = static_cast<int>(std::round(1.0F / cfg.control_dt));
    mcfg.joint_remap             = nullptr;  // standing path skips remap; we apply below
    mcfg.standing_joint_pos      = cfg.standing_joint_pos;
    mcfg.standing_base_pos       = cfg.standing_base_pos;
    mcfg.standing_base_quat_wxyz = cfg.standing_base_quat_wxyz;
    mcfg.min_output_frames       = min_output_frames;

    auto r = policy::load_motion(mcfg);
    if (!r) {
        return ResT::Error{r.error()};
    }
    policy::MotionData motion = std::move(r.value());
    // load_motion's standing path keeps joints in deployment order. The
    // observation builder reads dof_poss[r, i] with i indexing training-order
    // slots, so apply the remap here for parity with the CSV path.
    policy::apply_joint_remap(motion, remap);
    return motion;
}

policy::MotionData* MotionSwitcher::switch_to_original(
    const Eigen::Vector4f& robot_quat_wxyz) noexcept {
    if (original_ != nullptr) {
        policy::align_motion_to_yaw(*original_, robot_quat_wxyz);
        active_ = original_;
    } else {
        // No CSV motion (the system/cpp tracker nodes, which only receive
        // runtime-pushed plans): fall back to the standing motion, yaw-aligned
        // to the robot. Without this the CONTROL-entry switch_to_original() is a
        // no-op, the standing reference keeps its authored yaw = 0 while the
        // robot faces its real heading, and the policy snaps the body to close
        // that yaw delta on the first CONTROL tick.
        policy::align_motion_to_yaw(standing_, robot_quat_wxyz);
        active_ = &standing_;
    }
    return active_;
}

policy::MotionData* MotionSwitcher::switch_to_standing(
    const Eigen::Vector4f& robot_quat_wxyz) noexcept {
    policy::align_motion_to_yaw(standing_, robot_quat_wxyz);
    active_ = &standing_;
    return active_;
}

}  // namespace rt_deploy::g1
