//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// ActionDecoder — torque-aware clip + default-angle offset + joint remap.
//
// Parity with deploy/utils/core/base_controller.py:compute_motor_targets
// (all 29 G1 joints position-controlled). The decoder converts a raw policy
// action (training order, length 29) to a MotorCommand (deployment order)
// ready to send over DDS.
//
// All operations are noexcept and hot-path safe. Configuration (kp, kd,
// default_angles, action_scale, max_torque) is captured once at construction
// from g1::Config.

#include "rt_deploy/g1/config.hpp"
#include "rt_deploy/math/types.hpp"
#include "rt_deploy/policy/joint_remap.hpp"

#include <Eigen/Core>

namespace rt_deploy::g1 {

class ActionDecoder {
public:
    static constexpr std::size_t kN = kDof;
    using JointVec                  = Eigen::Matrix<float, static_cast<Eigen::Index>(kN), 1>;

    ActionDecoder(const Config& cfg, const policy::JointRemap<kN>& remap) noexcept
        : default_angles_{cfg.default_angles},
          joint_kps_{cfg.joint_kps},
          joint_kds_{cfg.joint_kds},
          action_scale_{cfg.action_scale},
          max_torque_{cfg.max_torque},
          remap_{remap} {}

    // Decode a raw policy action into a MotorCommand. Also writes the
    // post-clip action (in training order) into `clipped_action_train` for
    // the observation builder's action history buffer.
    //
    // Inputs:
    //   raw_action_train  : policy output, length 29, training order.
    //                       Already globally clipped to [-clip_action, +clip_action]
    //                       upstream; this fn applies the torque-aware clamp.
    //   current_pos_deploy: current joint positions, deployment order (from sensor).
    //   current_vel_deploy: current joint velocities, deployment order.
    //
    // Outputs:
    //   cmd                    : filled MotorCommand (positions, kp/kd, etc.)
    //   clipped_action_train   : action after torque-aware clamp, training order.
    template <typename ActIn, typename PosIn, typename VelIn, typename ActOut>
    void decode(const ActIn&            raw_action_train,
                const PosIn&            current_pos_deploy,
                const VelIn&            current_vel_deploy,
                math::MotorCommand<kN>& cmd,
                ActOut&                 clipped_action_train) const noexcept {
        // 1. Remap training → deployment so we can compute per-joint clamps
        //    against deployment-ordered state / gains / defaults.
        JointVec action_deploy;
        for (std::size_t i = 0; i < kN; ++i) {
            action_deploy[static_cast<Eigen::Index>(i)]
                = raw_action_train[static_cast<Eigen::Index>(remap_.train_to_deploy[i])];
        }

        // 2. Torque-aware clamp (position control: all 29 joints for G1).
        JointVec clipped_deploy = action_deploy;
        for (std::size_t i = 0; i < kN; ++i) {
            const Eigen::Index ei = static_cast<Eigen::Index>(i);
            const float        kp = joint_kps_[ei];
            if (kp <= 0.0F) {
                continue;
            }
            const float kd  = joint_kds_[ei];
            const float tau = max_torque_[ei];
            const float v   = current_vel_deploy[ei];
            const float q   = current_pos_deploy[ei];

            // PD torque = kp * (q_target - q) - kd * v must stay in [-tau, tau],
            // so q_target - q is bounded by (+-tau + kd * v) / kp.
            const float damp          = kd * v;
            const float pos_range_min = (-tau + damp) / kp;
            const float pos_range_max = (tau + damp) / kp;
            const float pos_min       = q + pos_range_min;
            const float pos_max       = q + pos_range_max;

            const float scale = action_scale_[ei];
            if (scale == 0.0F) {
                continue;
            }
            const float d          = default_angles_[ei];
            const float action_min = (pos_min - d) / scale;
            const float action_max = (pos_max - d) / scale;
            const float a          = action_deploy[ei];
            clipped_deploy[ei]     = std::min(action_max, std::max(action_min, a));
        }

        // 3. Target positions: clipped_action * scale + default_angles (deploy order).
        cmd.positions = clipped_deploy.cwiseProduct(action_scale_) + default_angles_;
        cmd.velocities.setZero();
        cmd.kp      = joint_kps_;
        cmd.kd      = joint_kds_;
        cmd.torques.setZero();
        cmd.modes.fill(0);

        // 4. Record the clipped action in training order for the history buffer.
        //    clipped_action_train[i] = clipped_deploy[deploy_to_train[i]]
        for (std::size_t i = 0; i < kN; ++i) {
            clipped_action_train[static_cast<Eigen::Index>(i)]
                = clipped_deploy[static_cast<Eigen::Index>(remap_.deploy_to_train[i])];
        }
    }

private:
    JointVec                default_angles_;
    JointVec                joint_kps_;
    JointVec                joint_kds_;
    JointVec                action_scale_;
    JointVec                max_torque_;
    policy::JointRemap<kN>  remap_{};
};

}  // namespace rt_deploy::g1
