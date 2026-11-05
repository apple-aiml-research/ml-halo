//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// StateMachine — per-tick state controller for the real-robot lifecycle.
//
// Parity with deploy/g1/deploy_real.py:G1RealController.run_real.
//
// States (see math::ControlState):
//   INIT              — hold damping; wait for START button
//   MOVING_TO_DEFAULT — 2 s cosine ramp from current qpos to init_joint_pos
//                       with kp * 0.5 (same safety halving as the Python ramp)
//   DEFAULT_POS       — hold init_joint_pos with full gains; wait for A
//   CONTROL           — publish the caller-supplied policy command
//                       B rising-edge → flags().standup_requested
//                       A rising-edge → flags().motion_replay_requested
//   STOPPING          — damping-only command; loop should exit
//
// Button edge detection is done here (once per tick) so the ControlLoop
// doesn't need to track previous remote state. SELECT is observed by the
// SafetyMonitor, not here — keeping the two concerns separate.

#include "rt_deploy/core/clock.hpp"
#include "rt_deploy/g1/config.hpp"
#include "rt_deploy/math/types.hpp"
#include "rt_deploy/robot/i_robot_backend.hpp"

#include <Eigen/Core>
#include <cmath>
#include <cstdint>
#include <optional>

namespace rt_deploy::g1 {

struct StateMachineConfig {
    float         move_duration_s = 2.0F;  // Python deploy_real.py move_duration
    float         soft_start_kp_scale = 0.5F;
    float         damping_kd          = 8.0F;
    // Initial state — defaults to INIT (operator-gated flow on real hardware).
    // Sim apps that don't have a remote should set this to kControl so the
    // policy runs immediately without synthesized button presses.
    math::ControlState initial_state  = math::ControlState::kInit;
};

struct StateMachineFlags {
    bool standup_requested      = false;
    bool motion_replay_requested = false;
};

class StateMachine {
public:
    using JointVec = Eigen::Matrix<float, static_cast<Eigen::Index>(kDof), 1>;

    StateMachine(const Config&             cfg,
                  const StateMachineConfig& sm_cfg,
                  JointVec                  init_joint_pos) noexcept
        : cfg_{cfg}, sm_cfg_{sm_cfg}, init_joint_pos_{init_joint_pos},
          state_{sm_cfg.initial_state} {}

    // Hot-path update. Fills `cmd_out` with the command to publish and returns
    // whether the loop should continue.
    //   policy_cmd: the freshly-decoded command. Inference runs every CONTROL
    //               tick, so this is normally set in CONTROL; nullopt is a
    //               defensive hold (cmd_out keeps the value the caller passed).
    //   now_ns:     monotonic timestamp, used for the move-to-default ramp.
    bool update(const math::RobotState<kDof>&         state,
                 const robot::RemoteState&             remote,
                 std::optional<math::MotorCommand<kDof>> policy_cmd,
                 std::int64_t                          now_ns,
                 math::MotorCommand<kDof>&             cmd_out) noexcept {
        // Detect button rising edges once per tick (one small std::array copy).
        const bool start_rising = remote.button[rt_deploy::math::key_map::kStart] != 0
                                  && prev_remote_.button[rt_deploy::math::key_map::kStart] == 0;
        const bool a_rising = remote.button[rt_deploy::math::key_map::kA] != 0
                              && prev_remote_.button[rt_deploy::math::key_map::kA] == 0;
        const bool b_rising = remote.button[rt_deploy::math::key_map::kB] != 0
                              && prev_remote_.button[rt_deploy::math::key_map::kB] == 0;
        prev_remote_ = remote;

        switch (state_) {
            case math::ControlState::kInit: {
                // Damp while we wait for the operator.
                cmd_out.set_damping(sm_cfg_.damping_kd);
                if (start_rising) {
                    start_move_to_default(state, now_ns);
                }
                return true;
            }
            case math::ControlState::kMovingToDefault: {
                tick_move_to_default(now_ns, cmd_out);
                return true;
            }
            case math::ControlState::kDefaultPos: {
                hold_default(cmd_out);
                if (a_rising) {
                    state_ = math::ControlState::kControl;
                }
                return true;
            }
            case math::ControlState::kControl: {
                if (policy_cmd.has_value()) {
                    cmd_out = *policy_cmd;  // fresh policy output (every CONTROL tick)
                }
                // else: defensive hold — caller passed the last command as cmd_out.
                flags_.standup_requested       = flags_.standup_requested       || b_rising;
                flags_.motion_replay_requested = flags_.motion_replay_requested || a_rising;
                return true;
            }
            case math::ControlState::kStopping: {
                cmd_out.set_damping(sm_cfg_.damping_kd);
                return false;
            }
            default:
                return false;
        }
    }

    // External E-stop (e.g. safety-monitor latch) moves the state machine to
    // STOPPING so the outer loop can observe it through state().
    void request_stop() noexcept { state_ = math::ControlState::kStopping; }

    [[nodiscard]] math::ControlState     state() const noexcept { return state_; }
    [[nodiscard]] const StateMachineFlags& flags() const noexcept { return flags_; }
    void clear_flags() noexcept { flags_ = {}; }

private:
    void start_move_to_default(const math::RobotState<kDof>& state, std::int64_t now_ns) noexcept {
        start_pos_       = state.joint_positions;
        move_start_ns_   = now_ns;
        state_           = math::ControlState::kMovingToDefault;
    }

    void tick_move_to_default(std::int64_t now_ns, math::MotorCommand<kDof>& cmd) noexcept {
        const float elapsed = static_cast<float>(now_ns - move_start_ns_) * 1.0e-9F;
        const float dur     = sm_cfg_.move_duration_s > 0.0F ? sm_cfg_.move_duration_s : 1.0F;
        const float alpha   = elapsed / dur;
        if (alpha >= 1.0F) {
            // Ramp finished: transition to DEFAULT_POS and emit the full-gain
            // hold so the published command matches the new state on this tick.
            state_ = math::ControlState::kDefaultPos;
            hold_default(cmd);
            return;
        }
        const float smooth_alpha = 0.5F * (1.0F - std::cos(alpha * static_cast<float>(M_PI)));
        cmd.positions = start_pos_ + smooth_alpha * (init_joint_pos_ - start_pos_);
        cmd.velocities.setZero();
        cmd.kp      = cfg_.joint_kps * sm_cfg_.soft_start_kp_scale;
        cmd.kd      = cfg_.joint_kds;
        cmd.torques.setZero();
        cmd.modes.fill(0);
    }

    void hold_default(math::MotorCommand<kDof>& cmd) noexcept {
        cmd.positions = init_joint_pos_;
        cmd.velocities.setZero();
        cmd.kp = cfg_.joint_kps;
        cmd.kd = cfg_.joint_kds;
        cmd.torques.setZero();
        cmd.modes.fill(0);
    }

    Config                cfg_;
    StateMachineConfig    sm_cfg_;
    JointVec              init_joint_pos_;
    math::ControlState    state_{math::ControlState::kInit};  // overwritten by ctor from sm_cfg.initial_state
    JointVec              start_pos_{JointVec::Zero()};
    std::int64_t          move_start_ns_{0};
    robot::RemoteState    prev_remote_{};
    StateMachineFlags     flags_{};
};

}  // namespace rt_deploy::g1
