//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// SafetyMonitor — centralised guard executed every control tick.
//
// Responsibilities (executed in order; first failure latches E-stop):
//   1) State freshness     — age since last fresh DDS snapshot > `stale_ns`
//   2) NaN / Inf in state  — joint pos/vel, imu quat/gyro/accel
//   3) NaN / Inf in action — the just-decoded MotorCommand
//   4) Joint-position hard limits (config-driven, optional)
//   5) Button E-stop       — SELECT press latches
//
// (Parity note: the Python reference has no torque-limit latch, so neither
// does this monitor — the per-joint torque budget is enforced inside the
// action decoders' torque-aware clamp, not here.)
//
// On latch, the SafetyMonitor replaces the outgoing MotorCommand with a
// damping command (`kp=0, kd=8`). The state machine observes `is_latched()`
// and transitions to STOPPING. The latch is sticky until the controller
// process restarts — there is no software un-latch by design.

#include "rt_deploy/math/types.hpp"
#include "rt_deploy/robot/i_robot_backend.hpp"

#include <Eigen/Core>
#include <array>
#include <cstdint>
#include <string_view>

namespace rt_deploy::robot {

enum class SafetyFault : std::uint8_t {
    kNone          = 0,
    kStaleState    = 1,
    kStateNonFinite = 2,
    kActionNonFinite = 3,
    kJointLimit    = 4,
    kEStopButton   = 5,
    kStaleOdom     = 7,   // odometry stale in CONTROL (set by the control loop)
};

[[nodiscard]] inline std::string_view to_string(SafetyFault f) noexcept {
    switch (f) {
        case SafetyFault::kNone:            return "none";
        case SafetyFault::kStaleState:      return "stale_state";
        case SafetyFault::kStateNonFinite:  return "state_non_finite";
        case SafetyFault::kActionNonFinite: return "action_non_finite";
        case SafetyFault::kJointLimit:      return "joint_limit";
        case SafetyFault::kEStopButton:     return "estop_button";
        case SafetyFault::kStaleOdom:       return "stale_odom";
    }
    return "unknown";
}

template <std::size_t N>
struct SafetyConfig {
    using JointVec = Eigen::Matrix<float, static_cast<Eigen::Index>(N), 1>;

    std::int64_t stale_ns = 60'000'000;        // 3 × 20 ms control period
    JointVec     joint_min{JointVec::Constant(-std::numeric_limits<float>::infinity())};
    JointVec     joint_max{JointVec::Constant( std::numeric_limits<float>::infinity())};
    float        damping_kd = 8.0F;            // Unitree damping gain
    std::uint8_t estop_button_index = 3;       // KeyMap::select
};

template <std::size_t N>
class SafetyMonitor {
public:
    explicit SafetyMonitor(const SafetyConfig<N>& cfg) noexcept : cfg_{cfg} {}

    // Hot-path check. Returns true if the tick is safe; false if the monitor
    // latched a fault on this or a previous tick. On false, `cmd` is rewritten
    // to a damping command and the fault is recorded.
    [[nodiscard]] bool check_and_guard(std::int64_t                  now_ns,
                                        const math::RobotState<N>&    state,
                                        const RemoteState&            remote,
                                        math::MotorCommand<N>&        cmd) noexcept {
        if (latched_) {
            cmd.set_damping(cfg_.damping_kd);
            return false;
        }

        // 1. Stale state watchdog.
        if (now_ns - state.timestamp_ns > cfg_.stale_ns) {
            latch(SafetyFault::kStaleState, cmd);
            return false;
        }

        // 2. NaN/Inf in state.
        if (!is_finite_vec(state.joint_positions)
            || !is_finite_vec(state.joint_velocities)
            || !is_finite_vec(state.imu_quat_wxyz)
            || !is_finite_vec(state.imu_angular_velocity)) {
            latch(SafetyFault::kStateNonFinite, cmd);
            return false;
        }

        // 3. NaN/Inf in outgoing action.
        if (!is_finite_vec(cmd.positions) || !is_finite_vec(cmd.velocities)
            || !is_finite_vec(cmd.kp) || !is_finite_vec(cmd.kd)
            || !is_finite_vec(cmd.torques)) {
            latch(SafetyFault::kActionNonFinite, cmd);
            return false;
        }

        // 4. Joint position hard limits (optional; inf-filled config disables).
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(N); ++i) {
            const float p = cmd.positions[i];
            if (p < cfg_.joint_min[i] || p > cfg_.joint_max[i]) {
                latch(SafetyFault::kJointLimit, cmd);
                return false;
            }
        }

        // 5. E-stop button (SELECT by default). Latch on rising edge — if the
        // user holds the button down we keep the damping command. The index is
        // bounds-checked so a misconfigured estop_button_index (>= button
        // array size) can never read out of bounds on the hot path.
        if (cfg_.estop_button_index < remote.button.size()
            && remote.button[cfg_.estop_button_index] != 0) {
            latch(SafetyFault::kEStopButton, cmd);
            return false;
        }

        return true;
    }

    // Latch a fault detected outside check_and_guard (e.g. the control loop's
    // odometry staleness check). Same effect as a built-in fault: damping now,
    // sticky until restart. A second call keeps the first fault.
    void trigger_fault(SafetyFault f, math::MotorCommand<N>& cmd) noexcept {
        if (latched_) {
            cmd.set_damping(cfg_.damping_kd);
            return;
        }
        latch(f, cmd);
    }

    [[nodiscard]] bool        is_latched() const noexcept { return latched_; }
    [[nodiscard]] SafetyFault fault() const noexcept { return fault_; }

private:
    template <typename Derived>
    [[nodiscard]] static bool is_finite_vec(const Eigen::MatrixBase<Derived>& v) noexcept {
        return v.allFinite();
    }

    void latch(SafetyFault f, math::MotorCommand<N>& cmd) noexcept {
        latched_ = true;
        fault_   = f;
        cmd.set_damping(cfg_.damping_kd);
    }

    SafetyConfig<N> cfg_;
    bool            latched_{false};
    SafetyFault     fault_{SafetyFault::kNone};
};

}  // namespace rt_deploy::robot
