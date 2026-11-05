//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// RobotState<N> / MotorCommand<N> — Eigen fixed-size, zero-heap.
// Parity with deploy/utils/math/types.py:RobotState and MotorCommand.
//
// All joint arrays are in *deployment* order. The policy-side conversion to
// training order happens in the policy module, not here.

#include <Eigen/Core>
#include <array>
#include <chrono>
#include <cstdint>

namespace rt_deploy::math {

// ─── State machine (parity with deploy/utils/core/base_controller.py ControlState)
enum class ControlState : std::uint32_t {
    kInit             = 0,
    kMovingToDefault  = 1,
    kDefaultPos       = 2,
    kStopping         = 3,
    kControl          = 10,
};

// ─── Unitree wireless-remote keymap (deploy/utils/core/remote_controller.py:KeyMap)
namespace key_map {
constexpr std::uint8_t kR1     = 0;
constexpr std::uint8_t kL1     = 1;
constexpr std::uint8_t kStart  = 2;
constexpr std::uint8_t kSelect = 3;
constexpr std::uint8_t kR2     = 4;
constexpr std::uint8_t kL2     = 5;
constexpr std::uint8_t kF1     = 6;
constexpr std::uint8_t kF2     = 7;
constexpr std::uint8_t kA      = 8;
constexpr std::uint8_t kB      = 9;
constexpr std::uint8_t kX      = 10;
constexpr std::uint8_t kY      = 11;
constexpr std::uint8_t kUp     = 12;
constexpr std::uint8_t kRight  = 13;
constexpr std::uint8_t kDown   = 14;
constexpr std::uint8_t kLeft   = 15;
}  // namespace key_map

// ─── RobotState<N> ───────────────────────────────────────────────────────────
template <std::size_t N>
struct RobotState {
    using JointVec = Eigen::Matrix<float, static_cast<Eigen::Index>(N), 1>;

    JointVec        joint_positions{JointVec::Zero()};
    JointVec        joint_velocities{JointVec::Zero()};
    JointVec        joint_torques{JointVec::Zero()};
    Eigen::Vector4f imu_quat_wxyz{1.0F, 0.0F, 0.0F, 0.0F};
    Eigen::Vector3f imu_angular_velocity{Eigen::Vector3f::Zero()};
    Eigen::Vector3f imu_linear_acceleration{Eigen::Vector3f::Zero()};

    // Optional sport-mode state-estimator output (G1 rt/odommodestate topic).
    // Mirrors the odom_* fields of deploy/utils/math/types.py:RobotState.
    // `odom_valid` says whether the fields below carry data on this read; they
    // default to identity/zero so consumers that don't care can ignore it.
    bool            odom_valid{false};
    Eigen::Vector3f odom_position{Eigen::Vector3f::Zero()};        // xyz, world frame
    Eigen::Vector3f odom_velocity{Eigen::Vector3f::Zero()};        // xyz, world frame
    Eigen::Vector4f odom_quaternion_wxyz{1.0F, 0.0F, 0.0F, 0.0F};  // base orient. (wxyz)
    float           odom_yaw_speed{0.0F};                           // rad/s
    // Monotonic receive time of the odom fields (0 = never received). The real
    // backend clears odom_valid once this is older than its staleness timeout.
    std::int64_t    odom_timestamp_ns{0};

    std::int64_t    timestamp_ns{0};

    [[nodiscard]] static constexpr std::size_t dof() noexcept { return N; }
};

// ─── MotorCommand<N> ─────────────────────────────────────────────────────────
// Fields mirror the Python MotorCommand so the mapping to LowCmd motor_cmd[i]
// matches deploy/utils/backends/unitree_backend.py:send_command.
template <std::size_t N>
struct MotorCommand {
    using JointVec  = Eigen::Matrix<float, static_cast<Eigen::Index>(N), 1>;
    using ModeArray = std::array<std::int32_t, N>;

    JointVec  positions{JointVec::Zero()};   // motor_cmd[i].q
    JointVec  velocities{JointVec::Zero()};  // motor_cmd[i].dq
    JointVec  kp{JointVec::Zero()};
    JointVec  kd{JointVec::Zero()};
    JointVec  torques{JointVec::Zero()};     // motor_cmd[i].tau
    ModeArray modes{};                        // 0=position, 1=velocity

    // Parity with remote_controller.py:create_damping_cmd (Unitree: kd=8).
    void set_damping(float damping_kd = 8.0F) noexcept {
        positions.setZero();
        velocities.setZero();
        kp.setZero();
        kd.setConstant(damping_kd);
        torques.setZero();
        modes.fill(0);
    }
};

}  // namespace rt_deploy::math
