//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// G1 robot config loaded from a YAML file in configs/ (g1_v2.yaml etc.).
//
// YAML is parsed once at startup (off the hot path). Per-joint arrays are
// validated to have exactly kDof = 29 elements and stored in fixed-size Eigen
// vectors, so the hot path indexes them with no heap traffic. The observation
// layout (`observation:` block) is parsed separately by g1::load_observation_spec.

#include "rt_deploy/core/expected.hpp"

#include <Eigen/Core>
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace rt_deploy::g1 {

constexpr std::size_t kDof = 29;

template <std::size_t N>
struct RobotConfig {
    static constexpr std::size_t kDof = N;
    using Vec = Eigen::Matrix<float, static_cast<Eigen::Index>(N), 1>;

    // Identification
    std::string robot_name;
    std::string xml_path;

    // YAML `num_joints`; load_config rejects a value other than N.
    std::int32_t num_joints = static_cast<std::int32_t>(N);

    // Joint orders (deployment = hardware wiring; training = policy I/O).
    std::array<std::string, N> deployment_joint_order{};
    std::array<std::string, N> training_joint_order{};

    // Per-joint numeric params (deployment order unless noted).
    Vec default_angles{Vec::Zero()};
    Vec joint_kps{Vec::Zero()};
    Vec joint_kds{Vec::Zero()};
    Vec action_scale{Vec::Zero()};
    Vec max_torque{Vec::Zero()};
    // SIM ONLY (optional): the sim plant's identified drive. The MuJoCo motor realises joint_kps x sim_kp_scale and
    // joint_kds x sim_kd_scale, the gains training simulates; the real robot ignores both.
    std::optional<Vec> sim_kp_scale{};
    std::optional<Vec> sim_kd_scale{};

    // G1-specific standing references (deployment order).
    Vec                  standing_joint_pos{Vec::Zero()};
    Eigen::Vector3f      standing_base_pos{Eigen::Vector3f::Zero()};
    Eigen::Vector4f      standing_base_quat_wxyz{1.0F, 0.0F, 0.0F, 0.0F};

    // Timing.
    float        simulation_dt  = 0.005F;
    std::int32_t decimation     = 4;      // YAML control_decimation: physics substeps per tick
    float        control_dt     = 0.02F;  // control/inference period (one inference per tick)

    // Safety limits.
    float clip_action      = 100.0F;
    float clip_observation = 100.0F;
};

using Config = RobotConfig<kDof>;

// Parse a YAML file into a Config. Returns an error message on failure.
// {ROOT_DIR} in xml_path is substituted with the given root_dir, or, if empty,
// with the nearest ancestor of the config file that contains assets/robots/
// (falling back to the config file's own directory).
[[nodiscard]] core::Result<Config> load_config(std::string_view yaml_path,
                                                std::string_view root_dir = {});

}  // namespace rt_deploy::g1
