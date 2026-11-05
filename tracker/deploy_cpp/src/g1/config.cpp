//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#include "rt_deploy/g1/config.hpp"

#include "rt_deploy/policy/yaml_helpers.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <filesystem>
#include <optional>
#include <string>
#include <sstream>
#include <utility>

namespace rt_deploy::g1 {

using policy::yaml_helpers::load_joint_order;
using policy::yaml_helpers::load_vec_field;
using policy::yaml_helpers::replace_all;
using policy::yaml_helpers::resolve_root_dir;

namespace {

// Body of load_config. yaml-cpp reports a wrongly typed value (a string where a
// number belongs, a map where a list belongs) by throwing; the wrapper below turns
// that into a config error instead of letting it terminate the process.
core::Result<Config> parse_config(const YAML::Node& node, const std::filesystem::path& path,
                                  std::string_view root_dir) {
    Config cfg;

    // Identification. Same default as the Python ControllerConfig.
    cfg.robot_name = node["robot_name"] ? node["robot_name"].as<std::string>() : "unknown";

    // {ROOT_DIR} substitution: explicit root_dir, else the nearest ancestor of
    // the config file that holds `assets/robots/` (else the config's directory).
    const std::string resolved_root =
        root_dir.empty() ? resolve_root_dir(path) : std::string{root_dir};
    if (auto xml = node["xml_path"]; xml && xml.IsScalar()) {
        cfg.xml_path = replace_all(xml.as<std::string>(), "{ROOT_DIR}", resolved_root);
    }

    // Dimensions
    cfg.num_joints = node["num_joints"] ? node["num_joints"].as<std::int32_t>() : static_cast<std::int32_t>(kDof);

    if (cfg.num_joints != static_cast<std::int32_t>(kDof)) {
        std::ostringstream os;
        os << "config: num_joints=" << cfg.num_joints << " but this binary is built for N=" << kDof;
        return typename core::Result<Config>::Error{os.str()};
    }

    // Joint orders
    if (auto r = load_joint_order<kDof>(node, "deployment_joint_order", cfg.deployment_joint_order); !r) {
        return typename core::Result<Config>::Error{r.error()};
    }
    if (auto r = load_joint_order<kDof>(node, "training_joint_order", cfg.training_joint_order); !r) {
        return typename core::Result<Config>::Error{r.error()};
    }
    // The two orders must be permutations of one another (no duplicates, same
    // names), which is what policy::build_joint_remap needs. Checked here so
    // a bad config is a load error rather than an exception at startup.
    for (std::size_t i = 0; i < kDof; ++i) {
        const std::string& name = cfg.training_joint_order[i];
        for (std::size_t j = i + 1; j < kDof; ++j) {
            if (cfg.training_joint_order[j] == name || cfg.deployment_joint_order[j]
                                                           == cfg.deployment_joint_order[i]) {
                return typename core::Result<Config>::Error{
                    "config: duplicate joint name in training_joint_order or "
                    "deployment_joint_order (entries " + std::to_string(i) + " and "
                    + std::to_string(j) + ")"};
            }
        }
        if (std::find(cfg.deployment_joint_order.begin(), cfg.deployment_joint_order.end(),
                      name) == cfg.deployment_joint_order.end()) {
            return typename core::Result<Config>::Error{
                "config: training joint '" + name + "' is not in deployment_joint_order"};
        }
    }

    // Per-joint vectors
    if (auto r = load_vec_field<kDof>(node, "default_angles", cfg.default_angles); !r) {
        return typename core::Result<Config>::Error{r.error()};
    }
    if (auto r = load_vec_field<kDof>(node, "joint_kps", cfg.joint_kps); !r) {
        return typename core::Result<Config>::Error{r.error()};
    }
    if (auto r = load_vec_field<kDof>(node, "joint_kds", cfg.joint_kds); !r) {
        return typename core::Result<Config>::Error{r.error()};
    }

    // action_scale — per-joint vector (required, as in the Python ControllerConfig).
    if (auto r = load_vec_field<kDof>(node, "action_scale", cfg.action_scale); !r) {
        return typename core::Result<Config>::Error{r.error()};
    }

    // max_torque — per-joint vector.
    if (auto r = load_vec_field<kDof>(node, "max_torque", cfg.max_torque); !r) {
        return typename core::Result<Config>::Error{r.error()};
    }

    // sim_kp_scale / sim_kd_scale — optional per-joint vectors (MuJoCo plant drive), finite and > 0.
    const std::pair<const char*, std::optional<Config::Vec>*> sim_scales[] = {
        {"sim_kp_scale", &cfg.sim_kp_scale}, {"sim_kd_scale", &cfg.sim_kd_scale}};
    for (const auto& [key, target] : sim_scales) {
        if (!node[key]) continue;
        Config::Vec v{Config::Vec::Zero()};
        if (auto r = load_vec_field<kDof>(node, key, v); !r) {
            return typename core::Result<Config>::Error{r.error()};
        }
        if (!v.allFinite() || (v.array() <= 0.0F).any()) {
            return typename core::Result<Config>::Error{std::string{"config: "} + key + " must be finite and > 0"};
        }
        *target = v;
    }

    // Standing references (deployment order). Required: the standing pose is the
    // motion every run holds before and after a clip, so a missing or mis-sized
    // entry must not fall back to a pose nobody chose.
    if (auto r = load_vec_field<kDof>(node, "standing_joint_pos", cfg.standing_joint_pos); !r) {
        return typename core::Result<Config>::Error{r.error()};
    }
    if (auto r = load_vec_field<3>(node, "standing_base_pos", cfg.standing_base_pos); !r) {
        return typename core::Result<Config>::Error{r.error()};
    }
    if (auto r = load_vec_field<4>(node, "standing_base_quat_wxyz", cfg.standing_base_quat_wxyz);
        !r) {
        return typename core::Result<Config>::Error{r.error()};
    }

    // Timing
    cfg.simulation_dt = node["simulation_dt"] ? node["simulation_dt"].as<float>() : 0.005F;
    cfg.decimation    = node["control_decimation"] ? node["control_decimation"].as<std::int32_t>() : 4;
    cfg.control_dt    = node["control_dt"] ? node["control_dt"].as<float>()
                                            : cfg.simulation_dt * static_cast<float>(cfg.decimation);

    // Safety
    cfg.clip_action      = node["clip_action"]      ? node["clip_action"].as<float>() : 100.0F;
    cfg.clip_observation = node["clip_observation"] ? node["clip_observation"].as<float>() : 100.0F;

    return cfg;
}

}  // namespace

core::Result<Config> load_config(std::string_view yaml_path, std::string_view root_dir) {
    namespace fs = std::filesystem;
    fs::path path{std::string{yaml_path}};
    if (!fs::exists(path)) {
        return typename core::Result<Config>::Error{"config file not found: " + path.string()};
    }

    try {
        const YAML::Node node = YAML::LoadFile(path.string());
        return parse_config(node, path, root_dir);
    } catch (const YAML::Exception& e) {
        return typename core::Result<Config>::Error{"config " + path.string() + ": " + e.what()};
    } catch (const std::filesystem::filesystem_error& e) {   // {ROOT_DIR} resolution
        return typename core::Result<Config>::Error{"config " + path.string() + ": " + e.what()};
    }
}

}  // namespace rt_deploy::g1
