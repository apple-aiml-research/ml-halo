//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// Internal YAML helpers shared by the config loaders (src/g1/config.cpp,
// src/vision/config.cpp). Include only from .cpp files so yaml-cpp does not
// leak into the public headers.
//
// Off the hot path — used once at startup. yaml-cpp throws YAML::Exception on
// a wrongly typed value (e.g. `as<float>()` on a string); the helpers let it
// propagate, and every loader that calls them (g1::load_config,
// vision::load_vision_config) catches it and returns a config error.

#include "rt_deploy/core/expected.hpp"

#include <yaml-cpp/yaml.h>

#include <Eigen/Core>
#include <array>
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <string>
#include <string_view>

namespace rt_deploy::policy::yaml_helpers {

// Replace every occurrence of `from` in `s` with `to`. Used to substitute
// `{ROOT_DIR}` in xml_path etc.
[[nodiscard]] inline std::string
replace_all(std::string s, std::string_view from, std::string_view to) {
    if (from.empty()) {
        return s;
    }
    std::string::size_type pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

// Walk up from the YAML config's directory looking for a directory that holds
// `assets/robots/` (the shared asset tree of the Python and C++ deployments).
// weakly_canonical makes this work for relative and absolute paths alike.
// Falls back to the YAML's parent directory if no anchor is found.
[[nodiscard]] inline std::string
resolve_root_dir(const std::filesystem::path& yaml_path) {
    namespace fs = std::filesystem;
    fs::path probe = fs::weakly_canonical(yaml_path).parent_path();
    while (!probe.empty() && probe != probe.root_path()) {
        if (fs::exists(probe / "assets" / "robots")) {
            return probe.string();
        }
        probe = probe.parent_path();
    }
    return yaml_path.parent_path().string();
}

// Load an N-vector of floats from `node[field]` into a fixed-size Eigen
// column. Returns an error message if the field is missing, not a sequence,
// or has the wrong number of entries.
template <std::size_t N>
[[nodiscard]] inline core::Result<void>
load_vec_field(const YAML::Node&                                                  node,
                std::string_view                                                   field,
                Eigen::Matrix<float, static_cast<Eigen::Index>(N), 1>&             out) {
    const auto f = node[std::string{field}];
    if (!f || !f.IsSequence()) {
        std::ostringstream os;
        os << "config: missing or non-sequence field '" << field << "'";
        return typename core::Result<void>::Error{os.str()};
    }
    if (f.size() != N) {
        std::ostringstream os;
        os << "config: field '" << field << "' has " << f.size()
           << " entries, expected " << N;
        return typename core::Result<void>::Error{os.str()};
    }
    for (std::size_t i = 0; i < N; ++i) {
        out[static_cast<Eigen::Index>(i)] = f[i].as<float>();
    }
    return {};
}

// Load an N-vector of joint names into a fixed-size std::array<std::string, N>.
template <std::size_t N>
[[nodiscard]] inline core::Result<void>
load_joint_order(const YAML::Node&            node,
                  std::string_view             field,
                  std::array<std::string, N>&  out) {
    const auto f = node[std::string{field}];
    if (!f || !f.IsSequence()) {
        std::ostringstream os;
        os << "config: missing or non-sequence field '" << field << "'";
        return typename core::Result<void>::Error{os.str()};
    }
    if (f.size() != N) {
        std::ostringstream os;
        os << "config: field '" << field << "' has " << f.size()
           << " entries, expected " << N;
        return typename core::Result<void>::Error{os.str()};
    }
    for (std::size_t i = 0; i < N; ++i) {
        out[i] = f[i].as<std::string>();
    }
    return {};
}

}  // namespace rt_deploy::policy::yaml_helpers
