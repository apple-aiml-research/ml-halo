//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// Joint index remapping between deployment order (hardware wiring) and
// training order (policy I/O).
//
// Parity with build_index_map in the Python deployment (deploy/utils/math/types.py).
//
//   deploy_to_train[i]: index in deployment order that feeds training slot i.
//                       qj_train[i] = qj_deploy[deploy_to_train[i]]
//   train_to_deploy[i]: index in training order that feeds deployment slot i.
//                       action_deploy[i] = action_train[train_to_deploy[i]]
//
// Both arrays are computed once from the YAML joint-order lists; applying a
// remap on the hot path is a plain indexed gather.

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace rt_deploy::policy {

template <std::size_t N>
struct JointRemap {
    std::array<std::int32_t, N> deploy_to_train{};
    std::array<std::int32_t, N> train_to_deploy{};
};

// Build a JointRemap from two joint-name lists. Throws std::runtime_error on a
// missing or duplicate entry — this runs once at startup, off the hot path.
// g1::load_config applies the same checks, so a config that loaded never throws
// here.
template <std::size_t N>
JointRemap<N> build_joint_remap(const std::array<std::string, N>& deployment_order,
                                 const std::array<std::string, N>& training_order) {
    JointRemap<N> out;

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i + 1; j < N; ++j) {
            if (deployment_order[i] == deployment_order[j] || training_order[i] == training_order[j]) {
                throw std::runtime_error("duplicate joint name in joint order list");
            }
        }
    }

    auto find_index = [](const std::array<std::string, N>& arr,
                          std::string_view                  name) -> std::int32_t {
        for (std::size_t i = 0; i < N; ++i) {
            if (arr[i] == name) {
                return static_cast<std::int32_t>(i);
            }
        }
        throw std::runtime_error("joint name not found in list: " + std::string{name});
    };

    // deploy_to_train[i]: the *deployment* index that, when indexed into the
    // deployment state vector, yields the correct value for *training* slot i.
    // Python: for i, tgt in enumerate(training_order): map[i] = deployment_order.index(tgt)
    for (std::size_t i = 0; i < N; ++i) {
        out.deploy_to_train[i] = find_index(deployment_order, training_order[i]);
    }
    // train_to_deploy[i]: symmetric — training index for deployment slot i.
    for (std::size_t i = 0; i < N; ++i) {
        out.train_to_deploy[i] = find_index(training_order, deployment_order[i]);
    }

    return out;
}

}  // namespace rt_deploy::policy
