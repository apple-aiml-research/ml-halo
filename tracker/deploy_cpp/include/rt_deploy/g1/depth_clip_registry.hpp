//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// Depth-clip contract: the encode clip the tracker's depth student trains with.
//
// WHY THIS EXISTS. The `student_depth` encode is clip([near,far]) -> log1p. Get
// the clip wrong and every pixel is mis-scaled — but the tensor shape, num_obs
// and the ONNX signature all still match, so the policy runs, the robot moves,
// and nothing anywhere reports a problem. It is a silent-failure class, so the
// config's declared value is checked against the training clip
// (training/.../g1/motion_tracking_scene_distill_cfg.py: DEPTH_CLIP) and a
// mismatch refuses to start unless --allow_depth_clip_mismatch is given.
//
// Mirrors deploy/g1/variant_support.py's EXPECTED_DEPTH_CLIP / check_depth_clip,
// so the Python and C++ deployments cannot disagree about the contract.

#include "rt_deploy/g1/observation_builder.hpp"

#include <string>
#include <string_view>

namespace rt_deploy::g1 {

struct DepthClip {
    float near_m;
    float far_m;
};

// The clip the tracker's depth student trains with (training DEPTH_CLIP).
[[nodiscard]] DepthClip expected_depth_clip() noexcept;

// Empty when `spec`'s image group declares the training clip (or when the spec
// has no image group). Otherwise a human-readable explanation of the mismatch,
// suitable for printing before refusing to run. `policy_path` only names the
// policy in the message.
[[nodiscard]] std::string check_depth_clip_matches_checkpoint(const ObsSpec&   spec,
                                                              std::string_view policy_path);

}  // namespace rt_deploy::g1
