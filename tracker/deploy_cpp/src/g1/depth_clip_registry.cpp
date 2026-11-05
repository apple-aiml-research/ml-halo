//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#include "rt_deploy/g1/depth_clip_registry.hpp"

#include <cmath>
#include <cstdio>

namespace rt_deploy::g1 {
namespace {

[[nodiscard]] std::string fmt2(float v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.4g", static_cast<double>(v));
    return std::string{buf};
}

}  // namespace

DepthClip expected_depth_clip() noexcept {
    return DepthClip{0.2F, 2.5F};
}

std::string check_depth_clip_matches_checkpoint(const ObsSpec&   spec,
                                                std::string_view policy_path) {
    const GroupCfg* img = nullptr;
    for (const auto& g : spec.groups) {
        if (g.is_image) {
            img = &g;
            break;
        }
    }
    if (img == nullptr) {
        return {};  // proprioceptive-only policy: no depth contract to check
    }

    // Compare at the precision a YAML scalar can express.
    const DepthClip want = expected_depth_clip();
    const auto same = [](float a, float b) { return std::fabs(a - b) < 1.0e-4F; };
    if (same(img->clip_min, want.near_m) && same(img->clip_max, want.far_m)) {
        return {};
    }
    return "observation group '" + img->name + "' declares clip [" + fmt2(img->clip_min) + ", "
           + fmt2(img->clip_max) + "] m but the tracker's depth student trains at [" + fmt2(want.near_m)
           + ", " + fmt2(want.far_m) + "] m (policy '" + std::string{policy_path}
           + "').\n"
             "               Every depth pixel would be mis-scaled and nothing downstream can "
             "detect it:\n"
             "               the tensor shape, num_obs and the ONNX signature all still match.\n"
             "               Fix `observation.groups."
           + img->name
           + ".clip` in the config, or pass --allow_depth_clip_mismatch for a checkpoint\n"
             "               trained with a different clip.";
}

}  // namespace rt_deploy::g1
