//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#include "rt_deploy/math/quat.hpp"

#include <algorithm>
#include <limits>

namespace rt_deploy::math {

Quat quat_slerp(const Quat& q1, const Quat& q2, float tau) noexcept {
    if (tau <= 0.0F) {
        return q1;
    }
    if (tau >= 1.0F) {
        return q2;
    }

    float d = q1.dot(q2);
    Quat  q2_adj = q2;
    if (d < 0.0F) {
        d      = -d;
        q2_adj = -q2;
    }

    constexpr float kEps = 4.0F * std::numeric_limits<float>::epsilon();
    if (1.0F - d < kEps) {
        return q1;
    }

    const float angle    = std::acos(std::min(1.0F, std::max(-1.0F, d)));
    if (std::fabs(angle) < kEps) {
        return q1;
    }
    const float isin  = 1.0F / std::sin(angle);
    const float w1    = std::sin((1.0F - tau) * angle) * isin;
    const float w2    = std::sin(tau * angle) * isin;
    return Quat{w1 * q1[0] + w2 * q2_adj[0], w1 * q1[1] + w2 * q2_adj[1],
                w1 * q1[2] + w2 * q2_adj[2], w1 * q1[3] + w2 * q2_adj[3]};
}

}  // namespace rt_deploy::math
