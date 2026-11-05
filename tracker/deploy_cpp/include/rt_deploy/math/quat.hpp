//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// Quaternion math (wxyz convention, parity with the Python deployment's
// deploy/utils/math/math_utils.py, referenced as math_utils.py below).
//
// - All functions are noexcept and hot-path safe; all but quat_slerp are
//   header-inline.
// - Eigen::Vector4f layout is [w, x, y, z]. Eigen::Quaternionf stores
//   (x, y, z, w) internally, so it is deliberately not used here: the memory
//   layout matches the Python / ONNX pipeline.

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace rt_deploy::math {

using Quat  = Eigen::Matrix<float, 4, 1>;  // wxyz
using Vec3  = Eigen::Matrix<float, 3, 1>;
using Mat3  = Eigen::Matrix<float, 3, 3>;

// Straight Hamilton product of two wxyz quaternions. Numerical parity with
// Python's factored form in math_utils.py:quat_mul is verified to 1e-7.
[[nodiscard]] inline Quat quat_mul(const Quat& a, const Quat& b) noexcept {
    const float w1 = a[0], x1 = a[1], y1 = a[2], z1 = a[3];
    const float w2 = b[0], x2 = b[1], y2 = b[2], z2 = b[3];
    Quat        out;
    out[0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
    out[1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
    out[2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
    out[3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
    return out;
}

[[nodiscard]] inline Quat quat_conjugate(const Quat& q) noexcept {
    return Quat{q[0], -q[1], -q[2], -q[3]};
}

// q_rel such that q2 = q1 * q_rel.  Mirrors math_utils.py:quat_difference.
[[nodiscard]] inline Quat quat_difference(const Quat& q1, const Quat& q2) noexcept {
    return quat_mul(quat_conjugate(q1), q2);
}

// Rotate the world-frame gravity vector [0, 0, -1] into the body frame using
// the inverse of q. Exact formula from math_utils.py:get_gravity_orientation:
//     gx =  2 * (-qz * qx + qw * qy)
//     gy = -2 * ( qz * qy + qw * qx)
//     gz =  1 - 2 * (qw*qw + qz*qz)
[[nodiscard]] inline Vec3 gravity_body(const Quat& q) noexcept {
    const float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
    return Vec3{
        2.0F * (-qz * qx + qw * qy),
        -2.0F * (qz * qy + qw * qx),
        1.0F - 2.0F * (qw * qw + qz * qz),
    };
}

// Rotate vector `v` by the INVERSE of q (wxyz), i.e. world frame -> body frame.
// Parity with deploy/utils/core/observation_builder.py:quat_rotate_inverse_batch.
// For v = [0,0,-1] this equals gravity_body(q).
[[nodiscard]] inline Vec3 quat_rotate_inverse(const Quat& q, const Vec3& v) noexcept {
    const float w  = q[0];
    const float qx = -q[1], qy = -q[2], qz = -q[3];  // conjugate vector part
    const Vec3  uv{qy * v[2] - qz * v[1],
                   qz * v[0] - qx * v[2],
                   qx * v[1] - qy * v[0]};
    const Vec3  uuv{qy * uv[2] - qz * uv[1],
                    qz * uv[0] - qx * uv[2],
                    qx * uv[1] - qy * uv[0]};
    return Vec3{v[0] + 2.0F * (w * uv[0] + uuv[0]),
                v[1] + 2.0F * (w * uv[1] + uuv[1]),
                v[2] + 2.0F * (w * uv[2] + uuv[2])};
}

// Standard Hamilton quaternion-to-rotation-matrix. Input is wxyz.
[[nodiscard]] inline Mat3 quat_to_rot(const Quat& q) noexcept {
    const float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
    Mat3        r;
    r << 1.0F - 2.0F * (qy * qy + qz * qz),  //
        2.0F * (qx * qy - qw * qz),          //
        2.0F * (qx * qz + qw * qy),          //
        2.0F * (qx * qy + qw * qz),          //
        1.0F - 2.0F * (qx * qx + qz * qz),   //
        2.0F * (qy * qz - qw * qx),          //
        2.0F * (qx * qz - qw * qy),          //
        2.0F * (qy * qz + qw * qx),          //
        1.0F - 2.0F * (qx * qx + qy * qy);
    return r;
}

// Euler (roll, pitch, yaw) from wxyz quaternion. ZYX intrinsic.
[[nodiscard]] inline Vec3 quat_to_euler(const Quat& q) noexcept {
    const float w = q[0], x = q[1], y = q[2], z = q[3];

    const float sinr_cosp = 2.0F * (w * x + y * z);
    const float cosr_cosp = 1.0F - 2.0F * (x * x + y * y);
    const float roll      = std::atan2(sinr_cosp, cosr_cosp);

    float       sinp  = 2.0F * (w * y - z * x);
    sinp              = std::max(-1.0F, std::min(1.0F, sinp));
    const float pitch = std::asin(sinp);

    const float siny_cosp = 2.0F * (w * z + x * y);
    const float cosy_cosp = 1.0F - 2.0F * (y * y + z * z);
    const float yaw       = std::atan2(siny_cosp, cosy_cosp);

    return Vec3{roll, pitch, yaw};
}

// Spherical linear interpolation. tau in [0, 1].
[[nodiscard]] Quat quat_slerp(const Quat& q1, const Quat& q2, float tau) noexcept;

}  // namespace rt_deploy::math
