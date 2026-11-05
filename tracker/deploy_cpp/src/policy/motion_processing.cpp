//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#include "rt_deploy/policy/motion_processing.hpp"

#include "rt_deploy/math/quat.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace rt_deploy::policy {

namespace {

// Axis-angle from a (possibly negated) wxyz quaternion. Same math as the
// axis_angle_of helper in motion_loader.cpp (parity with
// math_utils.py:axis_angle_from_quat), kept local to this translation unit.
Eigen::Vector3f axis_angle_wxyz(const math::Quat& q) noexcept {
    math::Quat qf = q;
    if (qf[0] < 0.0F) qf = -qf;
    const float mag        = std::sqrt(qf[1] * qf[1] + qf[2] * qf[2] + qf[3] * qf[3]);
    const float half_angle = std::atan2(mag, qf[0]);
    const float angle      = 2.0F * half_angle;
    constexpr float kEps   = 1.0e-6F;
    float sin_half_over_angle;
    if (std::fabs(angle) > kEps) {
        sin_half_over_angle = std::sin(half_angle) / angle;
    } else {
        sin_half_over_angle = 0.5F - angle * angle / 48.0F;
    }
    if (std::fabs(sin_half_over_angle) < 1e-12F) {
        return Eigen::Vector3f::Zero();
    }
    return Eigen::Vector3f{qf[1] / sin_half_over_angle,
                            qf[2] / sin_half_over_angle,
                            qf[3] / sin_half_over_angle};
}

}  // namespace

std::int32_t resample_output_frames(std::int32_t n_rows,
                                    std::int32_t input_fps,
                                    std::int32_t output_fps,
                                    std::int32_t scratch_capacity,
                                    double       start_input_row) noexcept {
    using Eigen::Index;

    if (scratch_capacity <= 0 || n_rows < 1 || input_fps <= 0 || output_fps <= 0) {
        return 0;
    }
    const Index max_out = static_cast<Index>(scratch_capacity - kMaxTransitionSteps);
    if (max_out <= 0) {
        return 0;
    }

    const float input_dt  = 1.0F / static_cast<float>(input_fps);
    const float output_dt = 1.0F / static_cast<float>(output_fps);

    const float duration = (n_rows >= 2)
        ? static_cast<float>(n_rows - 1) * input_dt
        : 0.0F;

    const bool  shifted = (n_rows >= 2) && start_input_row != 0.0
                          && std::isfinite(start_input_row);
    const double in_per_out = static_cast<double>(input_fps) / static_cast<double>(output_fps);
    const Index n_out_input = (n_rows >= 2)
        ? (shifted
               ? std::max<Index>(1, static_cast<Index>(std::ceil(
                     (static_cast<double>(n_rows - 1) - start_input_row) / in_per_out - 1e-9)))
               : std::max<Index>(
                     1, static_cast<Index>(std::ceil(
                            static_cast<double>(duration)
                            / static_cast<double>(output_dt)))))
        : Index{1};

    return static_cast<std::int32_t>(std::min<Index>(n_out_input, max_out));
}

bool resample_motion(const double* raw_qpos,
                     std::int32_t  n_rows,
                     std::int32_t  input_fps,
                     std::int32_t  output_fps,
                     MotionData&   slot,
                     double        start_input_row) noexcept {
    using Eigen::Index;

    const Index n_out = resample_output_frames(n_rows, input_fps, output_fps,
                                               slot.scratch_capacity, start_input_row);
    if (n_out <= 0) {
        return false;
    }

    constexpr std::int32_t kStartRow = kMaxTransitionSteps;
    const float input_dt  = 1.0F / static_cast<float>(input_fps);
    const float output_dt = 1.0F / static_cast<float>(output_fps);
    const float duration = (n_rows >= 2)
        ? static_cast<float>(n_rows - 1) * input_dt
        : 0.0F;
    const bool  shifted = (n_rows >= 2) && start_input_row != 0.0
                          && std::isfinite(start_input_row);
    const double in_per_out = static_cast<double>(input_fps) / static_cast<double>(output_fps);

    slot.output_fps  = output_fps;
    slot.input_fps   = input_fps;
    slot.start_row   = kStartRow;
    slot.frame_count = static_cast<std::int32_t>(n_out);

    constexpr std::size_t kCols = 36;
    for (Index i = 0; i < n_out; ++i) {
        const Index dst = kStartRow + i;

        Index i0    = 0;
        float blend = 0.0F;
        if (shifted) {
            // Exact fractional grid (double): position = start + i * input/output.
            double pos = start_input_row + static_cast<double>(i) * in_per_out;
            pos = std::clamp(pos, 0.0, static_cast<double>(n_rows - 1));
            i0    = std::min<Index>(static_cast<Index>(std::floor(pos + 1e-9)), n_rows - 1);
            blend = static_cast<float>(std::max(0.0, pos - static_cast<double>(i0)));
        } else if (n_rows >= 2) {
            const float t     = static_cast<float>(i) * output_dt;
            const float phase = (duration > 0.0F) ? (t / duration) : 0.0F;
            i0    = static_cast<Index>(std::floor(phase * static_cast<float>(n_rows - 1)));
            blend = phase * static_cast<float>(n_rows - 1) - static_cast<float>(i0);
        }
        const Index i1    = std::min<Index>(i0 + 1, n_rows - 1);

        const double* row0 = raw_qpos + i0 * static_cast<Index>(kCols);
        const double* row1 = raw_qpos + i1 * static_cast<Index>(kCols);

        slot.base_poss(dst, 0) = static_cast<float>(
            row0[0] * (1.0 - static_cast<double>(blend)) + row1[0] * static_cast<double>(blend));
        slot.base_poss(dst, 1) = static_cast<float>(
            row0[1] * (1.0 - static_cast<double>(blend)) + row1[1] * static_cast<double>(blend));
        slot.base_poss(dst, 2) = static_cast<float>(
            row0[2] * (1.0 - static_cast<double>(blend)) + row1[2] * static_cast<double>(blend));

        const math::Quat q0{static_cast<float>(row0[6]), static_cast<float>(row0[3]),
                             static_cast<float>(row0[4]), static_cast<float>(row0[5])};
        const math::Quat q1{static_cast<float>(row1[6]), static_cast<float>(row1[3]),
                             static_cast<float>(row1[4]), static_cast<float>(row1[5])};
        const Eigen::Vector4f q = math::quat_slerp(q0, q1, blend);
        slot.base_rots(dst, 0) = q[0];
        slot.base_rots(dst, 1) = q[1];
        slot.base_rots(dst, 2) = q[2];
        slot.base_rots(dst, 3) = q[3];

        for (std::size_t j = 0; j < kG1Dof; ++j) {
            const double v0 = row0[7 + j];
            const double v1 = row1[7 + j];
            slot.dof_poss(dst, static_cast<Index>(j)) = static_cast<float>(
                v0 * (1.0 - static_cast<double>(blend))
              + v1 * static_cast<double>(blend));
        }
    }

    return true;
}

void compute_motion_derivatives(MotionData& slot) noexcept {
    using Eigen::Index;

    if (slot.frame_count <= 0 || slot.output_fps <= 0) {
        return;
    }

    const float output_dt = 1.0F / static_cast<float>(slot.output_fps);
    const Index r0        = slot.start_row;
    const Index rN        = slot.start_row + slot.frame_count;
    const Index n_out     = slot.frame_count;

    if (n_out == 1) {
        slot.dof_vels.row(r0).setZero();
    } else {
        slot.dof_vels.row(r0)
            = (slot.dof_poss.row(r0 + 1) - slot.dof_poss.row(r0)) / output_dt;
        slot.dof_vels.row(rN - 1)
            = (slot.dof_poss.row(rN - 1) - slot.dof_poss.row(rN - 2)) / output_dt;
        for (Index i = r0 + 1; i < rN - 1; ++i) {
            slot.dof_vels.row(i)
                = (slot.dof_poss.row(i + 1) - slot.dof_poss.row(i - 1))
                  / (2.0F * output_dt);
        }
    }

    if (n_out == 1) {
        slot.base_lin_vels.row(r0).setZero();
    } else {
        slot.base_lin_vels.row(r0)
            = (slot.base_poss.row(r0 + 1) - slot.base_poss.row(r0)) / output_dt;
        slot.base_lin_vels.row(rN - 1)
            = (slot.base_poss.row(rN - 1) - slot.base_poss.row(rN - 2)) / output_dt;
        for (Index i = r0 + 1; i < rN - 1; ++i) {
            slot.base_lin_vels.row(i)
                = (slot.base_poss.row(i + 1) - slot.base_poss.row(i - 1))
                  / (2.0F * output_dt);
        }
    }

    // Same gradient rule as the linear velocities above (and so3_rate in
    // motion_loader.cpp): central difference inside, one-sided at the first /
    // last row.
    slot.base_ang_vels.row(r0).setZero();
    if (n_out >= 2) {
        for (Index i = r0; i < rN; ++i) {
            const Index a = (i > r0) ? i - 1 : i;
            const Index b = (i < rN - 1) ? i + 1 : i;
            const math::Quat qa{slot.base_rots(a, 0), slot.base_rots(a, 1),
                                  slot.base_rots(a, 2), slot.base_rots(a, 3)};
            const math::Quat qb{slot.base_rots(b, 0), slot.base_rots(b, 1),
                                  slot.base_rots(b, 2), slot.base_rots(b, 3)};
            const math::Quat q_rel = math::quat_mul(qb, math::quat_conjugate(qa));
            slot.base_ang_vels.row(i)
                = (axis_angle_wxyz(q_rel) / (static_cast<float>(b - a) * output_dt)).transpose();
        }
    }
}

}  // namespace rt_deploy::policy
