//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#include "rt_deploy/policy/motion_loader.hpp"

#include "rt_deploy/g1/active_plan_snapshot.hpp"
#include "rt_deploy/math/quat.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

namespace rt_deploy::policy {

namespace {

using Row36    = Eigen::Matrix<float, 1, static_cast<Eigen::Index>(kCsvCols)>;
using RowsMat  = Eigen::Matrix<float, Eigen::Dynamic, static_cast<Eigen::Index>(kCsvCols),
                                Eigen::RowMajor>;

// Minimal CSV parser for this project's motion format: numeric-only,
// comma-delimited, no header. Rows with the wrong column count are an error.
core::Result<RowsMat> parse_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        return typename core::Result<RowsMat>::Error{"cannot open motion file: " + path};
    }

    std::vector<float> values;
    values.reserve(1 << 14);
    std::int64_t rows = 0;
    std::string  line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::int64_t cols = 0;
        std::size_t  pos  = 0;
        while (pos <= line.size()) {
            const std::size_t comma = line.find(',', pos);
            const std::string tok   = line.substr(pos, (comma == std::string::npos)
                                                          ? std::string::npos
                                                          : comma - pos);
            try {
                values.push_back(std::stof(tok));
            } catch (...) {
                return typename core::Result<RowsMat>::Error{
                    "CSV parse error at row " + std::to_string(rows)
                    + " col " + std::to_string(cols)};
            }
            ++cols;
            if (comma == std::string::npos) {
                break;
            }
            pos = comma + 1;
        }
        if (cols != static_cast<std::int64_t>(kCsvCols)) {
            return typename core::Result<RowsMat>::Error{
                "row " + std::to_string(rows) + " has " + std::to_string(cols)
                + " columns, expected " + std::to_string(kCsvCols)};
        }
        ++rows;
    }

    RowsMat m(rows, static_cast<Eigen::Index>(kCsvCols));
    for (Eigen::Index r = 0; r < rows; ++r) {
        for (Eigen::Index c = 0; c < static_cast<Eigen::Index>(kCsvCols); ++c) {
            m(r, c) = values[static_cast<std::size_t>(r) * kCsvCols + static_cast<std::size_t>(c)];
        }
    }
    return m;
}

// Linear interpolation between rows a and b of a matrix, with blend factor t.
template <typename Mat, typename OutRow>
void lerp_rows(const Mat& src, Eigen::Index i0, Eigen::Index i1, float t, OutRow& out) {
    out = src.row(i0) * (1.0F - t) + src.row(i1) * t;
}

// SLERP of two quaternion rows (wxyz).
Eigen::Vector4f slerp_rows(const Eigen::Matrix<float, Eigen::Dynamic, 4, Eigen::RowMajor>& src,
                            Eigen::Index                                                    i0,
                            Eigen::Index                                                    i1,
                            float                                                           t) {
    const math::Quat a{src(i0, 0), src(i0, 1), src(i0, 2), src(i0, 3)};
    const math::Quat b{src(i1, 0), src(i1, 1), src(i1, 2), src(i1, 3)};
    return math::quat_slerp(a, b, t);
}

// ── Trim re-gauge / downsample helpers ──

using Base3MatT = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
using Base4MatT = Eigen::Matrix<float, Eigen::Dynamic, 4, Eigen::RowMajor>;
using DofMatT   = Eigen::Matrix<float, Eigen::Dynamic,
                                 static_cast<Eigen::Index>(kG1Dof), Eigen::RowMajor>;

// Yaw-only re-gauge anchored at a frame-0 world pose. Recenters position so
// frame 0 sits at the origin (XYZ all → 0) and removes ONLY the frame-0 yaw
// from orientation (roll/pitch of the quaternion stay absolute). The rotation
// is about world-Z, so it's z-invariant — the (already-recentered) Z column
// passes through unchanged. The receiving end (align_motion_to_anchor) puts
// absolute XYZ back via the install anchor.
//
// Why yaw-only and not full SO(3) on the orientation: if we also removed
// frame-0 roll/pitch, forward travel in the absolute frame would partially
// get re-expressed as Z in the rotated frame (~sin(tilt) · forward_dist).
// align_motion_to_anchor only carries yaw, so it can't undo that mixing —
// and the error compounds into a Z ramp across reference-anchored
// trim→resend cycles.
struct YawGauge {
    Eigen::Vector3f p0{Eigen::Vector3f::Zero()};
    math::Quat      qz_inv{1.0F, 0.0F, 0.0F, 0.0F};  // rot about world-z by -yaw0 (wxyz)
    float           c{1.0F};                          // cos(yaw0)
    float           s{0.0F};                          // sin(yaw0)
};

YawGauge make_yaw_gauge(const Eigen::Vector3f& p0, const math::Quat& q0_wxyz) noexcept {
    const float yaw0 = math::quat_to_euler(q0_wxyz)[2];
    const float half = 0.5F * yaw0;
    YawGauge g;
    g.p0     = p0;
    g.qz_inv = math::Quat{std::cos(half), 0.0F, 0.0F, -std::sin(half)};
    g.c      = std::cos(yaw0);
    g.s      = std::sin(yaw0);
    return g;
}

// Apply the yaw-only gauge to a world (pos, quat_wxyz) and write the first 7
// wire columns [xyz(3), quat_xyzw(4)] of a RawMotionPlan row at `out`.
void write_gauged_pose(double* out, const YawGauge& g,
                       const Eigen::Vector3f& p_world,
                       const math::Quat&      q_world_wxyz) noexcept {
    const float dx = p_world[0] - g.p0[0];
    const float dy = p_world[1] - g.p0[1];
    const float dz = p_world[2] - g.p0[2];
    out[0] = static_cast<double>(g.c * dx + g.s * dy);   // rotate XY by -yaw0
    out[1] = static_cast<double>(-g.s * dx + g.c * dy);
    out[2] = static_cast<double>(dz);                    // z recentered, NOT rotated
    math::Quat ql = math::quat_mul(g.qz_inv, q_world_wxyz);
    ql.normalize();  // quat_mul drifts off unit norm in float; renormalize.
    out[3] = static_cast<double>(ql[1]);  // wxyz -> xyzw on the wire
    out[4] = static_cast<double>(ql[2]);
    out[5] = static_cast<double>(ql[3]);
    out[6] = static_cast<double>(ql[0]);
}

// Downsample a training-order/wxyz live window [r0, r0+n_full) to
// `target_input_fps` with matched LERP (pos/joints) / SLERP (quat) on the same
// grid resample uses, apply the yaw-only re-gauge, and write deployment-order /
// xyzw rows into `out` (RawMotionPlan::kCols stride). Returns n_out (<= kMaxRows).
// Parity with G1Controller.trim_ref_motion's downsample fallback —
// interpolated, NOT nearest-index linspace truncation.
std::int32_t downsample_window_to_plan(
    const Base3MatT& base_poss, const Base4MatT& base_rots, const DofMatT& dof_poss,
    Eigen::Index r0, std::int32_t n_full,
    std::int32_t output_fps, std::int32_t target_input_fps,
    const JointRemap<kG1Dof>& remap, double* out) noexcept {
    using Eigen::Index;

    // Output row count: parity with np.arange(0, duration, 1/input_fps) length
    // (same ceil convention resample_motion uses). fps-match / single-frame is
    // an identity copy.
    std::int32_t n_out;
    if (n_full <= 1 || target_input_fps == output_fps) {
        n_out = n_full;
    } else {
        const double output_dt = 1.0 / static_cast<double>(output_fps);
        const double duration  = static_cast<double>(n_full - 1) * output_dt;
        n_out = std::max<std::int32_t>(
            1, static_cast<std::int32_t>(
                   std::ceil(duration * static_cast<double>(target_input_fps))));
    }
    n_out = std::min<std::int32_t>(n_out, RawMotionPlan::kMaxRows);

    // Frame-0 gauge from the first selected (interpolated) frame == source r0.
    const Eigen::Vector3f p0{base_poss(r0, 0), base_poss(r0, 1), base_poss(r0, 2)};
    const math::Quat      q0{base_rots(r0, 0), base_rots(r0, 1),
                             base_rots(r0, 2), base_rots(r0, 3)};
    const YawGauge        gauge = make_yaw_gauge(p0, q0);

    const double input_dt  = 1.0 / static_cast<double>(target_input_fps);
    const double output_dt = 1.0 / static_cast<double>(output_fps);
    const double duration  = static_cast<double>(n_full - 1) * output_dt;

    for (std::int32_t i = 0; i < n_out; ++i) {
        double fidx;
        if (n_full <= 1 || duration <= 0.0) {
            fidx = 0.0;
        } else {
            const double t = static_cast<double>(i) * input_dt;
            fidx = (t / duration) * static_cast<double>(n_full - 1);
        }
        Index i0 = static_cast<Index>(std::floor(fidx));
        if (i0 < 0) i0 = 0;
        if (i0 > n_full - 1) i0 = n_full - 1;
        const Index i1    = std::min<Index>(i0 + 1, n_full - 1);
        const float blend = static_cast<float>(fidx - static_cast<double>(i0));

        const Index s0 = r0 + i0;
        const Index s1 = r0 + i1;

        const Eigen::Vector3f p_world{
            base_poss(s0, 0) * (1.0F - blend) + base_poss(s1, 0) * blend,
            base_poss(s0, 1) * (1.0F - blend) + base_poss(s1, 1) * blend,
            base_poss(s0, 2) * (1.0F - blend) + base_poss(s1, 2) * blend};
        const math::Quat qa{base_rots(s0, 0), base_rots(s0, 1),
                            base_rots(s0, 2), base_rots(s0, 3)};
        const math::Quat qb{base_rots(s1, 0), base_rots(s1, 1),
                            base_rots(s1, 2), base_rots(s1, 3)};
        const math::Quat q_world = math::quat_slerp(qa, qb, blend);

        double* row = out + static_cast<std::size_t>(i)
                          * static_cast<std::size_t>(RawMotionPlan::kCols);
        write_gauged_pose(row, gauge, p_world, q_world);

        // Joints: training order in the window, deployment order on the wire.
        for (std::size_t j = 0; j < kG1Dof; ++j) {
            const Index src = static_cast<Index>(remap.train_to_deploy[j]);
            row[7 + j] = static_cast<double>(
                dof_poss(s0, src) * (1.0F - blend) + dof_poss(s1, src) * blend);
        }
    }
    return n_out;
}

// Master trim: the master clip sampled at row `mf` and every following input
// frame (mf, mf+1, ...), LERP pos/joints + SLERP quat, then the yaw-only
// re-gauge (parity with G1Controller.trim_ref_motion's master path). `master` rows are the raw wire layout [xyz(3), quat_xyzw(4),
// joints_deploy(29)] at master_fps. The trims pass a WHOLE row
// (master_trim_start), so the blend is 0 and the result is an exact slice of
// stored frames (re-fed prefixes are never low-passed); the fractional gap is
// re-applied by the next install (MotionData::trim_phase). `mf` is clamped to
// [0, master_rows - 1], so the prefix is never empty. Returns n_out (<= kMaxRows).
std::int32_t master_interp_to_plan(
    const Eigen::Matrix<float, Eigen::Dynamic,
                        static_cast<Eigen::Index>(kCsvCols), Eigen::RowMajor>& master,
    std::int32_t master_rows, double mf, double* out) noexcept {
    using Eigen::Index;
    if (master_rows <= 0) return 0;
    const double last = static_cast<double>(master_rows - 1);
    if (!(mf > 0.0)) mf = 0.0;            // also catches NaN
    if (mf > last) mf = last;
    const std::int32_t n_out = std::min<std::int32_t>(
        static_cast<std::int32_t>(std::floor(last - mf + 1e-9)) + 1, RawMotionPlan::kMaxRows);

    auto sample = [&](double f, Eigen::Vector3f& p, math::Quat& q, float* joints) {
        Index i0 = static_cast<Index>(std::floor(f + 1e-9));
        if (i0 < 0) i0 = 0;
        if (i0 > master_rows - 1) i0 = master_rows - 1;
        const Index i1  = std::min<Index>(i0 + 1, master_rows - 1);
        float       b   = static_cast<float>(f - static_cast<double>(i0));
        if (b < 1e-6F) b = 0.0F;
        if (b > 1.0F) b = 1.0F;
        for (int k = 0; k < 3; ++k) p[k] = master(i0, k) * (1.0F - b) + master(i1, k) * b;
        const math::Quat qa{master(i0, 6), master(i0, 3), master(i0, 4), master(i0, 5)};  // xyzw->wxyz
        const math::Quat qb{master(i1, 6), master(i1, 3), master(i1, 4), master(i1, 5)};
        q = (b == 0.0F) ? qa : math::quat_slerp(qa, qb, b);
        for (std::size_t j = 0; j < kG1Dof; ++j) {
            const Index c = 7 + static_cast<Index>(j);
            joints[j] = master(i0, c) * (1.0F - b) + master(i1, c) * b;
        }
    };

    Eigen::Vector3f p0;
    math::Quat      q0;
    float           j0[kG1Dof];
    sample(mf, p0, q0, j0);
    const YawGauge gauge = make_yaw_gauge(p0, q0);
    for (std::int32_t i = 0; i < n_out; ++i) {
        Eigen::Vector3f p;
        math::Quat      q;
        float           jj[kG1Dof];
        sample(mf + static_cast<double>(i), p, q, jj);
        double* row = out + static_cast<std::size_t>(i)
                          * static_cast<std::size_t>(RawMotionPlan::kCols);
        write_gauged_pose(row, gauge, p, q);
        for (std::size_t j = 0; j < kG1Dof; ++j) row[7 + j] = static_cast<double>(jj[j]);
    }
    return n_out;
}

// np.gradient over axis 0 (central differences, 1-sided at the endpoints).
template <typename Mat>
Mat gradient_axis0(const Mat& x, float dt) {
    Mat g(x.rows(), x.cols());
    if (x.rows() == 0) {
        return g;
    }
    if (x.rows() == 1) {
        g.setZero();
        return g;
    }
    g.row(0)            = (x.row(1) - x.row(0)) / dt;
    g.row(x.rows() - 1) = (x.row(x.rows() - 1) - x.row(x.rows() - 2)) / dt;
    for (Eigen::Index i = 1; i < x.rows() - 1; ++i) {
        g.row(i) = (x.row(i + 1) - x.row(i - 1)) / (2.0F * dt);
    }
    return g;
}

// Axis-angle vector of a quaternion (wxyz), matching math_utils.py:axis_angle_from_quat.
Eigen::Vector3f axis_angle_of(const math::Quat& q) noexcept {
    math::Quat qf = q;
    if (qf[0] < 0.0F) {
        qf = -qf;  // ensure positive real part
    }
    const float mag = std::sqrt(qf[1] * qf[1] + qf[2] * qf[2] + qf[3] * qf[3]);
    const float half_angle = std::atan2(mag, qf[0]);
    const float angle      = 2.0F * half_angle;
    constexpr float kEps   = 1.0e-6F;
    float           sin_half_over_angle;
    if (std::fabs(angle) > kEps) {
        sin_half_over_angle = std::sin(half_angle) / angle;
    } else {
        sin_half_over_angle = 0.5F - angle * angle / 48.0F;
    }
    if (std::fabs(sin_half_over_angle) < 1e-12F) {
        return Eigen::Vector3f::Zero();
    }
    return Eigen::Vector3f{qf[1] / sin_half_over_angle, qf[2] / sin_half_over_angle,
                            qf[3] / sin_half_over_angle};
}

// Angular velocity of row i of a quaternion sequence whose live rows are [lo, hi], with the
// np.gradient rule: central difference axis_angle(q[i+1] * q[i-1]^-1) / (2 dt) inside, one-sided at
// lo / hi. Parity with math_utils.py:so3_derivative and the training NPZ converter.
Eigen::Vector3f so3_rate(const MotionData::Base4Mat& quats, Eigen::Index i, Eigen::Index lo,
                         Eigen::Index hi, float dt) noexcept {
    if (hi <= lo) {
        return Eigen::Vector3f::Zero();
    }
    const Eigen::Index a = (i > lo) ? i - 1 : i;
    const Eigen::Index b = (i < hi) ? i + 1 : i;
    const math::Quat q_a{quats(a, 0), quats(a, 1), quats(a, 2), quats(a, 3)};
    const math::Quat q_b{quats(b, 0), quats(b, 1), quats(b, 2), quats(b, 3)};
    const math::Quat q_rel = math::quat_mul(q_b, math::quat_conjugate(q_a));
    return axis_angle_of(q_rel) / (static_cast<float>(b - a) * dt);
}

// Angular velocity from a quaternion sequence (wxyz); see so3_rate.
MotionData::Base3Mat so3_derivative(const MotionData::Base4Mat& quats, float dt) {
    const Eigen::Index n = quats.rows();
    MotionData::Base3Mat out(n, 3);
    for (Eigen::Index i = 0; i < n; ++i) {
        out.row(i) = so3_rate(quats, i, 0, n - 1, dt).transpose();
    }
    return out;
}

MotionData build_standing_motion(const MotionLoaderConfig& cfg) {
    const std::int32_t fps    = cfg.output_fps > 0 ? cfg.output_fps : 50;
    const std::int32_t frames = std::max<std::int32_t>(cfg.min_output_frames,
                                                        static_cast<std::int32_t>(10 * fps));

    MotionData data;
    data.output_fps    = fps;
    // Honor a configured input_fps so a trim of the standing pose round-trips
    // to the generator's rate; fall back to output fps (1:1).
    data.input_fps     = cfg.input_fps > 0 ? cfg.input_fps : fps;
    data.dof_poss.resize(frames, static_cast<Eigen::Index>(kG1Dof));
    data.dof_vels.resize(frames, static_cast<Eigen::Index>(kG1Dof));
    data.base_poss.resize(frames, 3);
    data.base_rots.resize(frames, 4);
    data.base_lin_vels.resize(frames, 3);
    data.base_ang_vels.resize(frames, 3);

    data.dof_poss.rowwise()  = cfg.standing_joint_pos.transpose();
    data.dof_vels.setZero();
    data.base_poss.rowwise() = cfg.standing_base_pos.transpose();
    data.base_rots.rowwise() = cfg.standing_base_quat_wxyz.transpose();
    data.base_lin_vels.setZero();
    data.base_ang_vels.setZero();
    data.commit_layout();
    return data;
}

// Extend the motion to `required_frames` (no-op if it is already that long).
// After the last clip frame the joints ramp linearly to `end_pos_train` over K
// frames and then hold it; the root position and orientation hold the last
// frame. K = compute_transition_steps(max_j |end_j - last_j|), the start-of-plan
// rule. Twin of deploy/utils/core/motion_loader.py:MotionLoader.padded, and
// written to give the same float32 values: with delta = end - last, padded
// frame i = 1, 2, ... is
//   i <  K: joints = last + delta * (float(i) / float(K)),
//           joint velocity = delta * (float(output_fps) / float(K))
//   i >= K: joints = end, joint velocity = 0
// Root linear / angular velocities are 0. A pad shorter than K ends inside the
// ramp. The products and sums are separate statements so they are rounded one
// at a time, as numpy does (no fused multiply-add; -ffast-math builds may still
// fuse them, which moves the last bit).
void pad_motion_if_needed(MotionData& data, std::int32_t required_frames,
                            const Eigen::Matrix<float, kG1Dof, 1>& end_pos_train) {
    const std::int32_t cur = data.frame_count;
    if (cur >= required_frames || required_frames <= 0 || cur <= 0) {
        return;
    }
    const std::int32_t pad = required_frames - cur;

    MotionData::JointMat dof_poss(required_frames, static_cast<Eigen::Index>(kG1Dof));
    MotionData::JointMat dof_vels(required_frames, static_cast<Eigen::Index>(kG1Dof));
    MotionData::Base3Mat base_poss(required_frames, 3);
    MotionData::Base4Mat base_rots(required_frames, 4);
    MotionData::Base3Mat base_lin_vels(required_frames, 3);
    MotionData::Base3Mat base_ang_vels(required_frames, 3);

    // Copy the live window [start_row, start_row + cur) to rows [0, cur) of
    // the freshly-allocated matrices. The leading slack rows of any prior
    // scratch are dropped (they were stale anyway).
    const Eigen::Index r0 = data.start_row;
    dof_poss.topRows(cur)      = data.dof_poss.middleRows(r0, cur);
    dof_vels.topRows(cur)      = data.dof_vels.middleRows(r0, cur);
    base_poss.topRows(cur)     = data.base_poss.middleRows(r0, cur);
    base_rots.topRows(cur)     = data.base_rots.middleRows(r0, cur);
    base_lin_vels.topRows(cur) = data.base_lin_vels.middleRows(r0, cur);
    base_ang_vels.topRows(cur) = data.base_ang_vels.middleRows(r0, cur);

    const Eigen::RowVector4f last_quat = base_rots.row(cur - 1);
    const Eigen::RowVector3f last_pos  = base_poss.row(cur - 1);
    std::array<float, kG1Dof> last{};
    std::array<float, kG1Dof> delta{};
    float                     gap = 0.0F;
    for (std::size_t j = 0; j < kG1Dof; ++j) {
        const auto jj = static_cast<Eigen::Index>(j);
        last[j]  = dof_poss(cur - 1, jj);
        delta[j] = end_pos_train[jj] - last[j];
        gap      = std::max(gap, std::fabs(delta[j]));
    }
    const std::int32_t k         = compute_transition_steps(gap);
    const float        fps_over_k = static_cast<float>(data.output_fps) / static_cast<float>(k);
    for (std::int32_t i = 1; i <= pad; ++i) {
        const Eigen::Index r = cur - 1 + i;
        if (i < k) {
            const float alpha = static_cast<float>(i) / static_cast<float>(k);
            for (std::size_t j = 0; j < kG1Dof; ++j) {
                const auto  jj   = static_cast<Eigen::Index>(j);
                const float step = delta[j] * alpha;
                dof_poss(r, jj)  = last[j] + step;
                dof_vels(r, jj)  = delta[j] * fps_over_k;
            }
        } else {
            dof_poss.row(r) = end_pos_train.transpose();
            dof_vels.row(r).setZero();
        }
        base_poss.row(r) = last_pos;
        base_rots.row(r) = last_quat;
        base_lin_vels.row(r).setZero();
        base_ang_vels.row(r).setZero();
    }

    data.dof_poss      = std::move(dof_poss);
    data.dof_vels      = std::move(dof_vels);
    data.base_poss     = std::move(base_poss);
    data.base_rots     = std::move(base_rots);
    data.base_lin_vels = std::move(base_lin_vels);
    data.base_ang_vels = std::move(base_ang_vels);
    data.commit_layout();
}

}  // namespace

void apply_joint_remap(MotionData& data, const JointRemap<kG1Dof>& remap) noexcept {
    // The Python remap is build_index_map(deployment_order, training_order)
    // and is applied as `motion_dof_poss = motion_dof_poss[:, joint_remap]`.
    // `joint_remap[i]` is the *deployment* index we want at training slot i.
    //
    // In-place over the live window [start_row, start_row + frame_count) with
    // a small per-row stack scratch. No heap.
    std::array<float, kG1Dof> tmp_pos{};
    std::array<float, kG1Dof> tmp_vel{};
    const Eigen::Index r0 = data.start_row;
    const Eigen::Index r1 = r0 + data.frame_count;
    for (Eigen::Index r = r0; r < r1; ++r) {
        for (std::size_t i = 0; i < kG1Dof; ++i) {
            tmp_pos[i] = data.dof_poss(r, static_cast<Eigen::Index>(i));
            tmp_vel[i] = data.dof_vels(r, static_cast<Eigen::Index>(i));
        }
        for (std::size_t i = 0; i < kG1Dof; ++i) {
            const auto src = static_cast<std::size_t>(remap.deploy_to_train[i]);
            data.dof_poss(r, static_cast<Eigen::Index>(i)) = tmp_pos[src];
            data.dof_vels(r, static_cast<Eigen::Index>(i)) = tmp_vel[src];
        }
    }
}

MotionData make_motion_scratch(std::int32_t capacity,
                                std::int32_t output_fps) noexcept {
    MotionData data;
    data.output_fps = output_fps;
    data.dof_poss.resize(capacity, static_cast<Eigen::Index>(kG1Dof));
    data.dof_vels.resize(capacity, static_cast<Eigen::Index>(kG1Dof));
    data.base_poss.resize(capacity, 3);
    data.base_rots.resize(capacity, 4);
    data.base_lin_vels.resize(capacity, 3);
    data.base_ang_vels.resize(capacity, 3);
    data.dof_poss.setZero();
    data.dof_vels.setZero();
    data.base_poss.setZero();
    // Default base quat: identity (wxyz = 1,0,0,0). Avoids accidental NaN
    // through ObservationBuilder's quat_difference if a slot is read before
    // its first fill.
    data.base_rots.setZero();
    data.base_rots.col(0).setConstant(1.0F);
    data.base_lin_vels.setZero();
    data.base_ang_vels.setZero();

    // Zero-loss trim master clip (deploy/xyzw, input-fps). Sized once to the max
    // input-fps plan length; starts empty (master_rows == 0 ⇒ trim resamples).
    data.master_qpos.resize(RawMotionPlan::kMaxRows,
                            static_cast<Eigen::Index>(kCsvCols));
    data.master_qpos.setZero();
    data.master_rows = 0;
    data.master_fps  = 0;

    data.scratch_capacity = capacity;
    data.start_row        = std::min<std::int32_t>(kMaxTransitionSteps, capacity);
    data.frame_count      = 0;
    return data;
}

core::Result<MotionData> load_motion(const MotionLoaderConfig& cfg) {
    using ResT = core::Result<MotionData>;

    if (cfg.output_fps <= 0) {
        return ResT::Error{"motion loader: output_fps must be > 0"};
    }

    if (cfg.csv_path.empty()) {
        auto data = build_standing_motion(cfg);
        // Standing-motion path keeps data in deployment order (no apply_joint_remap),
        // so pad with cfg.standing_joint_pos directly.
        pad_motion_if_needed(data, cfg.min_output_frames, cfg.standing_joint_pos);
        return data;
    }

    // input_fps sets the resample ratio; zero or negative would divide by zero.
    if (cfg.input_fps <= 0) {
        return ResT::Error{"motion loader: input_fps must be > 0"};
    }

    auto csv = parse_csv(cfg.csv_path);
    if (!csv) {
        return ResT::Error{csv.error()};
    }

    const RowsMat&    rows         = *csv;
    const Eigen::Index input_frames = rows.rows();
    if (input_frames < 2) {
        return ResT::Error{"motion CSV must have at least 2 frames"};
    }

    const float output_dt = 1.0F / static_cast<float>(cfg.output_fps);

    // Resample in double with a rounding-tolerant floor, so an output frame that
    // lands on an input frame copies it exactly (a 50 fps clip played at 50 Hz is
    // passed through unchanged). Same rule as Python's _resample_indices:
    // count = floor(duration * output_fps) + 1, keeping both ends of the clip.
    const double ratio = static_cast<double>(cfg.input_fps) / static_cast<double>(cfg.output_fps);
    const Eigen::Index output_frames = static_cast<Eigen::Index>(
        std::floor(static_cast<double>(input_frames - 1) / ratio + 1e-9)) + 1;

    MotionData data;
    data.output_fps = cfg.output_fps;
    data.input_fps  = cfg.input_fps;
    data.dof_poss.resize(output_frames, static_cast<Eigen::Index>(kG1Dof));
    data.dof_vels.resize(output_frames, static_cast<Eigen::Index>(kG1Dof));
    data.base_poss.resize(output_frames, 3);
    data.base_rots.resize(output_frames, 4);
    data.base_lin_vels.resize(output_frames, 3);
    data.base_ang_vels.resize(output_frames, 3);

    // Extract input columns as row-major matrices for easy indexing.
    MotionData::Base3Mat in_base_pos(input_frames, 3);
    in_base_pos = rows.block(0, 0, input_frames, 3);

    MotionData::Base4Mat in_base_quat_wxyz(input_frames, 4);
    for (Eigen::Index r = 0; r < input_frames; ++r) {
        // CSV quaternion is xyzw at columns 3..6. Reorder to wxyz = [6, 3, 4, 5].
        in_base_quat_wxyz(r, 0) = rows(r, 6);
        in_base_quat_wxyz(r, 1) = rows(r, 3);
        in_base_quat_wxyz(r, 2) = rows(r, 4);
        in_base_quat_wxyz(r, 3) = rows(r, 5);
    }

    MotionData::JointMat in_dof(input_frames, static_cast<Eigen::Index>(kG1Dof));
    in_dof = rows.block(0, 7, input_frames, static_cast<Eigen::Index>(kG1Dof));

    MotionData::Base3Mat out_base_pos(output_frames, 3);

    for (Eigen::Index i = 0; i < output_frames; ++i) {
        const double fidx = static_cast<double>(i) * ratio;
        const Eigen::Index i0 = std::min<Eigen::Index>(
            static_cast<Eigen::Index>(std::floor(fidx + 1e-9 * std::max(1.0, fidx))),
            input_frames - 1);
        const Eigen::Index i1
            = std::min<Eigen::Index>(i0 + 1, input_frames - 1);
        const float blend = static_cast<float>(
            std::clamp(fidx - static_cast<double>(i0), 0.0, 1.0));

        Eigen::Matrix<float, 1, 3>                                 bp;
        Eigen::Matrix<float, 1, static_cast<Eigen::Index>(kG1Dof)> dp;
        lerp_rows(in_base_pos, i0, i1, blend, bp);
        lerp_rows(in_dof, i0, i1, blend, dp);
        out_base_pos.row(i)  = bp;
        data.dof_poss.row(i) = dp;
        data.base_rots.row(i) = slerp_rows(in_base_quat_wxyz, i0, i1, blend).transpose();
    }

    // Velocities via np.gradient-equivalent central differences.
    data.dof_vels      = gradient_axis0(data.dof_poss, output_dt);
    data.base_lin_vels = gradient_axis0(out_base_pos, output_dt);
    data.base_ang_vels = so3_derivative(data.base_rots, output_dt);
    data.base_poss     = std::move(out_base_pos);

    // Commit the live window before any remap so apply_joint_remap iterates
    // over the populated rows.
    data.commit_layout();

    if (cfg.joint_remap != nullptr) {
        apply_joint_remap(data, *cfg.joint_remap);
    }

    // After apply_joint_remap, motion data is in *training* order. Mirror
    // Python: pad with standing_joint_pos remapped from deployment to training.
    Eigen::Matrix<float, kG1Dof, 1> standing_train = cfg.standing_joint_pos;
    if (cfg.joint_remap != nullptr) {
        for (std::size_t i = 0; i < kG1Dof; ++i) {
            standing_train[static_cast<Eigen::Index>(i)]
                = cfg.standing_joint_pos[static_cast<Eigen::Index>(
                    cfg.joint_remap->deploy_to_train[i])];
        }
    }
    pad_motion_if_needed(data, cfg.min_output_frames, standing_train);
    return data;
}

void get_time_indices(std::int32_t      counter,
                      std::int32_t      decimation,
                      double            sim_dt,
                      std::int32_t      motion_fps,
                      std::size_t       horizon,
                      std::int32_t      max_idx,
                      std::int32_t*     out,
                      std::int32_t      row_offset) noexcept {
    // Python:
    //   offsets = np.arange(horizon - 1, -1, -1)
    //   passed_times = (counter / decimation + offsets) * sim_dt * decimation
    //   time_indices = int(passed_times * motion_fps)
    //   return clip(time_indices, 0, max_idx)
    //
    // `row_offset` is added after clipping so the output indices land in
    // [row_offset, row_offset + max_idx].
    const std::int32_t dec = decimation > 0 ? decimation : 1;
    const double base
        = (static_cast<double>(counter) / static_cast<double>(dec))
          * sim_dt * static_cast<double>(dec);
    for (std::size_t k = 0; k < horizon; ++k) {
        const double offset = static_cast<double>(horizon - 1 - k);
        const double t      = base + offset * sim_dt * static_cast<double>(dec);
        // Floor with a small relative tolerance: sim_dt is usually the float
        // 0.02F (0.0199999995 as a double, ~2e-8 relative error), so t * fps
        // lands just below the integer and plain truncation would drop a frame
        // (stale reference, and a skip + freeze after every counter reset).
        // The tolerance (1e-6 relative, ~45x the float error) only absorbs
        // that rounding; non-integer frame times still floor like int().
        const double frames = t * static_cast<double>(motion_fps);
        std::int32_t idx    = static_cast<std::int32_t>(
            std::floor(frames + 1e-6 * std::max(1.0, frames)));
        if (idx < 0) idx = 0;
        if (idx > max_idx) idx = max_idx;
        out[k] = idx + row_offset;
    }
}

void align_motion_to_yaw(MotionData&                  motion,
                          const Eigen::Vector4f&        robot_quat_wxyz) noexcept {
    if (motion.frame_count == 0) {
        return;
    }

    const Eigen::Index r0 = motion.start_row;
    const Eigen::Index r1 = r0 + motion.frame_count;

    // Yaw of the robot now and yaw of the motion's first authored frame.
    // We use ZYX intrinsic Euler — same as math::quat_to_euler / Python's
    // quat_to_euler. Roll and pitch are intentionally ignored: we only want
    // to align the heading so the first command-feature anchor doesn't
    // contain a large yaw component the policy would try to "snap" out of.
    const float robot_yaw = math::quat_to_euler(robot_quat_wxyz)[2];

    const Eigen::Vector4f mfq{motion.base_rots(r0, 0), motion.base_rots(r0, 1),
                                motion.base_rots(r0, 2), motion.base_rots(r0, 3)};
    const float motion_yaw = math::quat_to_euler(mfq)[2];

    const float yaw_offset = robot_yaw - motion_yaw;
    if (std::fabs(yaw_offset) < 1.0e-3F) {
        return;  // already aligned within < 1 mrad
    }

    // Yaw rotation as a wxyz quaternion (rotation about world z by yaw_offset).
    const float           half = yaw_offset * 0.5F;
    const Eigen::Vector4f yaw_rot_wxyz{std::cos(half), 0.0F, 0.0F, std::sin(half)};
    // Pre-compute the 2D rotation for vector components in the XY plane.
    const float c = std::cos(yaw_offset);
    const float s = std::sin(yaw_offset);

    // Anchor the XY trajectory at frame 0's authored XY. Each frame's offset
    // from the anchor is rotated; frame 0 stays put, and the
    // trajectory's heading rotates with the orientation. Without this the
    // orientations rotate but the XY path stays in the authored frame, so a
    // downstream consumer that reads positions (the ghost renderer, or the
    // trim-RPC reply that ships base_poss on the wire) sees the trajectory
    // pointing the wrong way relative to the body.
    const float anchor_x = motion.base_poss(r0, 0);
    const float anchor_y = motion.base_poss(r0, 1);

    // Rotate every frame in the live window in place.
    for (Eigen::Index i = r0; i < r1; ++i) {
        // Quaternion: new_q = yaw_rot * old_q  (active rotation about world z)
        const Eigen::Vector4f old_q{motion.base_rots(i, 0), motion.base_rots(i, 1),
                                      motion.base_rots(i, 2), motion.base_rots(i, 3)};
        const Eigen::Vector4f new_q = math::quat_mul(yaw_rot_wxyz, old_q);
        motion.base_rots(i, 0) = new_q[0];
        motion.base_rots(i, 1) = new_q[1];
        motion.base_rots(i, 2) = new_q[2];
        motion.base_rots(i, 3) = new_q[3];

        // Base XY position: rotate around the anchor (frame-0 XY). Z untouched.
        // At i=r0 dx=dy=0 so the anchor is preserved exactly.
        const float dx = motion.base_poss(i, 0) - anchor_x;
        const float dy = motion.base_poss(i, 1) - anchor_y;
        motion.base_poss(i, 0) = anchor_x + dx * c - dy * s;
        motion.base_poss(i, 1) = anchor_y + dx * s + dy * c;

        // Linear velocity in world frame: rotate XY around z; Z unchanged.
        if (motion.base_lin_vels.rows() > i) {
            const float lvx = motion.base_lin_vels(i, 0);
            const float lvy = motion.base_lin_vels(i, 1);
            motion.base_lin_vels(i, 0) = lvx * c - lvy * s;
            motion.base_lin_vels(i, 1) = lvx * s + lvy * c;
        }

        // Angular velocity in world frame (so3_rate yields world-frame omega):
        // rotate XY around z; Z unchanged.
        if (motion.base_ang_vels.rows() > i) {
            const float avx = motion.base_ang_vels(i, 0);
            const float avy = motion.base_ang_vels(i, 1);
            motion.base_ang_vels(i, 0) = avx * c - avy * s;
            motion.base_ang_vels(i, 1) = avx * s + avy * c;
        }
    }
}

void align_motion_to_anchor(MotionData&            motion,
                            const Eigen::Vector3f& anchor_pos,
                            const Eigen::Vector4f& anchor_quat_wxyz) noexcept {
    if (motion.frame_count == 0) {
        return;
    }

    const Eigen::Index r0 = motion.start_row;
    const Eigen::Index r1 = r0 + motion.frame_count;

    // yaw_offset = anchor_yaw - frame0_yaw (ZYX intrinsic euler, matching
    // math::quat_to_euler / Python quat_to_euler).
    const float           anchor_yaw = math::quat_to_euler(anchor_quat_wxyz)[2];
    const Eigen::Vector4f mfq{motion.base_rots(r0, 0), motion.base_rots(r0, 1),
                                motion.base_rots(r0, 2), motion.base_rots(r0, 3)};
    const float           motion_yaw = math::quat_to_euler(mfq)[2];
    const float           yaw_offset = anchor_yaw - motion_yaw;

    // Frame-0 world pose, captured before the loop overwrites it.
    const float f0x = motion.base_poss(r0, 0);
    const float f0y = motion.base_poss(r0, 1);
    const float f0z = motion.base_poss(r0, 2);

    const float dxa = anchor_pos[0] - f0x;
    const float dya = anchor_pos[1] - f0y;
    const float dza = anchor_pos[2] - f0z;

    const bool  rotate     = std::fabs(yaw_offset) >= 1.0e-3F;
    const float xyz_delta2 = dxa * dxa + dya * dya + dza * dza;
    if (!rotate && xyz_delta2 < (1.0e-4F * 1.0e-4F)) {
        return;  // within Python's dual guard (yaw < 1 mrad and ‖Δ‖ < 0.1 mm)
    }

    const float           c    = std::cos(yaw_offset);
    const float           s    = std::sin(yaw_offset);
    const float           half = 0.5F * yaw_offset;
    const Eigen::Vector4f yaw_rot{std::cos(half), 0.0F, 0.0F, std::sin(half)};

    const float ax = anchor_pos[0];
    const float ay = anchor_pos[1];
    const float az = anchor_pos[2];

    for (Eigen::Index i = r0; i < r1; ++i) {
        if (rotate) {
            // Quaternion: new_q = yaw_rot * old_q (active rotation about world z).
            const Eigen::Vector4f oq{motion.base_rots(i, 0), motion.base_rots(i, 1),
                                       motion.base_rots(i, 2), motion.base_rots(i, 3)};
            Eigen::Vector4f nq = math::quat_mul(yaw_rot, oq);
            nq.normalize();  // quat_mul drifts off unit norm; renormalize each resend.
            motion.base_rots(i, 0) = nq[0];
            motion.base_rots(i, 1) = nq[1];
            motion.base_rots(i, 2) = nq[2];
            motion.base_rots(i, 3) = nq[3];
        }

        // Base XY: rotate the offset-from-frame0 by yaw_offset, then translate so
        // frame 0 lands at the anchor XY. Z: pure shift so frame 0 lands at anchor
        // Z (yaw rotation about world z leaves Z invariant).
        const float ox = motion.base_poss(i, 0) - f0x;
        const float oy = motion.base_poss(i, 1) - f0y;
        motion.base_poss(i, 0) = ax + (ox * c - oy * s);
        motion.base_poss(i, 1) = ay + (ox * s + oy * c);
        motion.base_poss(i, 2) = az + (motion.base_poss(i, 2) - f0z);

        if (rotate) {
            // Velocities are recomputed-after-align in Python; here derivatives
            // ran BEFORE align, so rotate the world-frame XY (Z untouched) — the
            // affine SE(2)+Z transform's time-derivative equals R(yaw)·v.
            if (motion.base_lin_vels.rows() > i) {
                const float lvx = motion.base_lin_vels(i, 0);
                const float lvy = motion.base_lin_vels(i, 1);
                motion.base_lin_vels(i, 0) = lvx * c - lvy * s;
                motion.base_lin_vels(i, 1) = lvx * s + lvy * c;
            }
            if (motion.base_ang_vels.rows() > i) {
                const float avx = motion.base_ang_vels(i, 0);
                const float avy = motion.base_ang_vels(i, 1);
                motion.base_ang_vels(i, 0) = avx * c - avy * s;
                motion.base_ang_vels(i, 1) = avx * s + avy * c;
            }
        }
    }
}

std::int32_t compute_transition_steps(float        error,
                                      std::int32_t min_steps,
                                      std::int32_t max_steps,
                                      float        gap_threshold) noexcept {
    // float32 throughout, in the Python operation order: (error / gap) * min.
    if (!std::isfinite(error)) {
        return max_steps;
    }
    if (error < gap_threshold) {
        return min_steps;
    }
    const auto steps = static_cast<std::int32_t>(error / gap_threshold
                                                 * static_cast<float>(min_steps));
    return std::min(std::max(steps, min_steps), max_steps);
}

bool commit_trim_cut(MotionData& motion, std::int32_t cut) noexcept {
    bool moved = false;
    // Window exhausted (the server clamps the cut to frame_count when the generator stalled past
    // the window's end): keep the HELD final frame -- the trim reply was built from it -- instead of
    // leaving the window untouched, which made the reset counter jump back to the window start.
    if (cut >= motion.frame_count && motion.frame_count > 0) {
        cut = motion.frame_count - 1;
    }
    if (cut > 0 && cut < motion.frame_count) {
        motion.start_row   += cut;
        motion.frame_count -= cut;
        // keep the master aligned with the new live-window start (exact, fractional)
        if (motion.master_rows > 0 && motion.output_fps > 0) {
            motion.master_offset += static_cast<double>(cut)
                * static_cast<double>(motion.master_fps)
                / static_cast<double>(motion.output_fps);
        }
        moved = true;
    }
    // The trim reply sent whole stored frames from master_trim_start(mf), mf = master_offset now;
    // the next install shifts its upsample grid by the gap (send_ref_motion).
    if (motion.master_rows > 0) {
        motion.trim_phase = motion.master_offset - master_trim_start(motion.master_offset);
    }
    return moved || cut == 0;
}

RawMotionPlan trim_motion(const MotionData&         motion,
                            std::int32_t              current_tick,
                            std::int32_t              target_input_fps,
                            const JointRemap<kG1Dof>& remap) noexcept {
    RawMotionPlan plan{};
    plan.n_rows       = 0;
    plan.input_fps    = target_input_fps > 0 ? target_input_fps : motion.input_fps;
    plan.flags        = kFlagsDefault;

    // Bound check: nothing to trim if motion is empty, current_tick is past
    // the live window, or the output fps is unusable.
    if (motion.frame_count <= 0 || motion.output_fps <= 0
        || plan.input_fps <= 0) {
        return plan;
    }
    const std::int32_t tick = std::max<std::int32_t>(0, current_tick);
    const std::int32_t n_full = motion.frame_count - tick;
    if (n_full <= 0) {
        return plan;
    }

    const Eigen::Index r0 = static_cast<Eigen::Index>(motion.start_row + tick);

    // Zero-loss path: if a pristine input-fps master clip is present and matches
    // the requested fps, return stored frames of it — never re-downsample
    // (parity with G1Controller.trim_ref_motion's master path).
    //   mf = master_offset + tick * master_fps / output_fps (exact master row of
    //   the playback point), tick = ticks into the live window. The prefix
    //   starts at the NEAREST whole stored frame; the fractional gap is
    //   re-applied by the next install (MotionData::trim_phase).
    if (motion.master_rows > 0 && motion.master_fps == plan.input_fps) {
        const double mf = motion.master_offset
            + static_cast<double>(tick) * static_cast<double>(motion.master_fps)
              / static_cast<double>(motion.output_fps);
        plan.n_rows = master_interp_to_plan(motion.master_qpos, motion.master_rows,
                                            master_trim_start(mf), plan.data.data());
        return plan;
    }

    // Fallback: downsample the output_fps live window to target_input_fps with
    // matched LERP/SLERP + yaw-only re-gauge (parity with
    // G1Controller.trim_ref_motion's fallback).
    plan.n_rows = downsample_window_to_plan(
        motion.base_poss, motion.base_rots, motion.dof_poss,
        r0, n_full, motion.output_fps, plan.input_fps, remap, plan.data.data());
    return plan;
}

// trim_ref_motion_from_snapshot — same math as trim_motion(MotionData,...) above,
// but the source rows live in g1::ActivePlanSnapshot's fixed-size matrices and
// the live window is normalized to start at row 0 (snapshot is published with
// start_row=0, so `current_idx` indexes directly into [0, snap.frame_count)).
RawMotionPlan trim_ref_motion_from_snapshot(const g1::ActivePlanSnapshot& snap,
                                             std::int32_t                  current_idx,
                                             std::int32_t                  target_input_fps,
                                             const JointRemap<kG1Dof>&     remap) noexcept {
    RawMotionPlan plan{};
    plan.n_rows    = 0;
    plan.input_fps = target_input_fps > 0 ? target_input_fps : snap.input_fps;
    plan.flags     = kFlagsDefault;

    if (snap.epoch == 0 || snap.frame_count <= 0 || snap.output_fps <= 0
        || plan.input_fps <= 0) {
        return plan;
    }
    // A truncated snapshot (source motion longer than the copy, with a tail that
    // does not just hold the last row) has no data at or past its end: clamping
    // would answer with a stale prefix from before the robot's real position.
    if (!snap.tail_held && current_idx >= snap.frame_count) {
        return plan;
    }
    // Clamp current_idx into [0, frame_count - 1] so a robot that has held
    // the snapshot's last frame past the end (e.g. standing-pose elapsing
    // while a slow generator hasn't sent plan #1) still gets a valid prefix —
    // the held last frame, not an empty plan. Mirrors Python G1Controller
    // .trim_ref_motion: `cur = min(self.counter, max(0, n_frames - 1))`.
    // An empty trim would leave a generator that conditions on the trimmed
    // prefix with nothing to continue from.
    const std::int32_t idx = std::min<std::int32_t>(
        std::max<std::int32_t>(0, current_idx),
        std::max<std::int32_t>(0, snap.frame_count - 1));
    const std::int32_t n_full = snap.frame_count - idx;

    const Eigen::Index r0 = static_cast<Eigen::Index>(idx);

    // Zero-loss path: stored frames of the snapshot's pristine input-fps master
    // clip when present and fps-matched (parity with g1_controller master path),
    // from the nearest whole frame to mf (same rule as trim_motion above).
    if (snap.master_rows > 0 && snap.master_fps == plan.input_fps) {
        const double mf = snap.master_offset
            + static_cast<double>(idx) * static_cast<double>(snap.master_fps)
              / static_cast<double>(snap.output_fps);
        plan.n_rows = master_interp_to_plan(snap.master_qpos, snap.master_rows,
                                            master_trim_start(mf), plan.data.data());
        return plan;
    }

    // Fallback: downsample the snapshot's output_fps window to target_input_fps
    // (matched LERP/SLERP + yaw-only re-gauge).
    plan.n_rows = downsample_window_to_plan(
        snap.base_poss, snap.base_rots, snap.dof_poss,
        r0, n_full, snap.output_fps, plan.input_fps, remap, plan.data.data());
    return plan;
}

}  // namespace rt_deploy::policy
