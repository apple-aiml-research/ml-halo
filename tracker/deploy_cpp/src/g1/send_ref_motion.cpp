//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

// Implementation of g1::send_ref_motion (Python G1Controller.send_ref_motion).
// See the header for the contract. Allocation-free, noexcept.

#include "rt_deploy/g1/send_ref_motion.hpp"

#include "rt_deploy/policy/motion_processing.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace rt_deploy::g1 {

SwapResult send_ref_motion(const policy::RawMotionPlan&         raw,
                             const policy::MotionData*            previous,
                             const math::RobotState<kDof>&        state,
                             std::int32_t                         inference_tick,
                             policy::MotionData&                  scratch,
                             const policy::JointRemap<kDof>&      remap) noexcept {
    SwapResult result{};
    result.raw_n_rows    = raw.n_rows;
    result.raw_input_fps = raw.input_fps;

    // 0. Header bounds. Every later step indexes raw.data and the master clip
    //    (both kMaxRows long) by n_rows, so a plan built by a producer that did
    //    not check it must be refused before anything is read or written.
    if (raw.n_rows < 1 || raw.n_rows > policy::RawMotionPlan::kMaxRows) {
        std::fprintf(stderr,
                     "[control_loop] inbox plan rejected: n_rows=%d outside 1..%d\n",
                     raw.n_rows, policy::RawMotionPlan::kMaxRows);
        result.rejection = SwapRejection::kInvalidHeader;
        return result;
    }
    // A NaN / inf anywhere in the plan would reach the observation and hold the policy off for the whole
    // window it covers: refuse it and keep the current reference.
    const std::size_t n_vals = static_cast<std::size_t>(raw.n_rows)
                               * static_cast<std::size_t>(policy::RawMotionPlan::kCols);
    for (std::size_t i = 0; i < n_vals; ++i) {
        if (!std::isfinite(raw.data[i])) {
            std::fprintf(stderr, "[control_loop] inbox plan rejected: non-finite value at row %zu\n",
                         i / static_cast<std::size_t>(policy::RawMotionPlan::kCols));
            result.rejection = SwapRejection::kNonFinite;
            return result;
        }
    }

    // 1. Capture the PREVIOUS reference's pose at the current tick BEFORE
    //    resample overwrites the scratch — on the 2nd+ swap the old reference
    //    IS the scratch we're about to write, so this read must happen first.
    //    SE(2)+Z anchor for align_anchor="reference" parity with Python;
    //    k ↔ self.counter == inference_tick.
    bool            have_ref_anchor = false;
    Eigen::Vector3f ref_anchor_pos  = Eigen::Vector3f::Zero();
    Eigen::Vector4f ref_anchor_quat{1.0F, 0.0F, 0.0F, 0.0F};
    if (previous != nullptr && previous->frame_count > 0) {
        const std::int32_t k = std::min<std::int32_t>(
            inference_tick, previous->frame_count - 1);
        const Eigen::Index kr =
            static_cast<Eigen::Index>(previous->start_row + k);
        ref_anchor_pos = Eigen::Vector3f{previous->base_poss(kr, 0),
                                          previous->base_poss(kr, 1),
                                          previous->base_poss(kr, 2)};
        ref_anchor_quat = Eigen::Vector4f{previous->base_rots(kr, 0),
                                           previous->base_rots(kr, 1),
                                           previous->base_rots(kr, 2),
                                           previous->base_rots(kr, 3)};
        have_ref_anchor = true;
    }

    // 1b. Exact timing: the last trim sent WHOLE stored frames, starting at the
    //     stored frame nearest the true playback point, which lies `trim_phase`
    //     (in [-0.5, 0.5]) master rows from it. Read it before the resample
    //     (previous may BE the scratch). The new plan's row 0 = that whole
    //     frame, so its 50 Hz grid is shifted by the phase: it then plays
    //     exactly where the old plan would have. Only for auto-start-offset
    //     plans built from that trim, at the same input fps.
    const bool auto_offset =
        (raw.flags & policy::kFlagAutoStartOffset) != 0U;
    double phase = 0.0;
    if (auto_offset && previous != nullptr && previous->master_rows > 0
        && previous->master_fps == raw.input_fps && std::isfinite(previous->trim_phase)) {
        phase = previous->trim_phase;
    }

    // 1c. Validate BEFORE writing: after the first install `previous` IS the scratch
    //     (the live reference), so a plan rejected after the resample below would
    //     already have overwritten what the robot is tracking. resample_output_frames
    //     is the exact row count resample_motion will produce (same phase, same cap).
    const std::int32_t start_offset =
        auto_offset ? inference_tick : 0;
    const std::int32_t n_out = policy::resample_output_frames(
        raw.n_rows, raw.input_fps, scratch.output_fps, scratch.scratch_capacity, phase);
    if (n_out <= 0) {
        std::fprintf(stderr,
                     "[control_loop] inbox plan rejected: "
                     "scratch capacity %d too small or invalid header "
                     "(n_rows=%d input_fps=%d)\n",
                     scratch.scratch_capacity, raw.n_rows, raw.input_fps);
        result.rejection = SwapRejection::kResampleFailed;
        return result;
    }
    if (start_offset >= n_out) {
        std::fprintf(stderr,
                     "[control_loop] inbox plan rejected: "
                     "auto-slice inference_tick=%d exceeds chunk "
                     "frame_count=%d (chunk arrived too late)\n",
                     start_offset, n_out);
        result.rejection = SwapRejection::kStartOffsetExceedsLen;
        return result;
    }

    // 2. Process raw qpos → MotionData scratch with the FULL input. Slicing
    //    happens in output-fps space below.
    const bool resampled = policy::resample_motion(
        raw.data.data(), raw.n_rows, raw.input_fps,
        scratch.output_fps, scratch, phase);
    if (!resampled) {   // not reached: n_out > 0 above implies success
        result.rejection = SwapRejection::kResampleFailed;
        return result;
    }
    scratch.trim_phase = 0.0;      // consumed; a new trim sets it again

    // 3. Joint remap on the full pre-slice window so post-slice rows are
    //    already in training order.
    policy::apply_joint_remap(scratch, remap);

    // 4. Slice in OUTPUT-fps space — equivalent to Python qpos_data[counter:].
    //    Only the live-window descriptors move; the scratch matrices keep
    //    their values.
    if (start_offset > 0) {
        scratch.start_row   += start_offset;
        scratch.frame_count -= start_offset;
    }

    // 5. Zero-loss master: keep the pristine input-fps clip so trim returns
    //    stored frames instead of re-downsampling the interpolated playback.
    //    Live frame 0 plays the exact
    //    (fractional) input row m_exact = phase + start_offset * input_fps /
    //    output_fps. Drop the whole input rows before it (floor) and keep the
    //    remainder as master_offset, so every later trim knows the exact
    //    playback point (no per-install rounding).
    {
        const std::int32_t in_fps = raw.input_fps;
        const double m_exact = phase + static_cast<double>(start_offset)
                               * static_cast<double>(in_fps)
                               / static_cast<double>(scratch.output_fps);
        const std::int32_t m_drop =
            std::max<std::int32_t>(0, static_cast<std::int32_t>(std::floor(m_exact)));
        const std::int32_t md    = std::clamp(m_drop, 0, raw.n_rows);
        scratch.master_offset    = m_exact - static_cast<double>(md);
        const std::int32_t mrows = raw.n_rows - md;
        for (std::int32_t r = 0; r < mrows; ++r) {
            const double* src = raw.data.data()
                + static_cast<std::size_t>(md + r)
                  * static_cast<std::size_t>(policy::RawMotionPlan::kCols);
            for (std::int32_t c = 0;
                 c < policy::RawMotionPlan::kCols; ++c) {
                scratch.master_qpos(r, c) = static_cast<float>(src[c]);
            }
        }
        scratch.master_rows = mrows;
        scratch.master_fps  = in_fps;
    }

    // 6. Compute velocities AFTER the slice so the post-slice frame 0's
    //    lin/ang_vels use one-sided forward diff against frame 1 — matching
    //    Python np.gradient order.
    policy::compute_motion_derivatives(scratch);

    // 7. Rebase the chunk onto an anchor. Mirrors Python G1Controller
    //    .send_ref_motion's `align_anchor` kwarg via the wire flag
    //    kFlagRobotAnchor (1<<2):
    //      - kFlagRobotAnchor SET (align_anchor="robot"): anchor on the
    //        robot's CURRENT pose. Use for auto-stationary "hold" plans that
    //        re-ground the reference onto where the robot actually is, not
    //        where the (drifted) prior reference was.
    //      - kFlagRobotAnchor CLEAR (align_anchor="reference", the default):
    //        anchor on the prior reference's pose at this tick when one
    //        exists, fall back to the robot odom pose for cold starts.
    //    SE(2)+Z: rotate + translate so the chunk's post-slice frame 0 lands
    //    at the anchor. The kFlagAlignYaw (1<<0) wire bit is not consulted
    //    here (kept defined for wire compatibility).
    const bool robot_anchor_requested =
        (raw.flags & policy::kFlagRobotAnchor) != 0U;
    if (have_ref_anchor && !robot_anchor_requested) {
        // Mirror of the Python "[align] REFERENCE anchor: …" log line.
        // Reports the splice pose used (motion_base_poss[k] of the previous
        // reference at the current inference_tick) and how far it sits from
        // the robot's odom pose; a small |ref-robot| means the splice is
        // continuous, a large value means we're carrying drift forward.
        const Eigen::Vector3f robot_xy = state.odom_valid
            ? state.odom_position
            : Eigen::Vector3f{0.0F, 0.0F, 0.0F};
        const float dx = ref_anchor_pos[0] - robot_xy[0];
        const float dy = ref_anchor_pos[1] - robot_xy[1];
        const float dist_mm =
            std::sqrt(dx * dx + dy * dy) * 1000.0F;
        std::fprintf(stderr,
                     "[align] REFERENCE anchor: inference_tick=%d  "
                     "k=%d/%d  ref_anchor_xy=(%+.3f,%+.3f)  "
                     "robot_xy=(%+.3f,%+.3f)  |ref-robot|=%.0fmm\n",
                     inference_tick,
                     std::min<std::int32_t>(inference_tick,
                                            previous->frame_count - 1),
                     previous->frame_count - 1,
                     static_cast<double>(ref_anchor_pos[0]),
                     static_cast<double>(ref_anchor_pos[1]),
                     static_cast<double>(robot_xy[0]),
                     static_cast<double>(robot_xy[1]),
                     static_cast<double>(dist_mm));
        policy::align_motion_to_anchor(
            scratch, ref_anchor_pos, ref_anchor_quat);
    } else {
        const Eigen::Vector3f robot_pos = state.odom_valid
            ? state.odom_position
            : Eigen::Vector3f{0.0F, 0.0F,
                              scratch.base_poss(scratch.start_row, 2)};
        const Eigen::Vector4f robot_quat = state.odom_valid
            ? state.odom_quaternion_wxyz
            : state.imu_quat_wxyz;
        // Mirror of the Python "[align] ROBOT anchor …" log line.
        // Two reasons we land here: (a) generator set kFlagRobotAnchor (e.g.
        // auto-stationary "hold" plans), or (b) cold start with no previous
        // reference — `have_ref_anchor` is false on the very first install.
        const char* reason = robot_anchor_requested
            ? "kFlagRobotAnchor"
            : "no previous reference (cold start)";
        std::fprintf(stderr,
                     "[align] ROBOT anchor: inference_tick=%d  "
                     "robot_xy=(%+.3f,%+.3f)  reason=%s\n",
                     inference_tick,
                     static_cast<double>(robot_pos[0]),
                     static_cast<double>(robot_pos[1]),
                     reason);
        policy::align_motion_to_anchor(scratch, robot_pos, robot_quat);
    }

    // 8. Swap-error diagnostic: distance of the chunk's post-slice frame 0
    //    from the robot's current state. Caller pushes these into ring
    //    vectors.
    const Eigen::Index r0 = scratch.start_row;
    Eigen::Matrix<float, static_cast<Eigen::Index>(kDof), 1> state_train;
    for (std::size_t i = 0; i < kDof; ++i) {
        const Eigen::Index src = static_cast<Eigen::Index>(remap.deploy_to_train[i]);
        state_train[static_cast<Eigen::Index>(i)] = state.joint_positions[src];
    }
    const Eigen::Matrix<float, static_cast<Eigen::Index>(kDof), 1>
        chunk_train = scratch.dof_poss.row(r0).transpose();
    const Eigen::Matrix<float, static_cast<Eigen::Index>(kDof), 1>
        jdiff = state_train - chunk_train;
    result.metrics.joint_mean = jdiff.cwiseAbs().mean();
    result.metrics.joint_l2   = jdiff.norm();
    const Eigen::Vector3f chunk_base{
        scratch.base_poss(r0, 0), scratch.base_poss(r0, 1),
        scratch.base_poss(r0, 2)};
    const Eigen::Vector3f robot_base = state.odom_valid
        ? state.odom_position
        : Eigen::Vector3f{0.0F, 0.0F, chunk_base[2]};
    result.metrics.base_xy =
        (chunk_base.head<2>() - robot_base.head<2>()).norm();
    result.metrics.base_z = std::fabs(chunk_base[2] - robot_base[2]);

    result.installed = true;
    return result;
}

}  // namespace rt_deploy::g1
