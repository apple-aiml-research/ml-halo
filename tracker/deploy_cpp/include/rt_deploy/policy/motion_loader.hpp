//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// MotionLoader — CSV motion data loader with SLERP/LERP resampling.
//
// Parity with deploy/utils/core/motion_loader.py:MotionLoader.
//
// CSV format (36 columns per row):
//   [0:3]   xyz base position (m)
//   [3:7]   base orientation quaternion in xyzw order (converted to wxyz at load)
//   [7:36]  29 joint positions in deployment order (radians)
//
// The loader:
//   1) parses CSV to (input_frames, 36) float32
//   2) reorders quaternion to wxyz
//   3) resamples from input_fps to output_fps using LERP (positions/joints)
//      and SLERP (base quat). output_fps = round(1 / control_dt).
//   4) applies joint remap (deployment → training order) to DOF columns
//   5) computes velocities via np.gradient-equivalent central differences on
//      positions; angular velocity via the axis-angle central difference
//      (parity with math_utils.py:so3_derivative).
//
// After load, five aligned sequences are available:
//   motion_dof_poss   : (F, 29) in training order
//   motion_dof_vels   : (F, 29) in training order
//   motion_base_rots  : (F, 4)  wxyz
//   motion_base_lin_vels : (F, 3)
//   motion_base_ang_vels : (F, 3)
//
// `get_time_indices` matches the Python helper of the same name in
// deploy/utils/core/motion_loader.py.
//
// Scratch / live-window model
// ---------------------------
// `MotionData` carries three integer fields — `start_row`, `frame_count`,
// `scratch_capacity` — that describe a *window* into the underlying Eigen
// matrices. Readers (observation builder, ghost render, etc.) only ever look
// at rows [start_row, start_row + frame_count); rows outside that window are
// stale scratch space.
//
// For CSV-loaded and standing motions, the loader calls `commit_layout()`
// which sets `start_row = 0` and `frame_count = scratch_capacity = nrows`, so
// the window is the entire matrix.
//
// For runtime-pushed plans, `make_motion_scratch(capacity, fps)` pre-allocates
// a fixed-size scratch (`capacity` rows) once at startup with
// `start_row = kMaxTransitionSteps`. The control thread resamples an incoming
// plan into rows [start_row, start_row + N) (see motion_processing.hpp); the
// leading kMaxTransitionSteps rows are reserved and never part of a window.

#include "rt_deploy/core/expected.hpp"
#include "rt_deploy/policy/joint_remap.hpp"
#include "rt_deploy/policy/raw_motion_plan.hpp"

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace rt_deploy::policy {

constexpr std::size_t kG1Dof = 29;
constexpr std::size_t kCsvCols = 3 + 4 + kG1Dof;  // xyz + quat_xyzw + joints

// Leading rows reserved in every plan scratch (the start_row of a fresh plan).
// Streamed plans are installed without a blend (continuity at the seam is the
// generator's job, as in the Python deployment), so nothing is written there;
// the value is part of the scratch sizing (see ActivePlanSnapshot).
constexpr std::int32_t kMaxTransitionSteps = 50;

struct MotionLoaderConfig {
    std::string                                   csv_path;
    int                                           input_fps   = 30;
    int                                           output_fps  = 50;
    const JointRemap<kG1Dof>*                     joint_remap = nullptr;
    // Optional standing pose used when csv_path is empty (static motion).
    Eigen::Matrix<float, kG1Dof, 1>               standing_joint_pos{
        Eigen::Matrix<float, kG1Dof, 1>::Zero()};
    Eigen::Vector3f                               standing_base_pos{0.0F, 0.0F, 0.76F};
    Eigen::Vector4f                               standing_base_quat_wxyz{1.0F, 0.0F, 0.0F, 0.0F};
    // If > 0, pad the loaded motion to at least this many output frames: a ramp
    // from the last clip frame to standing_joint_pos, then the standing pose
    // held (see compute_transition_steps for the ramp length).
    std::int32_t                                  min_output_frames = 0;
};

struct MotionData {
    using JointMat = Eigen::Matrix<float, Eigen::Dynamic, static_cast<Eigen::Index>(kG1Dof),
                                    Eigen::RowMajor>;
    using Base3Mat = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
    using Base4Mat = Eigen::Matrix<float, Eigen::Dynamic, 4, Eigen::RowMajor>;
    // Pristine input-fps master clip: rows are the raw wire layout
    // [xyz(3), quat_xyzw(4), joints_deploy(29)] at `master_fps`. Capacity is
    // RawMotionPlan::kMaxRows (the max input-fps plan length).
    using MasterMat = Eigen::Matrix<float, Eigen::Dynamic,
                                     static_cast<Eigen::Index>(kCsvCols), Eigen::RowMajor>;

    JointMat dof_poss;        // (scratch_capacity, 29) training order
    JointMat dof_vels;        // (scratch_capacity, 29)
    Base3Mat base_poss;       // (scratch_capacity, 3) xyz world position
    Base4Mat base_rots;       // (scratch_capacity, 4) wxyz
    Base3Mat base_lin_vels;   // (scratch_capacity, 3)
    Base3Mat base_ang_vels;   // (scratch_capacity, 3)

    // Live-window: readers access rows [start_row, start_row + frame_count).
    std::int32_t start_row        = 0;
    std::int32_t frame_count      = 0;
    std::int32_t scratch_capacity = 0;

    std::int32_t output_fps  = 0;
    // Original input fps (CSV load and standing: cfg.input_fps, or output_fps
    // when unset; motion inbox: raw->input_fps). `trim_motion` round-trips back to
    // this fps so the recovered RawMotionPlan can re-enter the same pipeline
    // without an extra resample step.
    std::int32_t input_fps   = 0;

    // Zero-loss trim source: the pristine input-fps clip of the last swap, so
    // `trim_motion` returns exact slices of it instead of re-downsampling the
    // interpolated output-fps window. `master_rows == 0` ⇒ no master, trim falls
    // back to the resampling path.
    MasterMat    master_qpos;     // (kMaxRows, 36) deploy-order / xyzw at master_fps
    std::int32_t master_rows = 0;
    std::int32_t master_fps  = 0;
    // Exact time alignment: the master row (fractional, at master_fps) that the
    // live window's first row (start_row) corresponds to. Playback row
    // start_row+k plays master row master_offset + k*master_fps/output_fps.
    // Kept fractional instead of rounding to whole input frames. Advanced by
    // trim-commit cuts.
    double       master_offset = 0.0;
    // Set by the last trim commit: the trim prefix starts at the WHOLE master
    // row master_trim_start(mf) and the true playback point mf is `trim_phase`
    // (in [-0.5, 0.5]) master rows from it. The next install shifts its 50 Hz
    // upsample grid by this amount, so the new plan plays exactly where the old
    // one would have been (no seam error). Reset to 0 on install.
    double       trim_phase    = 0.0;

    [[nodiscard]] std::int32_t num_frames() const noexcept { return frame_count; }
    [[nodiscard]] std::int32_t max_index()  const noexcept {
        return frame_count > 0 ? frame_count - 1 : 0;
    }

    // Set start_row=0, frame_count=scratch_capacity=dof_poss.rows().
    // Call after populating the matrices in the "full motion" pattern
    // (CSV load, standing pose, inline builders) so num_frames() reports
    // the expected row count.
    void commit_layout() noexcept {
        start_row        = 0;
        frame_count      = static_cast<std::int32_t>(dof_poss.rows());
        scratch_capacity = frame_count;
    }
};

[[nodiscard]] core::Result<MotionData> load_motion(const MotionLoaderConfig& cfg);

// Pre-allocate a MotionData scratch with `capacity` rows and
// `start_row = kMaxTransitionSteps`. All matrices are sized once; the live
// window starts empty (frame_count = 0).
//
// Caller is responsible for filling rows [start_row, start_row + frame_count)
// and updating `frame_count` to match. `output_fps` is stored as-is.
[[nodiscard]] MotionData make_motion_scratch(std::int32_t capacity,
                                              std::int32_t output_fps) noexcept;

// Time-indices generator mirroring motion_loader.py:get_time_indices.
// Writes `horizon` indices into `out` (oldest-future first, current-frame last).
// `sim_dt` is double so callers passing `1/motion_fps`-style ratios match the
// Python float64 truncation behavior (float32 representation of 0.02 rounds
// downward, breaking int(time * fps) at integer-aligned boundaries).
//
// `row_offset` is added to every output index, so callers can pass a motion's
// `start_row` to keep dof_poss/.../base_rots reads unchanged regardless of
// whether the motion lives at row 0 (CSV) or deeper into a plan scratch.
void get_time_indices(std::int32_t      counter,
                      std::int32_t      decimation,
                      double            sim_dt,
                      std::int32_t      motion_fps,
                      std::size_t       horizon,
                      std::int32_t      max_idx,
                      std::int32_t*     out,
                      std::int32_t      row_offset = 0) noexcept;

// Apply the deployment→training joint remap to the motion's joint columns
// (`dof_poss`, `dof_vels`). After this call, motion.dof_*[r, i_train] holds
// the value originally stored at deployment slot `remap.deploy_to_train[i]`.
//
// Operates IN PLACE on rows [start_row, start_row + frame_count); allocates
// nothing on the heap (uses a fixed 2*kG1Dof stack scratch per row).
//
// `load_motion` already runs this on the CSV path; the helper is exposed for
// callers that build a MotionData by other means (e.g. the standing-pose
// motion synthesised in motion_switcher) and need the same training-order
// invariant the observation builder expects.
void apply_joint_remap(MotionData& data, const JointRemap<kG1Dof>& remap) noexcept;

// Yaw-align an already-loaded motion to the robot's current orientation (the
// yaw-only counterpart of align_motion_to_anchor). Without it the policy
// entering kControl sees a large motion_anchor_ori component along the yaw
// axis (the difference between the motion's authored frame-0 yaw and the
// robot's actual world-frame yaw), and commands an abrupt body rotation to
// "track" the motion's reference frame.
//
// Rotates in-place over the live window [start_row, start_row + frame_count):
//   - every base_rots quaternion (left-multiplied by yaw_rot)
//   - every base_poss XY offset from frame 0 (rotated about frame 0's XY)
//   - every base_lin_vels XY pair  (rotated by yaw_offset around z)
//   - every base_ang_vels XY pair  (same)
// dof_poss / dof_vels are joint-space and unaffected.
//
// No-op when |yaw_offset| < 1 mrad. Idempotent: calling twice with the same
// robot quaternion does the work once and then becomes a no-op.
void align_motion_to_yaw(MotionData&                   motion,
                          const Eigen::Vector4f&        robot_quat_wxyz) noexcept;

// SE(2)+Z rebase: align a motion so its live-window frame 0 lands at an anchor
// pose. Parity with deploy/g1/g1_controller.py:align_reference_motion.
// The tracker uses this on every swap to pin the new chunk's
// frame 0 onto the PREVIOUS reference's pose at the current tick (continuation /
// replan continuity) — or, when no prior reference exists, onto the robot's
// current odom pose (cold start).
//
// Rotates the whole live window about world-z by `yaw_offset = anchor_yaw -
// frame0_yaw`, translates so frame 0's XY lands at the anchor XY, and shifts Z so
// frame 0's Z lands at the anchor Z (other frames keep their authored deltas):
//   - base_rots   : left-multiplied by the yaw quaternion (renormalized)
//   - base_poss   : XY rotated about frame 0 then translated to anchor; Z shifted
//   - base_lin_vels / base_ang_vels : XY rotated by yaw_offset (Z untouched), so
//     a derivatives-before-align caller matches Python's recompute-after-align
// dof_poss / dof_vels are joint-space and unaffected.
//
// Early-out (no-op) when |yaw_offset| < 1 mrad AND ‖anchor − frame0‖ < 0.1 mm,
// matching Python's dual guard. The quaternion/velocity rotation is skipped when
// |yaw_offset| < 1 mrad (pure translation), but the XY/Z translation still runs.
void align_motion_to_anchor(MotionData&            motion,
                            const Eigen::Vector3f& anchor_pos,
                            const Eigen::Vector4f& anchor_quat_wxyz) noexcept;

// Blend length, in frames, for a pose gap of `error` radians: `min_steps` below
// `gap_threshold`, else trunc(error / gap_threshold * min_steps) clamped to
// [min_steps, max_steps], evaluated in float32 in that order; a non-finite gap
// gets `max_steps`. Twin of deploy/utils/core/motion_loader.py:transition_steps
// (the rule of the Python start-of-plan transition, same defaults), so both
// deployments pick the same K. Used for the end-of-clip ramp to the standing
// pose (MotionLoaderConfig::min_output_frames), with error = the largest joint
// gap between the last clip frame and the standing pose.
[[nodiscard]] std::int32_t compute_transition_steps(float        error,
                                                    std::int32_t min_steps     = 10,
                                                    std::int32_t max_steps     = 100,
                                                    float        gap_threshold = 0.3F) noexcept;

// Trim the live window to [current_tick, frame_count) and emit the result as
// a RawMotionPlan in deployment-order/xyzw wire format.
//
// Re-gauges by frame 0: position recentred so frame 0 sits at the origin, only
// frame 0's yaw removed from the orientation (roll / pitch kept), downsamples
// output_fps -> target_input_fps via linspace index
// selection, converts quat wxyz->xyzw and joints train->deployment order.
// Returns n_rows == 0 when the window is empty (caller treats as exhausted).
//
// Pure function — does not mutate `motion`; the tick reset is the caller's
// job (see the control loop's trim-install path). Hot-path safe: no heap,
// no exceptions; output RawMotionPlan is fixed-size and trivially copyable.
[[nodiscard]] RawMotionPlan trim_motion(const MotionData&            motion,
                                          std::int32_t                 current_tick,
                                          std::int32_t                 target_input_fps,
                                          const JointRemap<kG1Dof>&    remap) noexcept;

// Whole master row a trim prefix starts at, for the exact (fractional) playback
// point `mf`: the NEAREST stored frame (never negative). The trim returns stored
// frames from here on (no blending); the gap mf - start, in [-0.5, 0.5], is
// carried to the next install (MotionData::trim_phase). Nearest, not "next": a
// plan always lands >= 1 tick (0.6 frames) after the trim, so the robot has
// always reached the prefix's first frame by then and the shifted grid never
// starts before it.
[[nodiscard]] inline double master_trim_start(double mf) noexcept {
    if (!(mf > 0.0)) return 0.0;                  // also NaN
    return std::floor(mf + 0.5 + 1e-9);
}

// Trim commit on the LIVE motion (the control loop's side of a trim reply):
// drop the `cut` rows already behind the counter, advance the exact master
// position by the same time, and record the trim phase for the next install. A
// cut at/past the window end (server clamp after a generator stall) keeps the
// held final frame. Returns false when nothing could be committed. Called by
// the control loop's trim commit.
bool commit_trim_cut(MotionData& motion, std::int32_t cut) noexcept;

}  // namespace rt_deploy::policy

// Forward-declare the G1 ActivePlanSnapshot so the snapshot-based trim below
// can take it by reference without pulling in the full header.
namespace rt_deploy::g1 { struct ActivePlanSnapshot; }

namespace rt_deploy::policy {

// trim_ref_motion_from_snapshot — identical math to trim_motion(MotionData,...),
// but the source is an owned ActivePlanSnapshot copy (see
// include/rt_deploy/g1/active_plan_snapshot.hpp). Used by a transport's
// server-side trim path: the server reads the snapshot from a TripleBuffer
// the control loop publishes on plan-install events, computes the trim
// locally, and replies without blocking on the next control tick.
//
// `current_idx` is the index
// INTO the snapshot's live window (already accounts for tick - install_tick
// clamping). The snapshot's start_row is always 0 by construction. An index at
// or past the window is clamped to the held last frame, except on a truncated
// snapshot (ActivePlanSnapshot::tail_held false), where it returns n_rows == 0.
[[nodiscard]] RawMotionPlan trim_ref_motion_from_snapshot(
    const g1::ActivePlanSnapshot& snap,
    std::int32_t                  current_idx,
    std::int32_t                  target_input_fps,
    const JointRemap<kG1Dof>&     remap) noexcept;

}  // namespace rt_deploy::policy
