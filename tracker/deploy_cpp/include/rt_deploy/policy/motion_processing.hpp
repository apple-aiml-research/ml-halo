//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// motion_processing — resampling + derivative primitives for runtime-pushed
// plans.
//
// The control loop calls these when a RawMotionPlan arrives on its motion
// inbox, so every transport hands over raw wire rows and none re-implements
// the motion math.
//
// Cost is small enough for the 50 Hz control thread: on the order of 100 µs
// for a few hundred output rows on a swap tick (budget 20 ms).
//
// All functions are noexcept, allocation-free, and operate on a pre-sized
// MotionData scratch built via policy::make_motion_scratch(...).

#include "rt_deploy/policy/motion_loader.hpp"

#include <cstdint>

namespace rt_deploy::policy {

// Number of output rows resample_motion will write for these arguments (0 if it
// would fail). The single source of that count, so a caller can validate a plan
// (e.g. its start offset) BEFORE resample_motion overwrites the slot.
[[nodiscard]] std::int32_t resample_output_frames(std::int32_t n_rows,
                                                  std::int32_t input_fps,
                                                  std::int32_t output_fps,
                                                  std::int32_t scratch_capacity,
                                                  double       start_input_row = 0.0) noexcept;

// Resample (n_rows × 36) qpos data at input_fps into `slot` at output_fps.
//
// Writes rows [slot.start_row, slot.start_row + n_out) of base_poss /
// base_rots / dof_poss using LERP for positions/joints and SLERP for the
// base quaternion. Source quaternion is read as xyzw and stored as wxyz
// (matches the MotionLoader convention).
//
// Joints are written in DEPLOYMENT order — call apply_joint_remap(slot, ...)
// afterwards to get training-order data.
//
// Velocities are NOT computed here — call compute_motion_derivatives(slot)
// next.
//
// Sets slot.output_fps = output_fps, slot.input_fps = input_fps,
// slot.start_row = kMaxTransitionSteps, slot.frame_count = n_out (capped by
// the available scratch).
//
// start_input_row: output row i samples input row
// start_input_row + i * input_fps / output_fps (clamped to [0, n_rows - 1]).
// 0 = the grid anchored at input row 0. The install passes the last trim's
// phase here so the new plan's output grid lines up with the true playback
// time.
//
// Preconditions:
//   - slot.scratch_capacity > kMaxTransitionSteps
//   - n_rows >= 1, input_fps > 0, output_fps > 0
//   - raw_qpos points to at least n_rows * 36 doubles
//
// Returns false if scratch_capacity is too small to hold even a single
// output row past kMaxTransitionSteps; true otherwise.
[[nodiscard]] bool resample_motion(const double* raw_qpos,
                                    std::int32_t  n_rows,
                                    std::int32_t  input_fps,
                                    std::int32_t  output_fps,
                                    MotionData&   slot,
                                    double        start_input_row = 0.0) noexcept;

// Compute dof_vels, base_lin_vels, base_ang_vels from already-populated
// positions via central differences (one-sided at the endpoints). Operates
// on rows [slot.start_row, slot.start_row + slot.frame_count). In-place,
// no heap, no exceptions.
//
// `slot.output_fps` must be > 0 (used as the sample rate for diffs).
void compute_motion_derivatives(MotionData& slot) noexcept;

}  // namespace rt_deploy::policy
