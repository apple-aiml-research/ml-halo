//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// ActivePlanSnapshot — an owned copy of the post-processed motion the control
// loop is currently tracking. Published into a
// `core::TripleBuffer<ActivePlanSnapshot>` by the control loop on plan-install
// and trim-commit events (NOT every tick). Consumed lock-free by a transport's
// server thread when servicing trim requests, so the trim computation runs
// entirely off the control thread and never waits for the next 50 Hz tick.
//
// Owns its data: the matrices below are *copies* taken at publish time, not
// pointers into controller-owned MotionData, so the snapshot can outlive the
// source plan.
//
// Trim flow:
//
//   control loop (producer), on plan install / trim commit:
//     snap.dof_poss.topRows(n) = active.dof_poss.middleRows(start_row, n)
//     snap.base_poss / base_rots / master_qpos: same
//     snap.frame_count = n, output_fps / input_fps / master_* from active
//     snap.install_tick = 0, snap.epoch = ++counter
//     triple_buffer.publish()
//
//   server thread (consumer), on a trim request:
//     if (auto* p = triple_buffer.try_acquire_latest()) held = p;
//     snap = held   // nullptr from try_acquire_latest means "unchanged", NOT "empty":
//                   // keep the pointer of the last successful acquire (valid until the next one)
//     if (snap == nullptr || snap->epoch == 0) reply empty   // nothing ever published
//     idx   = inference_tick_mirror - snap->install_tick
//     reply = policy::trim_ref_motion_from_snapshot(*snap, idx, fps, snap->remap)
//     then signal the trim commit through the trim-install atomics
//
// Sizing: kMaxSnapshotRows equals the plan scratch the system/cpp tracker nodes
// allocate, kMaxTransitionSteps + RawMotionPlan::kMaxRows * output_fps / 30 =
// 50 + 1800 * 50/30 = 3050 rows at 50 Hz, so an installed plan is never
// truncated (the nodes refuse a control rate whose scratch would not fit). Only
// a longer motion, e.g. the padded standing pose, is cut on publish; then
// `tail_held` says whether the rows past the copy all repeat its last row, and a
// trim past the copied window is refused when they do not (see
// policy::trim_ref_motion_from_snapshot).
//
// Memory: 3050 rows × (29 + 3 + 4) floats + 1800 × 36 master floats ≈ 700 KB
// per slot, ~2.1 MB across the TripleBuffer's three slots.

#include "rt_deploy/policy/joint_remap.hpp"
#include "rt_deploy/policy/motion_loader.hpp"   // for kG1Dof

#include <Eigen/Core>

#include <cstdint>

namespace rt_deploy::g1 {

inline constexpr std::int32_t kMaxSnapshotRows =
    policy::kMaxTransitionSteps + policy::RawMotionPlan::kMaxRows * 50 / 30;

struct ActivePlanSnapshot {
    // Dynamic-row (fixed-col) matrices, each resized ONCE to kMaxSnapshotRows
    // in the constructor below. Same layout as policy::MotionData. They are
    // intentionally NOT fixed-size Eigen::Matrix<float, kMaxSnapshotRows, ...>:
    // a 3050-row fixed matrix is ~350 KB, which (a) trips Eigen's 128 KiB
    // stack-allocation static_assert under debug builds (EIGEN_NO_DEBUG masks
    // it in Release) and (b) would overflow the stack for any by-value
    // temporary. Heap-backed rows give the same fixed capacity without either
    // hazard — the publisher's topRows(n) copies below never reallocate.
    using DofMat   = Eigen::Matrix<float, Eigen::Dynamic,
                                    static_cast<Eigen::Index>(policy::kG1Dof),
                                    Eigen::RowMajor>;
    using Base3Mat = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
    using Base4Mat = Eigen::Matrix<float, Eigen::Dynamic, 4, Eigen::RowMajor>;

    // Owned copies of the active motion's pose data — fixed row capacity
    // (kMaxSnapshotRows, allocated once at construction) so the TripleBuffer
    // slot never reallocates on the hot path. Joints are training order
    // (consumers apply the JointRemap at trim time, exactly like
    // policy::trim_motion does today).
    DofMat   dof_poss;        // (kMaxSnapshotRows, 29)
    Base3Mat base_poss;       // (kMaxSnapshotRows, 3)   xyz world
    Base4Mat base_rots;       // (kMaxSnapshotRows, 4)   wxyz

    // Pristine input-fps master clip (deploy-order / xyzw), copied from the
    // active MotionData at publish time so the server-side trim can return an
    // exact slice without re-downsampling. `master_rows == 0` ⇒ no master, trim
    // resamples instead.
    policy::MotionData::MasterMat master_qpos;  // (RawMotionPlan::kMaxRows, 36)

    ActivePlanSnapshot() {
        dof_poss.resize(kMaxSnapshotRows, static_cast<Eigen::Index>(policy::kG1Dof));
        base_poss.resize(kMaxSnapshotRows, 3);
        base_rots.resize(kMaxSnapshotRows, 4);
        master_qpos.resize(static_cast<Eigen::Index>(policy::RawMotionPlan::kMaxRows),
                           static_cast<Eigen::Index>(policy::kCsvCols));
    }

    // Live-window descriptors. Unlike MotionData, the snapshot is laid
    // out so the active window starts at row 0 — trimming on the server
    // side is just a slice from `idx` to `frame_count`.
    std::int32_t start_row    = 0;            // always 0; field kept for symmetry / sanity
    std::int32_t frame_count  = 0;            // valid rows in [0, frame_count)
    // False when the source motion was longer than kMaxSnapshotRows and its rows
    // past the copy are not all equal to the last copied row: the copy is then a
    // truncated prefix, and a trim at or past frame_count has no data to answer.
    bool         tail_held    = true;
    std::int32_t output_fps   = 0;            // control-rate fps the snapshot is sampled at
    std::int32_t input_fps    = 0;            // original input fps; trim resamples back to this
    std::int32_t master_rows  = 0;            // valid rows in master_qpos [0, master_rows)
    std::int32_t master_fps   = 0;            // fps of master_qpos (== original input fps)
    double       master_offset = 0.0;         // fractional master row of snapshot row 0 (MotionData::master_offset)

    // Synchronization helpers. install_tick is the value of the control
    // loop's inference_tick that corresponds to snapshot row 0; the server
    // thread computes its index in the snapshot as
    //   idx = inference_tick_mirror.load() - install_tick
    // bundled here (not a separate atomic) so a fresh slot is always
    // self-consistent with its own install_tick. The control loop indexes the
    // live window by inference_tick directly and the snapshot starts at the
    // window's row 0, so the loop always publishes install_tick = 0 (a trim
    // commit keeps the ticks played since the trim instead of resetting to 0).
    std::int32_t  install_tick = 0;
    // Monotonic counter, 0 = "never published". Server treats 0 as
    // "no live plan yet" and replies with an empty trim, instead of
    // reading uninitialized matrix data.
    std::uint64_t epoch        = 0;

    // Joint remap (training ↔ deployment) bundled with the snapshot so the
    // server thread can pass it straight to policy::trim_ref_motion_from_snapshot
    // without needing a side-channel for static config. Static for a given
    // binary; copied once here per publish (~232 B). The remap is the same
    // one held by the obs builder — reuses policy::JointRemap exactly.
    policy::JointRemap<policy::kG1Dof> remap{};
};

}  // namespace rt_deploy::g1
