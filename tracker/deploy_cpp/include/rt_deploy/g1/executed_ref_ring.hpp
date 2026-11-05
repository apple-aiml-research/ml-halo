//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// ExecutedRefEntry — the reference frame the active motion holds at the current
// control tick. The control loop publishes one per tick into
// ControlLoopDeps::ref_outbox (a TripleBuffer<ExecutedRefEntry>), so a transport
// can report the reference (e.g. a viewer ghost) without reading the motion
// buffers the control thread installs into and trims.

#include "rt_deploy/policy/motion_loader.hpp"   // kG1Dof

#include <cstdint>

namespace rt_deploy::g1 {

struct ExecutedRefEntry {
    std::int64_t  gtick{0};        // control-loop tick count (monotonic, 1 per control step)
    std::int32_t  local_tick{0};   // inference_tick (row index into the live window) this tick
    std::uint64_t epoch{0};        // active-plan snapshot epoch at this tick
    float         pos[3]{};        // reference base position, world
    float         quat_wxyz[4]{1.0F, 0.0F, 0.0F, 0.0F};
    float         dof_train[policy::kG1Dof]{};   // reference joints, TRAINING order
};

}  // namespace rt_deploy::g1
