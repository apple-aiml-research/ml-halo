//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// UnitreeBackend — real-robot backend for Unitree G1 over Unitree SDK2.
//
// Implementation verified against the canonical Unitree SDK2 headers at
// unitree_sdk2/include/unitree/{robot/channel,idl/hg,dds_wrapper/common}.
// See the .cpp for the precise header list and API citations.
//
// Build gating (cmake):
//   RT_DEPLOY_WITH_UNITREE=ON — enables this TU; the SDK is located by
//   cmake/FindUnitreeSDK2.cmake.
//
// Topic convention (HG / G1):
//   rt/lowcmd     publish    — LowCmd_   (CRC computed with crc32_core)
//   rt/lowstate   subscribe  — LowState_ (CRC validated on receive)
//
// Thread model:
//   The SDK invokes the subscriber callback on its own DDS thread pool,
//   serialized per topic. State snapshots hand off to the control thread
//   via a wait-free TripleBuffer<RobotState<29>>; remote state uses a
//   second TripleBuffer<RemoteState>. ChannelPublisher::Write is documented
//   thread-safe, so the control thread publishes directly on every tick.

#include "rt_deploy/core/expected.hpp"
#include "rt_deploy/g1/config.hpp"
#include "rt_deploy/robot/i_robot_backend.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace rt_deploy::g1 {

struct UnitreeBackendConfig {
    std::string network_interface = "eth0";
    int         domain             = 0;
    float       damping_kd         = 8.0F;
    // If true, inbound LowState_ messages are application-level CRC-checked
    // and dropped on mismatch. Defaults to false to match the Python backend
    // (deploy/utils/backends/unitree_backend.py) — DDS has its own
    // transport checksum, and a strict CRC check will drop every message if
    // the Unitree SDK struct layout differs between this binary and the
    // robot firmware. Turn ON only once you know the SDK versions match.
    bool        verify_inbound_crc = false;
    // DRY RUN. When false, send_command() builds the LowCmd_ exactly as normal —
    // CRC included — and then does NOT publish it. Everything upstream is
    // unchanged: DDS state still arrives, the observation is still built, the
    // policy still infers, the action is still decoded, and the loop still runs
    // at its real rate on its real RT thread. Only the actuation is withheld.
    //
    // This exists for timing runs: measuring control-loop period and jitter
    // with the depth thread running requires the identical CPU and GPU load, but
    // does NOT require the robot to move. Inference only runs
    // in the CONTROL state, so "just don't press the buttons" measures nothing.
    //
    // SAFETY: with this false the robot is never commanded AT ALL — including the
    // damping command cleanup() normally sends on exit. Whatever the joints do
    // when lowcmd stops arriving is firmware behaviour this code does not
    // control. Run with the robot supported on a stand and its motors disabled
    // or E-stopped; do not treat "no commands" as equivalent to "held safe".
    bool        send_enabled = true;
    // Subscribe only: no rt/lowcmd publisher is created at all, so this process
    // can never put a LowCmd_ on the wire, not even the damping command
    // cleanup() sends on exit. Implies send_enabled = false. For read-only tools
    // (g1_state_monitor) that run next to a live controller.
    bool        subscribe_only = false;
    // Odometry (rt/odommodestate) older than this is reported as invalid
    // (RobotState::odom_valid = false). 100 ms = 5 control periods at 50 Hz; the
    // estimator publishes at several hundred Hz, so this only trips on a stalled
    // or stopped stream. Same threshold as the Python deployment.
    std::int64_t odom_stale_ns = 100'000'000;
};

// Returns a live backend or an error string. Off the hot path.
[[nodiscard]] core::Result<std::unique_ptr<robot::IRobotBackend<kDof>>>
make_unitree_backend(const UnitreeBackendConfig& cfg);

}  // namespace rt_deploy::g1
