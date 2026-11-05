//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// IRobotBackend — abstract interface between the control loop and the world.
//
// Two concrete implementations:
//   - g1::UnitreeBackend : real G1 over Unitree SDK2 DDS (rt_deploy_g1_unitree)
//   - MujocoBackend     : MuJoCo physics (deploy_g1_mujoco, system/cpp sim node)
//
// All methods are noexcept and hot-path safe unless marked otherwise. The
// DDS-async nature of the real backend is hidden behind read_state(): it
// fills an owned RobotState snapshot and tells the caller whether the data
// is fresh since the last read.

#include "rt_deploy/core/triple_buffer.hpp"
#include "rt_deploy/g1/depth_frame.hpp"
#include "rt_deploy/math/types.hpp"

#include <Eigen/Core>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace rt_deploy::robot {

// State of the wireless-remote joystick (matches Unitree Python contract).
struct RemoteState {
    std::array<std::uint8_t, 16> button{};  // 0 or 1 per KeyMap index
    float                        lx{0.0F};
    float                        ly{0.0F};
    float                        rx{0.0F};
    float                        ry{0.0F};
};

enum class ReadStatus : std::uint8_t {
    kOk         = 0,  // fresh state returned
    kStaleOk    = 1,  // same snapshot as last call (no DDS update since)
    kNoData     = 2,  // state has not arrived yet (startup)
    kDisconnect = 3,  // backend reports link loss
};

// Non-hot-path counters exposed for diagnostics (state monitors, logs, etc.).
// Defaults to all-zero — concrete backends override the getter when they
// track these. Real Unitree DDS backend fills rx_count / crc_rejects /
// publish_failures / odom_rx_count; sim/mock backends just return zeros.
struct BackendDiagnostics {
    std::uint64_t rx_count         = 0;  // inbound state messages seen
    std::uint64_t crc_rejects      = 0;  // inbound messages dropped on CRC mismatch
    std::uint64_t publish_failures = 0;  // outbound send_command Write() failures
    std::uint64_t odom_rx_count    = 0;  // optional sport-mode estimator messages seen
};

template <std::size_t N>
class IRobotBackend {
public:
    virtual ~IRobotBackend() = default;

    IRobotBackend()                                = default;
    IRobotBackend(const IRobotBackend&)            = delete;
    IRobotBackend& operator=(const IRobotBackend&) = delete;
    IRobotBackend(IRobotBackend&&)                 = delete;
    IRobotBackend& operator=(IRobotBackend&&)      = delete;

    // Read the latest robot state. noexcept; fills `out` on success and sets
    // out.timestamp_ns to the backend's monotonic receive time for watchdog use.
    [[nodiscard]] virtual ReadStatus read_state(math::RobotState<N>& out) noexcept = 0;

    // Publish a motor command. noexcept. Expected to complete in << one tick.
    virtual void send_command(const math::MotorCommand<N>& cmd) noexcept = 0;

    // Read the latest remote-controller state. noexcept. Returns false if
    // the backend does not expose a remote (MuJoCo without UI, etc.).
    [[nodiscard]] virtual bool poll_remote(RemoteState& out) noexcept = 0;

    // Perform one simulation step (sim-only backends). Default no-op for real.
    virtual void step() noexcept {}

    // Publish a damping command and release any held resources. Idempotent.
    virtual void cleanup() noexcept = 0;

    // Startup gate: block up to `timeout_ns` for the first valid state. Off
    // the hot path.
    [[nodiscard]] virtual bool wait_for_state(std::int64_t timeout_ns) = 0;

    // Diagnostic counters. Default returns zeros for backends that don't
    // track these (sim / mock). Safe to call from any thread at any cadence.
    [[nodiscard]] virtual BackendDiagnostics diagnostics() const noexcept { return {}; }

    // Run a blocking viewer loop on the calling thread, returning when the
    // window is closed or `request_viewer_stop()` is called. Default no-op
    // (returns immediately) for backends without a viewer.
    //
    // This exists so macOS builds can satisfy AppKit's main-thread-only
    // constraint: the control loop runs on a worker and the main thread
    // drives the viewer. Linux builds may still auto-spawn the viewer on an
    // internal worker thread; backends that do so should make this method
    // return immediately.
    virtual void run_viewer_blocking() noexcept {}
    virtual void request_viewer_stop() noexcept {}

    // Optional ego-camera offscreen render (used by the system/cpp sim node's
    // image-carrying state reply). Default no-ops for backends that don't
    // support rendering (real Unitree, mock). init_ego_renderer is called
    // once (typically from the main thread for macOS GL); render_ego_image
    // runs on whatever thread services the state RPC. Wire format: rgb_out is
    // filled with width*height*3 raw RGB uint8 bytes (row-major, top-down).
    // Returns false when unsupported or init was never called / failed.
    [[nodiscard]] virtual bool init_ego_renderer(int /*width*/, int /*height*/,
                                                 const std::string& /*camera_name*/) {
        return false;
    }
    [[nodiscard]] virtual bool render_ego_image(std::vector<std::uint8_t>& /*rgb_out*/,
                                                int& /*w_out*/, int& /*h_out*/) noexcept {
        return false;
    }

    // Optional depth-camera offscreen render for the depth student (sim).
    // Mirrors the ego renderer: init_depth_renderer(...) is called once (main
    // thread, macOS GL) AFTER init_ego_renderer (the depth render reuses the ego
    // offscreen context). The backend then renders the named depth camera every
    // `obs_decim` step()s into depth_inbox() as metric-meter DepthFrames, which the
    // control loop drains (tick step 1c in control_loop.cpp). width x height must be the
    // DepthFrame size (84x48). Default no-ops: backends without a depth camera
    // (the real robot uses a RealSense source; mock) return false / nullptr, and a
    // config with an image obs group is then refused at start
    // (ControlLoopExit::kConfigError) unless another depth source is attached.
    //
    // `obs_decim` is the SIM half of the depth-rate contract. A synchronous
    // renderer would otherwise produce a fresh frame every control tick, i.e.
    // twice the 25 Hz the depth student trains at. Decimating
    // the render is equivalent to holding the frame, because the inbox keeps
    // serving the last published one — and it keeps the whole mechanism inside the
    // sim backend, so the hardware path cannot accidentally inherit it. See
    // g1::GroupCfg::obs_decim.
    [[nodiscard]] virtual bool init_depth_renderer(const std::string& /*camera_name*/,
                                                   int /*width*/, int /*height*/,
                                                   float /*clip_min*/, float /*clip_max*/,
                                                   int /*obs_decim*/ = 1) {
        return false;
    }
    [[nodiscard]] virtual core::TripleBuffer<g1::DepthFrame>* depth_inbox() noexcept {
        return nullptr;
    }
};

}  // namespace rt_deploy::robot
