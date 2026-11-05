//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// RealSenseStereo — D435i IR-pair capture source + dedicated capture thread.
//
// A PIMPL hides the librealsense2 headers from the rest of the tree, and a
// single capture thread owns the device. The thread blocks in rs2::pipeline::wait_for_frames() (the
// RT wake source), copies IR1/IR2 into the producer slot of a wait-free
// `core::TripleBuffer<StereoFrame>`, and publishes. Stale frames are dropped by
// construction — the consumer always reads the newest pair, so end-to-end
// latency stays at ~one inference regardless of consumer speed.
//
// D435i specifics handled here:
//   * IR stream index 1 = left, 2 = right; format Y8; hardware-rectified.
//   * The IR projector (emitter) FOLLOWS `config.emitter` — it is written as
//     RS2_OPTION_EMITTER_ENABLED = emitter ? 1 : 0, not forced off. For the
//     learned matcher (depth_source: cleaned, this class) it MUST be false: the
//     dot pattern is structured noise that visibly degrades the disparity. The
//     raw on-chip block matcher (RawDepthSource) usually wants it true, since a
//     classical correlator needs the projected texture on blank walls.
//   * Intrinsics (left-IR fx/fy/cx/cy) and the IR1→IR2 baseline are read once
//     at start-up so `depth = fx*B/disp` is metric at the streaming resolution.
//
// Hot-path safety: the capture loop performs no heap allocation. The only
// per-frame work is copying the two IR planes into the preallocated
// TripleBuffer slot — one bulk std::memcpy each when librealsense hands back
// unpadded rows, row-by-row when it does not.

#include "rt_deploy/core/expected.hpp"
#include "rt_deploy/core/rt_thread.hpp"
#include "rt_deploy/core/triple_buffer.hpp"
#include "rt_deploy/vision/camera_intrinsics.hpp"
#include "rt_deploy/vision/stereo_frame.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace rt_deploy::vision {

struct RealSenseStereoConfig {
    std::int32_t width   = 640;
    std::int32_t height  = 480;
    std::int32_t fps     = 30;
    bool         emitter = false;        // projector: false for the learned
                                         // matcher (dots are structured noise),
                                         // true suits the on-chip block matcher
    std::string  serial;                 // bind a specific device (optional)
    core::RtThreadConfig rt_thread;       // capture-thread RT attributes
};

class RealSenseStereo {
public:
    // `out` is the SPSC sink the capture thread publishes into. The caller
    // (DepthPipeline) owns it and is its sole consumer.
    RealSenseStereo(core::TripleBuffer<StereoFrame>& out, RealSenseStereoConfig config);
    ~RealSenseStereo();

    RealSenseStereo(const RealSenseStereo&)            = delete;
    RealSenseStereo& operator=(const RealSenseStereo&) = delete;
    RealSenseStereo(RealSenseStereo&&)                 = delete;
    RealSenseStereo& operator=(RealSenseStereo&&)      = delete;

    // Open the device, apply `config.emitter`, read intrinsics, and spawn the
    // capture thread. Returns an error string if the device cannot be opened or
    // configured; the thread is not started in that case.
    [[nodiscard]] core::Result<void, std::string> start();

    // Signal the capture thread to exit and join. Idempotent; called by dtor.
    void stop();

    // Valid only after a successful start(). Fixed for the session.
    [[nodiscard]] StereoIntrinsics intrinsics() const noexcept;

    // Cumulative count of frames published; for diagnostics / FPS reporting.
    [[nodiscard]] std::uint64_t frames_captured() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rt_deploy::vision
