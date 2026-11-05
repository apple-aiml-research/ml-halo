//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// StereoIntrinsics — the calibration needed to turn disparity into metric depth
// and to advertise the synthetic depth stream to downstream consumers.
//
// Read live from the D435i at start-up (left-IR intrinsics + IR1→IR2
// extrinsics). They are fixed for the session: the IR pair is
// hardware-rectified, so fx/baseline are valid for `depth = fx*B/disp` at the
// streaming resolution with no per-frame work.
//
// `depth_scale` describes the D435i's RAW Z16 stream (metres per integer
// count, RS2_OPTION_DEPTH_UNITS, nominally 0.001). It exists for the kRaw
// producer, which is the only consumer: it multiplies the Z16 counts by the
// LIVE value read from rs2::depth_sensor to get metres. The kCleaned producer
// computes `depth = fx*B/disp` directly and never applies it.
//
// Either way the PUBLISHED depth is float32 METRES, not Z16 and not
// millimetres — see DepthFrame in stereo_frame.hpp. depth_scale travels with
// the frame purely as source provenance.

#include <cstdint>

namespace rt_deploy::vision {

struct StereoIntrinsics {
    float fx         = 0.0F;  // left-IR focal length x [px] at streaming res
    float fy         = 0.0F;  // left-IR focal length y [px]
    float cx         = 0.0F;  // principal point x [px]
    float cy         = 0.0F;  // principal point y [px]
    float baseline_m = 0.0F;  // IR1→IR2 stereo baseline [m] (abs translation x)

    std::int32_t width  = 0;
    std::int32_t height = 0;

    // Metres per Z16 count of the RAW D435i depth stream (RS2_OPTION_DEPTH_UNITS,
    // nominally 0.001). Provenance only — the published depth is float32 metres.
    float depth_scale = 0.001F;

    // Convenience: the per-frame constant fed to disp_to_depth_clamped().
    [[nodiscard]] float fx_baseline() const noexcept { return fx * baseline_m; }
};

}  // namespace rt_deploy::vision
