//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// VisionConfig — depth-source configuration, loaded from configs/depth_d435i.yaml.
// Parsed once at startup (off hot path), mirroring the g1::load_config idiom
// (yaml-cpp + Result<>).
//
// Everything the operator tunes lives here: which depth source to run, the IR
// stream geometry, the TRT plan, the depth cutoff, and the per-thread real-time
// attributes (FIFO priority + CPU affinity for the capture and inference
// threads). CPU pinning matters on the shared Orin NX: the depth threads must
// stay OFF the core the g1 control loop is pinned to, and must run at a lower
// priority than it.
//
// There is no publish/ZMQ stage — the depth frame is handed to the control loop
// in-process through a wait-free core::TripleBuffer<g1::DepthFrame>.

#include "rt_deploy/core/expected.hpp"
#include "rt_deploy/vision/depth_ops.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace rt_deploy::vision {

// Which producer fills the control loop's depth inbox.
enum class DepthSource : std::uint8_t {
    // D435i IR pair -> LiteAnyStereo (TensorRT) -> fx*B/disp. The sim2real
    // path: a learned matcher with the projector OFF, far cleaner than the
    // on-chip block matcher.
    kCleaned = 0,
    // The D435i's own on-chip stereo depth stream (Z16 -> metres). Fallback /
    // A-B reference; needs no GPU, no TRT plan and no CUDA.
    kRaw = 1,
};

// How the network-resolution depth is reduced to the policy's 84x48 grid.
// Alias of vision::Reduce (depth_ops.hpp) so the config enum and the CUDA
// device code share one set of numeric values and cannot drift apart.
//
//   kMin   — closest depth over the footprint. Conservative (never hides an
//            obstacle) and naturally ignores invalid pixels, which sit at
//            clip_max. Biases the image NEARER than training, which
//            point-samples one ray per pixel centre.
//   kArea  — mean over the footprint. Smoothest, but averages across depth
//            discontinuities and drags edges toward the mean.
//   kPoint — nearest-neighbour at the footprint centre. Closest analogue to the
//            training renderer, but keeps sensor speckle that min/area suppress.
using DownsampleMode = Reduce;

[[nodiscard]] std::string_view to_string(DepthSource s) noexcept;
[[nodiscard]] std::string_view to_string(DownsampleMode m) noexcept;

struct ThreadRtConfig {
    int  fifo_priority = 0;   // 1..99; 0 = no SCHED_FIFO (plain SCHED_OTHER)
    int  cpu_affinity  = -1;  // -1 = no pinning
    bool lock_memory   = false;
};

struct VisionConfig {
    // ── Which producer ──
    DepthSource source = DepthSource::kCleaned;

    // ── Camera / network input resolution ──
    // The IR streaming resolution == the resolution fed to the network. 424x240
    // is a native D435i IR mode and downsamples to 84x48 with fx within ~0.5% of
    // the training camera's 42.19 px, so it is both the fastest and the most
    // faithful choice. Depth is reduced to `out_*` below, so this drives cost,
    // not output size. Keep both dims 32-divisible (the net pads to /32).
    std::int32_t width   = 424;
    std::int32_t height  = 240;
    std::int32_t fps     = 30;
    bool         emitter = false;  // keep OFF: the dot pattern is structured
                                   // noise for a learned matcher
    std::string  serial;           // optional: bind a specific D435i

    // ── Inference (kCleaned only) ──
    std::string  plan_path;             // stereo TensorRT .plan (built with onnx_to_trt)
    std::int32_t device_id      = 0;
    bool         use_cuda_graph = true;

    // ── Output (the depth-student `student_depth` observation) ──
    // MUST match g1::DepthFrame::kCols/kRows. Validated in validate().
    std::int32_t   out_width  = 84;
    std::int32_t   out_height = 48;
    // Metric clip window [m]. MUST equal the policy config's
    // observation.groups.<image>.clip — g1::setup_depth (apps/g1/depth_setup.hpp)
    // loads both and refuses to start if they disagree, because a mismatch
    // mis-scales every pixel without any visible error.
    //
    // Defaults: 0.2 / 2.5 = the HALO depth student's training clip. In the
    // deploy path the cross-check above is what makes them safe.
    float          clip_min_m = 0.2F;  // near clip [m]; valid depths below clamp UP
    float          clip_max_m = 2.5F;  // far clip [m]; valid above + invalid/occluded
                                       // map here (NEVER 0 — see depth_ops.hpp)
    // Mask-weighted mean by default: plain kArea averages far-filled holes and so
    // biases every cell farther with local hole density (+0.32 m on a 0.5 m wall
    // at the D435i's 15.4% dropout). See depth_d435i.yaml for the full comparison
    // and Reduce::kAreaValid for the measurement.
    DownsampleMode downsample = DownsampleMode::kAreaValid;

    // Optional intrinsics override. When fx_override > 0 the camera-read
    // intrinsics are ignored (useful for replaying recorded IR offline).
    float fx_override         = 0.0F;
    float baseline_m_override = 0.0F;

    // ── Real-time thread placement ──
    // Both must sit BELOW the control loop's priority and, where a core can be
    // spared, off its CPU. See configs/depth_d435i.yaml for the Orin NX layout.
    ThreadRtConfig capture_rt;    // blocks in wait_for_frames; kRaw's ONE thread
                                  // (capture + convert + publish) uses this too
    ThreadRtConfig inference_rt;  // kCleaned ONLY: GPU submit, shares the device
                                  // with the policy. kRaw validates and ignores it.

    // ── Run control ──
    float rate_hz          = 30.0F;  // depth publish cadence (0 = free-run)
    // >0 = diagnostic stderr printout period [s]. Honoured by the RAW path only
    // (RawDepthSource prints published/produce-ms/rate/overruns/errors and its
    // latency statistics); the cleaned path ignores it and logs once a second.
    float profile_period_s = 0.0F;

    // Reject configurations that would silently produce a wrong observation.
    // Checked at startup, before any device or GPU resource is acquired.
    [[nodiscard]] core::Result<void, std::string> validate() const;
};

// Parse a YAML file into a VisionConfig. Missing fields keep their defaults;
// type errors / unreadable files return an error string.
[[nodiscard]] core::Result<VisionConfig> load_vision_config(std::string_view yaml_path);

}  // namespace rt_deploy::vision
