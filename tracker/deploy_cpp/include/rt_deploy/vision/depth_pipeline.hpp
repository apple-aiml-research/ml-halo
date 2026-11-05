//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// DepthPipeline — the `kCleaned` depth source: D435i IR pair -> LiteAnyStereo
// (TensorRT) -> metric depth -> the g1 control loop's depth inbox.
//
//   [rs-capture thread]                    [depth-infer thread @ rate_hz]
//    wait_for_frames()                      PeriodicTicker
//    IR1/IR2 memcpy      --TripleBuffer-->  ir_to_chw -> TRT -> fx*B/disp
//                         <StereoFrame>     -> clamp -> downsample 84x48
//                         latest-wins                        |
//                                            --TripleBuffer<g1::DepthFrame>-->
//                                                     [g1 control loop @50Hz]
//
// Both hand-offs are wait-free latest-value slots (a frame-dropping queue of
// depth 1, without the mutex): the
// inference stage always pulls the newest IR pair and the control loop always
// reads the newest depth. End-to-end latency is one capture + one inference no
// matter how the stages drift apart, and no stage can ever block another.
//
// Overrun behaviour: core::PeriodicTicker resyncs to "now" when a cycle
// overruns its period rather than spinning to catch up, so a slow inference
// degrades the depth rate smoothly (30 -> 20 -> 15 Hz) and never backs up,
// never allocates, and never stalls the reader. Overruns are counted in
// DepthStats::overruns.
//
// The inference thread runs at a LOWER scheduling priority than the g1 control
// loop and, where a core can be spared, off its CPU (config.inference_rt).
//
// The StereoFrame TripleBuffer is ~6 MB at max resolution, so it is heap-held
// and the DepthPipeline object itself stays small.

#include "rt_deploy/core/expected.hpp"
#include "rt_deploy/core/triple_buffer.hpp"
#include "rt_deploy/g1/depth_frame.hpp"
#include "rt_deploy/vision/camera_intrinsics.hpp"
#include "rt_deploy/vision/config.hpp"
#include "rt_deploy/vision/depth_source.hpp"
#include "rt_deploy/vision/realsense_stereo.hpp"
#include "rt_deploy/vision/stereo_depth_engine.hpp"
#include "rt_deploy/vision/stereo_frame.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <thread>

namespace rt_deploy::vision {

class DepthPipeline final : public IDepthSource {
public:
    // `inbox` is the control loop's depth sink and must outlive the pipeline.
    DepthPipeline(VisionConfig config, core::TripleBuffer<g1::DepthFrame>& inbox);
    ~DepthPipeline() override;

    // Open the camera (reads live fx + baseline), build the TensorRT engine,
    // push the intrinsics-derived depth params into it, then spawn the capture
    // and inference threads. Off hot path.
    [[nodiscard]] core::Result<void, std::string> start() override;

    void stop() noexcept override;

    [[nodiscard]] DepthStats       stats() const noexcept override;
    [[nodiscard]] StereoIntrinsics intrinsics() const noexcept override { return intr_; }
    [[nodiscard]] std::string_view name() const noexcept override { return "cleaned"; }

private:
    // The rate-limited produce loop; body of the `depth-infer` thread.
    void infer_loop() noexcept;

    VisionConfig                          config_;
    core::TripleBuffer<g1::DepthFrame>&   inbox_;
    StereoIntrinsics                      intr_{};

    std::unique_ptr<core::TripleBuffer<StereoFrame>> frame_tb_;
    std::unique_ptr<RealSenseStereo>                 camera_;
    std::unique_ptr<IStereoDepthEngine>              engine_;

    // Engine scratch: the engine writes the reduced 84x48 metres plane here, and
    // infer_loop() copies it into the g1::DepthFrame slot. Heap-held (the struct
    // is ~256 KB) and touched only by the inference thread.
    std::unique_ptr<DepthFrame> scratch_;

    std::thread       thread_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> started_{false};

    // Diagnostics — written only by the inference thread, read from anywhere.
    std::atomic<std::uint64_t> published_{0};
    std::atomic<std::uint64_t> dropped_{0};
    std::atomic<std::uint64_t> overruns_{0};
    std::atomic<std::uint64_t> errors_{0};
    std::atomic<float>         last_ms_{0.0F};
    std::atomic<float>         ema_hz_{0.0F};

    // Arrival latency (StereoFrame::stamp_ns -> publish) for the 1 Hz log line:
    // the host stamp taken when the driver handed us the frame, so it EXCLUDES
    // exposure and USB transport. Written ONLY by the inference thread (its
    // single-writer contract). ~8 KB; the pipeline is heap-held, so this does
    // not grow anyone's stack.
    LatencyTracker            lat_arrival_;
};

}  // namespace rt_deploy::vision
