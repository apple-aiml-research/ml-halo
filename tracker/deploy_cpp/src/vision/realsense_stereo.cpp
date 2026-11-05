//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

// RealSenseStereo — librealsense2 implementation of the IR-pair capture source.
//
// Verified against librealsense2 API (rs.hpp, rs_pipeline.hpp, rs_sensor.hpp):
//   - rs2::config::enable_stream(RS2_STREAM_INFRARED, index, w, h, RS2_FORMAT_Y8, fps)
//   - rs2::pipeline::start(cfg) → rs2::pipeline_profile
//   - rs2::depth_sensor::set_option(RS2_OPTION_EMITTER_ENABLED, 0|1)
//   - rs2::video_stream_profile::get_intrinsics() → rs2_intrinsics {fx,fy,ppx,ppy}
//   - profile_ir1.get_extrinsics_to(profile_ir2).translation[0] → baseline
//   - rs2::pipeline::wait_for_frames() → rs2::frameset; get_infrared_frame(1|2)
//
// All librealsense symbols are confined to this TU (PIMPL), so the rest of the
// tree builds without the SDK on its include path.

#include "rt_deploy/vision/realsense_stereo.hpp"

#include "rt_deploy/core/clock.hpp"
#include "rt_deploy/core/rate_probe.hpp"

#include <librealsense2/rs.hpp>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <thread>

namespace rt_deploy::vision {

namespace {

// Throttle for the capture thread's error log. A device that fails on every
// frame at 30 Hz would otherwise flood stderr, and stderr is a blocking,
// lock-taking write shared with the 50 Hz control loop's own diagnostics.
constexpr std::int64_t kErrorLogPeriodNs = 1'000'000'000;

// Copy one Y8 IR plane into the tightly packed StereoFrame slot.
//
// librealsense pads rows to an alignment boundary on some modes and firmware
// revisions, so get_data() is not always width*height contiguous bytes.
// Treating a padded buffer as packed shears the image, and a sheared IR pair
// yields plausible-looking but wrong disparity — silently wrong depth, no
// crash. The bulk memcpy stays the fast path for the usual unpadded case.
//
// Deliberately not noexcept: get_data()/get_stride_in_bytes() are librealsense
// calls that throw, and run()'s try block is what must see them.
void copy_ir_plane(std::uint8_t* dst, const rs2::video_frame& src, std::size_t width,
                   std::size_t height) {
    const auto* bytes  = static_cast<const std::uint8_t*>(src.get_data());
    const auto  stride = static_cast<std::size_t>(src.get_stride_in_bytes());
    if (stride == width) {
        std::memcpy(dst, bytes, width * height);
        return;
    }
    for (std::size_t y = 0; y < height; ++y) {
        std::memcpy(dst + (y * width), bytes + (y * stride), width);
    }
}

}  // namespace

struct RealSenseStereo::Impl {
    Impl(core::TripleBuffer<StereoFrame>& out_, RealSenseStereoConfig cfg_) noexcept
        : out(out_), config(std::move(cfg_)) {}

    core::TripleBuffer<StereoFrame>& out;
    RealSenseStereoConfig            config;

    // Held in an optional because rs2::pipeline's default constructor runs
    // rs2_create_context + rs2_create_pipeline, both of which throw when the
    // backend, the udev rules or the USB permissions are missing. As a plain
    // member it would be constructed under this ctor's `noexcept` — a
    // std::terminate at start-up with no diagnostic. Born inside open()'s try
    // block instead, so a misconfigured host comes back as a Result error.
    std::optional<rs2::pipeline> pipeline;
    rs2::pipeline_profile        profile;
    StereoIntrinsics             intr{};

    std::thread                capture_thread;
    std::atomic<bool>          running{false};
    std::atomic<std::uint64_t> frames{0};
    std::atomic<std::uint64_t> errors{0};

    [[nodiscard]] core::Result<void, std::string> open();
    void                                           read_intrinsics();
    void                                           close() noexcept;
    void                                           run();
};

core::Result<void, std::string> RealSenseStereo::Impl::open() {
    using ResT = core::Result<void, std::string>;
    try {
        // First thing inside the try: constructing the pipeline can throw (see
        // the member's comment).
        pipeline.emplace();

        rs2::config cfg;
        if (!config.serial.empty()) {
            cfg.enable_device(config.serial);
        }
        // IR stream index 1 = left, 2 = right on the D4xx series.
        cfg.enable_stream(RS2_STREAM_INFRARED, 1, config.width, config.height,
                          RS2_FORMAT_Y8, config.fps);
        cfg.enable_stream(RS2_STREAM_INFRARED, 2, config.width, config.height,
                          RS2_FORMAT_Y8, config.fps);

        profile = pipeline->start(cfg);

        // Apply the IR projector setting. The cleaned path requires it OFF: the
        // dot pattern is structured noise for a learned matcher.
        rs2::device       dev          = profile.get_device();
        rs2::depth_sensor depth_sensor = dev.first<rs2::depth_sensor>();
        if (depth_sensor.supports(RS2_OPTION_EMITTER_ENABLED)) {
            depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED,
                                    config.emitter ? 1.0F : 0.0F);
        }

        read_intrinsics();
    } catch (const rs2::error& e) {
        return ResT::Error{std::string{"librealsense error ("} + e.get_failed_function()
                           + "): " + e.what()};
    } catch (const std::exception& e) {
        return ResT::Error{std::string{"RealSense open failed: "} + e.what()};
    }
    return ResT{};
}

void RealSenseStereo::Impl::read_intrinsics() {
    auto ir1 = profile.get_stream(RS2_STREAM_INFRARED, 1).as<rs2::video_stream_profile>();
    auto ir2 = profile.get_stream(RS2_STREAM_INFRARED, 2);

    const rs2_intrinsics k = ir1.get_intrinsics();
    // Translation x of IR1→IR2 is -baseline; take the magnitude.
    const rs2_extrinsics extr = ir1.get_extrinsics_to(ir2);

    intr.fx          = k.fx;
    intr.fy          = k.fy;
    intr.cx          = k.ppx;
    intr.cy          = k.ppy;
    intr.baseline_m  = std::fabs(extr.translation[0]);
    intr.width       = config.width;
    intr.height      = config.height;
    // Nominal D435i Z16 unit. Carried for provenance only — this source emits an
    // IR PAIR, and the cleaned path derives metres from fx*B/disparity, never
    // from a Z16 scale. Only RawDepthSource applies depth_scale, and it reads the
    // live RS2_OPTION_DEPTH_UNITS from the device rather than trusting this.
    intr.depth_scale = 0.001F;

    std::fprintf(stderr,
                 "[realsense] %dx%d@%d  fx=%.1fpx  baseline=%.1fmm  emitter=%s\n",
                 config.width, config.height, config.fps, static_cast<double>(intr.fx),
                 static_cast<double>(intr.baseline_m * 1000.0F),
                 config.emitter ? "ON" : "OFF");
}

void RealSenseStereo::Impl::close() noexcept {
    if (!pipeline.has_value()) {
        return;
    }
    try {
        pipeline->stop();
    } catch (const std::exception&) {
        // Already stopped, or the device was unplugged — nothing actionable at
        // shutdown, and every caller of close() is on a must-not-throw path.
    }
    pipeline.reset();
}

void RealSenseStereo::Impl::run() {
    core::RtThreadConfig tc = config.rt_thread;
    if (tc.name.empty()) {
        tc.name = "rs-capture";
    }
    if (const int rc = core::configure_current_thread(tc); rc != 0) {
        std::fprintf(stderr, "[realsense] configure_current_thread rc=%d (continuing)\n", rc);
    }

    static core::RateProbe probe{"capture"};

    const auto w = static_cast<std::size_t>(config.width);
    const auto h = static_cast<std::size_t>(config.height);

    // Count every dropped frame, but log at most once per kErrorLogPeriodNs —
    // see the constant's note on stderr and the control loop.
    core::MonotonicClock::Nanos last_err_log{0};
    const auto                  note_error = [&](const char* what) noexcept {
        const std::uint64_t n   = errors.fetch_add(1, std::memory_order_relaxed) + 1U;
        const auto          now = core::MonotonicClock::now();
        if (last_err_log.count() != 0 && (now - last_err_log).count() < kErrorLogPeriodNs) {
            return;
        }
        last_err_log = now;
        std::fprintf(stderr, "[realsense] capture error (%llu total): %s\n",
                     static_cast<unsigned long long>(n), what);
    };

    while (running.load(std::memory_order_acquire)) {
        probe.tick();
        // The whole iteration is guarded, not just the wait: get_infrared_frame()
        // and get_data() throw too, and this function is a thread body — an
        // escaping exception is std::terminate, which would kill the 50 Hz
        // control loop mid-motion with the robot in the air. Losing depth is
        // recoverable; losing the control loop is not.
        try {
            rs2::frameset frames_set;
            // Blocking wake source for the RT capture loop. 1 s timeout so a
            // device hiccup surfaces rather than wedging the thread forever.
            if (!pipeline->try_wait_for_frames(&frames_set, 1000)) {
                continue;
            }

            rs2::video_frame left  = frames_set.get_infrared_frame(1);
            rs2::video_frame right = frames_set.get_infrared_frame(2);
            if (!left || !right) {
                note_error("frameset carried no IR pair");
                continue;
            }

            // The slot is written at the configured extent; a frame that
            // disagrees would run off the end of the fixed-size arrays, so drop
            // it rather than trust it (same rule as the raw Z16 path).
            if (left.get_width() != config.width || left.get_height() != config.height
                || right.get_width() != config.width || right.get_height() != config.height) {
                note_error("IR frame extent disagrees with the negotiated resolution");
                continue;
            }

            auto scope = probe.scope();
            // Publish into the wait-free slot. Two plane copies, no allocation.
            StereoFrame& slot = out.producer_slot();
            slot.width        = config.width;
            slot.height       = config.height;
            slot.seq          = frames.load(std::memory_order_relaxed);
            slot.stamp_ns     = core::MonotonicClock::now().count();
            copy_ir_plane(slot.left.data(), left, w, h);
            copy_ir_plane(slot.right.data(), right, w, h);
            out.publish();

            frames.fetch_add(1, std::memory_order_relaxed);
        } catch (const rs2::error& e) {
            note_error(e.what());
        } catch (const std::exception& e) {
            note_error(e.what());
        } catch (...) {
            note_error("unknown exception");
        }
    }

    close();
}

// ─── Public wrapper ───────────────────────────────────────────────────────────

RealSenseStereo::RealSenseStereo(core::TripleBuffer<StereoFrame>& out,
                                 RealSenseStereoConfig config)
    : impl_{std::make_unique<Impl>(out, std::move(config))} {}

RealSenseStereo::~RealSenseStereo() {
    stop();
}

core::Result<void, std::string> RealSenseStereo::start() {
    using ResT = core::Result<void, std::string>;
    if (impl_->running.load(std::memory_order_acquire)) {
        return ResT{};
    }
    if (auto r = impl_->open(); !r) {
        // open() can fail after pipeline->start() already succeeded (emitter,
        // intrinsics), so release the device before reporting.
        impl_->close();
        return r;
    }
    impl_->running.store(true, std::memory_order_release);
    try {
        impl_->capture_thread = std::thread([this]() { impl_->run(); });
    } catch (const std::exception& e) {
        // std::thread's ctor throws system_error when the process is out of
        // threads or the RT limits refuse one. Report it instead of letting it
        // escape a function that promises a Result, and leave nothing running.
        impl_->running.store(false, std::memory_order_release);
        impl_->close();
        return ResT::Error{std::string{"failed to spawn rs-capture thread: "} + e.what()};
    }
    return ResT{};
}

void RealSenseStereo::stop() {
    if (impl_ == nullptr) {
        return;
    }
    impl_->running.store(false, std::memory_order_release);
    // Called from ~RealSenseStereo(), so this is effectively noexcept: an
    // escaping exception during destruction terminates the process. join() can
    // throw system_error; swallow and log, since we are already tearing down.
    try {
        if (impl_->capture_thread.joinable()) {
            impl_->capture_thread.join();
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[rs_stereo] teardown: %s\n", e.what());
    } catch (...) {
        std::fprintf(stderr, "[rs_stereo] teardown: unknown exception\n");
    }
}

StereoIntrinsics RealSenseStereo::intrinsics() const noexcept {
    return impl_->intr;
}

std::uint64_t RealSenseStereo::frames_captured() const noexcept {
    return impl_->frames.load(std::memory_order_relaxed);
}

}  // namespace rt_deploy::vision
