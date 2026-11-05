//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

// DepthPipeline — the `kCleaned` producer, end to end.
//
// This TU owns the two lifecycle questions the header states but cannot
// express: what a failed start() unwinds, and what the `depth-infer` thread
// does when the frames stop coming.
//
// Two producer threads, neither of which ever touches the control thread:
//
//   `rs-capture`  (RealSenseStereo) blocks in wait_for_frames() and publishes
//                 the IR pair into frame_tb_.
//   `depth-infer` (this file) wakes on a core::PeriodicTicker at rate_hz,
//                 takes the NEWEST pair, runs the engine, and memcpys the
//                 84x48 metres plane into the control loop's inbox.
//
// The hand-off to the control loop is one 16 KB memcpy plus a release CAS on
// the producer side, and a wait-free try_acquire_latest() on the reader side —
// no mutex, no condition variable, no allocation after start().
//
// Every failure mode degrades depth FRESHNESS only. A slow inference lowers the
// depth rate (PeriodicTicker resyncs to "now" rather than spinning to catch up
// on missed ticks), an engine error skips a frame, and the startup guard below
// takes down this thread alone. In all of them the control loop keeps reading
// its last-known-good frame at full rate; nothing here can stall it.
//
// Logging is throttled: fprintf to stderr can block on a slow terminal, which
// is exactly the kind of unbounded wait a produce path must not take per frame.

#include "rt_deploy/vision/depth_pipeline.hpp"

#include "rt_deploy/core/clock.hpp"
#include "rt_deploy/core/rate_probe.hpp"
#include "rt_deploy/core/rt_thread.hpp"

// Only for cudaSetDevice() on the inference thread — see infer_loop(). This TU
// is compiled only on the RT_DEPLOY_WITH_TRT path, which links CUDA::cudart.
#include <cuda_runtime_api.h>

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <thread>
#include <utility>

namespace rt_deploy::vision {

using inference::InferenceStatus;

// The publish is a fixed-size memcpy out of the engine scratch, so the scratch
// must be able to hold a whole g1::DepthFrame plane. VisionConfig::validate()
// pins the config side of the same identity; start() re-checks the live config.
static_assert(kMaxOutPixels >= g1::DepthFrame::kCount,
              "vision::DepthFrame::depth_m is too small to hold a g1::DepthFrame plane");

namespace {

// Linux truncates pthread names at 15 characters; keep both under that so `top
// -H` and the RT traces stay readable.
constexpr std::string_view kCaptureThreadName{"rs-capture"};
constexpr std::string_view kInferThreadName{"depth-infer"};

// The camera is either there or it is not. 10 s covers USB enumeration plus the
// D435i's auto-exposure warm-up on a cold Orin.
constexpr auto kStartupTimeout = std::chrono::seconds{10};

// Cadence of every in-loop log line (status and errors alike).
constexpr auto kLogPeriod = std::chrono::seconds{1};

// Free-run (rate_hz <= 0) has no ticker to pace it, so it polls the capture
// hand-off at ~1 kHz instead of burning a core on a spin.
constexpr auto kIdlePoll = std::chrono::milliseconds{1};

core::RtThreadConfig to_rt(const ThreadRtConfig& t, std::string_view name) noexcept {
    core::RtThreadConfig c;
    c.fifo_priority = t.fifo_priority;
    c.cpu_affinity  = t.cpu_affinity;
    c.lock_memory   = t.lock_memory;
    c.name          = name;
    return c;
}

}  // namespace

// ─── Construction / teardown ─────────────────────────────────────────────────

DepthPipeline::DepthPipeline(VisionConfig config, core::TripleBuffer<g1::DepthFrame>& inbox)
    : config_{std::move(config)}, inbox_{inbox} {}

DepthPipeline::~DepthPipeline() {
    // Joins the inference thread and stops the capture thread. The members then
    // destruct in reverse declaration order, so the camera (which publishes into
    // *frame_tb_) is gone before frame_tb_ is.
    stop();
}

// ─── Startup ─────────────────────────────────────────────────────────────────

core::Result<void, std::string> DepthPipeline::start() {
    using ResT = core::Result<void, std::string>;

    if (started_.load(std::memory_order_acquire)) {
        return ResT::Error{"start: already started"};
    }

    // A failed start() must leave nothing running: the caller is free to fall
    // back to kRaw, or to no depth at all, and keep flying.
    const auto fail = [this](std::string_view stage, std::string_view reason) {
        if (camera_) {
            camera_->stop();
        }
        engine_.reset();
        camera_.reset();
        scratch_.reset();
        frame_tb_.reset();
        return ResT::Error{std::string{stage} + ": " + std::string{reason}};
    };

    // Checked before anything is acquired: the publish path is a fixed-size
    // memcpy, so a wrong output geometry would scramble the observation rather
    // than fail. VisionConfig::validate() already rejects it, but the pipeline
    // is constructible from a config that never went through the factory.
    //
    // Both dimensions are compared, not just their product: 48x84 has the right
    // pixel count and would memcpy cleanly into the slot, but the kernel writes
    // row-major at out_width, so the policy would receive a transposed image
    // with no error anywhere. Matches RawDepthSource::start()'s check.
    if (config_.out_width != g1::DepthFrame::kCols
        || config_.out_height != g1::DepthFrame::kRows) {
        return ResT::Error{"output: out_width x out_height = "
                           + std::to_string(config_.out_width) + "x"
                           + std::to_string(config_.out_height)
                           + " but the student_depth observation is fixed at "
                           + std::to_string(g1::DepthFrame::kCols) + "x"
                           + std::to_string(g1::DepthFrame::kRows)
                           + " (g1::DepthFrame::kCols x kRows)"};
    }

    // ~6 MB of StereoFrame slots + a ~256 KB scratch, all up front so the
    // produce path never allocates.
    try {
        frame_tb_ = std::make_unique<core::TripleBuffer<StereoFrame>>();
        scratch_  = std::make_unique<DepthFrame>();
    } catch (const std::exception& e) {
        return fail("buffers", e.what());
    }

    RealSenseStereoConfig cam_cfg;
    cam_cfg.width     = config_.width;
    cam_cfg.height    = config_.height;
    cam_cfg.fps       = config_.fps;
    cam_cfg.emitter   = config_.emitter;
    cam_cfg.serial    = config_.serial;
    cam_cfg.rt_thread = to_rt(config_.capture_rt, kCaptureThreadName);

    // librealsense builds its context inside the rs2::pipeline member, and it
    // signals failure by throwing.
    try {
        camera_ = std::make_unique<RealSenseStereo>(*frame_tb_, cam_cfg);
    } catch (const std::exception& e) {
        return fail("camera", e.what());
    }
    if (auto r = camera_->start(); !r) {
        return fail("camera", r.error());
    }

    intr_ = camera_->intrinsics();
    // Overrides pin the geometry for offline replay of recorded IR, where the
    // attached camera is not the one the plan was calibrated against.
    if (config_.fx_override > 0.0F) {
        intr_.fx = config_.fx_override;
    }
    if (config_.baseline_m_override > 0.0F) {
        intr_.baseline_m = config_.baseline_m_override;
    }
    // fx*B == 0 would send every depth to infinity, hence to clip_max, hence to
    // an all-far image the policy reads as empty space. Fail loudly instead.
    if (!(intr_.fx_baseline() > 0.0F)) {
        return fail("intrinsics", "fx*baseline is not positive (fx=" + std::to_string(intr_.fx)
                                      + "px, baseline=" + std::to_string(intr_.baseline_m)
                                      + "m); set fx_override / baseline_m_override");
    }

    StereoDepthEngineConfig eng_cfg;
    eng_cfg.plan_path      = config_.plan_path;
    eng_cfg.device_id      = config_.device_id;
    eng_cfg.use_cuda_graph = config_.use_cuda_graph;
    eng_cfg.width          = config_.width;
    eng_cfg.height         = config_.height;
    eng_cfg.out_width      = config_.out_width;
    eng_cfg.out_height     = config_.out_height;
    eng_cfg.fx_baseline    = intr_.fx_baseline();
    eng_cfg.clip_min_m     = config_.clip_min_m;
    eng_cfg.clip_max_m     = config_.clip_max_m;
    eng_cfg.downsample     = config_.downsample;

    auto eng = make_stereo_depth_engine(eng_cfg);
    if (!eng) {
        return fail("engine", eng.error());
    }
    engine_ = std::move(eng).value();

    // MANDATORY before the first infer(): with CUDA graphs enabled these
    // parameters are baked into the captured graph, so a later call is a silent
    // no-op and the depth would come out scaled by whatever the plan defaults
    // to. The engine config above already carries them; this is the belt to its
    // braces, and the only supported ordering.
    engine_->set_output_params(intr_.fx_baseline(), config_.clip_min_m, config_.clip_max_m);

    stop_.store(false, std::memory_order_release);
    try {
        thread_ = std::thread{[this]() { infer_loop(); }};
    } catch (const std::exception& e) {
        return fail("thread", e.what());
    }
    started_.store(true, std::memory_order_release);

    std::fprintf(stderr,
                 "[depth_pipeline] started  backend=%s  zero_copy=%s  net=%dx%d  out=%dx%d  "
                 "reduce=%s  clip=[%.2f,%.2f]m  fx*B=%.3f  rate=%.1fHz\n",
                 std::string{engine_->backend_name()}.c_str(),
                 engine_->zero_copy() ? "yes" : "no", config_.width, config_.height,
                 config_.out_width, config_.out_height,
                 std::string{to_string(config_.downsample)}.c_str(),
                 static_cast<double>(config_.clip_min_m), static_cast<double>(config_.clip_max_m),
                 static_cast<double>(intr_.fx_baseline()), static_cast<double>(config_.rate_hz));
    return ResT{};
}

void DepthPipeline::stop() noexcept {
    stop_.store(true, std::memory_order_release);
    // Nothing below is allowed to escape: stop() is noexcept, is called from the
    // destructor, and must be safe after a failed start() (no thread, no camera)
    // and on a second call (thread already joined, RealSenseStereo::stop() is
    // itself idempotent).
    try {
        if (thread_.joinable()) {
            thread_.join();
        }
        if (camera_) {
            camera_->stop();
        }
    } catch (...) {
        std::fprintf(stderr, "[depth_pipeline] teardown threw; continuing\n");
    }
    // The camera and engine stay alive so stats() and intrinsics() keep
    // answering after a stop; the destructor releases them.
    started_.store(false, std::memory_order_release);
}

DepthStats DepthPipeline::stats() const noexcept {
    DepthStats s;
    s.captured  = camera_ ? camera_->frames_captured() : std::uint64_t{0};
    s.published = published_.load(std::memory_order_relaxed);
    s.dropped   = dropped_.load(std::memory_order_relaxed);
    s.overruns  = overruns_.load(std::memory_order_relaxed);
    s.errors    = errors_.load(std::memory_order_relaxed);
    s.last_ms   = last_ms_.load(std::memory_order_relaxed);
    s.ema_hz    = ema_hz_.load(std::memory_order_relaxed);
    return s;
}

// ─── Produce loop ────────────────────────────────────────────────────────────

void DepthPipeline::infer_loop() noexcept {
    // First thing on the thread: drop to the configured priority/core, BELOW
    // the control loop and, where a core can be spared, off its CPU.
    const core::RtThreadConfig tc = to_rt(config_.inference_rt, kInferThreadName);
    if (const int rc = core::configure_current_thread(tc); rc != 0) {
        std::fprintf(stderr, "[depth_pipeline] configure_current_thread rc=%d (continuing)\n", rc);
    }

    // The CUDA "current device" is a THREAD-LOCAL property, and the only
    // cudaSetDevice() in the pipeline runs inside make_stereo_depth_engine() on
    // whichever thread called start(). Without this the engine's kernels would be
    // submitted from here to device 0 while its buffers live on device_id — inert
    // on the single-GPU Orin, an illegal access on a multi-GPU dev box. A wrong
    // device is not a freshness problem, so the thread exits rather than submit
    // work that would fault; the control loop keeps running without depth.
    if (const cudaError_t e = cudaSetDevice(static_cast<int>(config_.device_id));
        e != cudaSuccess) {
        errors_.fetch_add(1, std::memory_order_relaxed);
        std::fprintf(stderr,
                     "[depth_pipeline] cudaSetDevice(%d) failed: %s; inference thread exiting, "
                     "depth stays unavailable\n",
                     static_cast<int>(config_.device_id), cudaGetErrorString(e));
        return;
    }

    // Function-local so the registry entry appears only when the pipeline runs;
    // constructed once, never on a cycle. Matches the capture thread's probe.
    static core::RateProbe probe{"infer"};

    using Nanos = core::MonotonicClock::Nanos;

    const bool  rate_limited = config_.rate_hz > 0.0F;
    const Nanos period       = rate_limited
                                   ? Nanos{static_cast<std::int64_t>(
                                       1.0e9 / static_cast<double>(config_.rate_hz))}
                                   : Nanos{0};
    // Driven only in the rate-limited branch; free-run paces itself on the
    // capture hand-off instead. wait_next() resyncs to "now" on an overrun, so a
    // slow inference degrades the rate smoothly and never backs up a queue.
    core::PeriodicTicker ticker{rate_limited ? period : Nanos{kIdlePoll}};

    const auto start_ns = core::MonotonicClock::now();
    auto       last_pub = start_ns;
    auto       last_log = start_ns;
    auto       last_err = start_ns - std::chrono::duration_cast<Nanos>(kLogPeriod);

    std::uint64_t prev_seq  = 0;
    bool          have_prev = false;  // processed a pair (gates drop accounting)
    bool          have_pub  = false;  // published once (gates the EMA's interval)
    float         ema_hz    = 0.0F;

    while (!stop_.load(std::memory_order_acquire)) {
        const Nanos cycle_begin =
            rate_limited ? ticker.wait_next().actual : core::MonotonicClock::now();
        if (stop_.load(std::memory_order_acquire)) {
            break;  // woke into a shutdown; don't submit more GPU work
        }
        probe.tick();

        const StereoFrame* f = frame_tb_->try_acquire_latest();
        if (f == nullptr) {
            // Startup guard: a camera that never produces is a wiring or
            // permissions problem, not something to retry silently forever.
            // There is no caller to return to, so it is logged, counted, and the
            // thread exits; the control loop keeps running without depth.
            if (!have_prev && (cycle_begin - start_ns) > kStartupTimeout) {
                errors_.fetch_add(1, std::memory_order_relaxed);
                std::fprintf(stderr,
                             "[depth_pipeline] no camera frames after %lld s: inference thread "
                             "exiting, depth stays unavailable\n",
                             static_cast<long long>(kStartupTimeout.count()));
                return;
            }
            if (!rate_limited) {
                std::this_thread::sleep_for(kIdlePoll);
            }
            continue;
        }

        // Everything the camera published between the pair we last processed and
        // this one was superseded before we ever looked at it. That is the whole
        // point of the latest-wins hand-off, but it is worth knowing how much of
        // the capture rate the inference stage is actually consuming.
        if (have_prev && f->seq > prev_seq + 1) {
            dropped_.fetch_add(f->seq - prev_seq - 1, std::memory_order_relaxed);
        }

        const Nanos     t0 = core::MonotonicClock::now();
        InferenceStatus st = InferenceStatus::kOk;
        {
            auto scope = probe.scope();
            st         = engine_->infer(*f, *scratch_);
        }
        const Nanos t1 = core::MonotonicClock::now();

        const float infer_ms =
            static_cast<float>(
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count())
            / 1000.0F;
        last_ms_.store(infer_ms, std::memory_order_relaxed);

        // The engine fills depth_m/width/height; the rest of the scratch frame
        // is ours (it travels with the vision::DepthFrame contract even though
        // the control loop only needs the plane and the sequence number).
        const bool usable =
            (st == InferenceStatus::kOk) && (scratch_->pixels() == g1::DepthFrame::kCount);
        if (usable) {
            scratch_->intr     = intr_;
            scratch_->seq      = f->seq;
            scratch_->stamp_ns = f->stamp_ns;
            scratch_->infer_ms = infer_ms;

            // RAW METRES into the sink — the ObservationBuilder image group is
            // what applies clip + log1p. Fill the slot, then publish; the slot
            // must not be touched afterwards.
            g1::DepthFrame& slot = inbox_.producer_slot();
            std::memcpy(slot.data.data(), scratch_->depth_m.data(),
                        g1::DepthFrame::kCount * sizeof(float));
            slot.seq = static_cast<std::int64_t>(f->seq);
            // Publish stamp, for the control loop's per-tick depth-age column.
            // Deliberately NOT f->stamp_ns (the CAPTURE time): capture→publish is
            // this pipeline's own latency and is reported separately, whereas
            // this field measures only the publish→consume hand-off leg.
            const std::int64_t pub_ns = core::MonotonicClock::now().count();
            slot.stamp_ns             = pub_ns;
            inbox_.publish();
            published_.fetch_add(1, std::memory_order_relaxed);

            // ── Arrival stamp -> availability ────────────────────────────────
            //
            // Measured to `pub_ns`, i.e. the instant before the release CAS makes
            // the slot visible, and to the SAME number the control loop reads as
            // this frame's stamp. "Capture -> available" here and the control
            // loop's "available -> consumed" then abut exactly rather than
            // overlapping by a clock read, so the two add up to the true age with
            // no double count.
            //
            // From f->stamp_ns, the host stamp taken when the driver handed us
            // the pair. Covers the capture->infer hand-off, the network, the
            // metres conversion, the reduce and the inbox memcpy. EXCLUDES
            // exposure and USB/uvc transport. record() is allocation-free,
            // lock-free and O(1); see vision::LatencyTracker.
            lat_arrival_.record(pub_ns - f->stamp_ns, pub_ns);

            if (have_pub) {
                const double dt_s = static_cast<double>((t1 - last_pub).count()) * 1e-9;
                const float  inst = dt_s > 0.0 ? static_cast<float>(1.0 / dt_s) : 0.0F;
                ema_hz = ema_hz > 0.0F ? (0.9F * ema_hz) + (0.1F * inst) : inst;
                ema_hz_.store(ema_hz, std::memory_order_relaxed);
            }
            last_pub = t1;
            have_pub = true;
        } else {
            // A bad frame costs depth freshness and nothing else, so the loop
            // keeps going even on kRuntimeError: the engine may recover, and
            // tearing the thread down here would guarantee it never does.
            errors_.fetch_add(1, std::memory_order_relaxed);
            if ((t1 - last_err) >= kLogPeriod) {
                last_err = t1;
                std::fprintf(stderr, "[depth_pipeline] infer status=%u out=%dx%d (frame dropped)\n",
                             static_cast<unsigned>(st), scratch_->width, scratch_->height);
            }
        }

        prev_seq  = f->seq;
        have_prev = true;

        const Nanos cycle_end = core::MonotonicClock::now();
        if (rate_limited && (cycle_end - cycle_begin) > period) {
            overruns_.fetch_add(1, std::memory_order_relaxed);
        }
        if ((cycle_end - last_log) >= kLogPeriod) {
            last_log = cycle_end;
            // Live view. summarize() walks 1024 buckets and does one division —
            // ~microseconds, once per second, on the 33 ms inference thread
            // (never on the control thread), and utterly dominated by the
            // fprintf it feeds. `lat` is the arrival-stamp latency.
            const auto lat = lat_arrival_.summarize();
            std::fprintf(stderr,
                         "[depth_pipeline] frame %llu  infer %.1f ms  rate %.1f Hz  "
                         "lat(arrival) p50/p99/max %.1f/%.1f/%.1f ms  trend %+.2f ms/min\n",
                         static_cast<unsigned long long>(
                             published_.load(std::memory_order_relaxed)),
                         static_cast<double>(infer_ms), static_cast<double>(ema_hz),
                         static_cast<double>(lat.p50_ms), static_cast<double>(lat.p99_ms),
                         static_cast<double>(lat.max_ms),
                         static_cast<double>(lat.slope_ms_per_min));
        }
    }
}

}  // namespace rt_deploy::vision
