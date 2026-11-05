//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

// RawDepthSource — librealsense2 implementation of the `kRaw` depth source.
//
// Verified against the librealsense2 API (rs.hpp, rs_pipeline.hpp,
// rs_sensor.hpp, rs_frame.hpp):
//   - rs2::config::enable_stream(RS2_STREAM_DEPTH, w, h, RS2_FORMAT_Z16, fps)
//   - rs2::pipeline::start(cfg) → rs2::pipeline_profile
//   - rs2::depth_sensor::get_depth_scale() → RS2_OPTION_DEPTH_UNITS [m/count]
//   - rs2::depth_sensor::set_option(RS2_OPTION_EMITTER_ENABLED, 0|1)
//   - rs2::video_stream_profile::get_intrinsics() → rs2_intrinsics {fx,fy,ppx,ppy}
//   - rs2::pipeline::try_wait_for_frames(&fs, ms); fs.get_depth_frame()
//
// Same PIMPL / device-open / error-handling / thread idiom as
// realsense_stereo.cpp — all librealsense symbols stay in this TU, so the rest
// of the tree builds without the SDK on its include path.
//
// The implementation is guarded by RT_DEPLOY_WITH_REALSENSE, which CMake
// defines for every TU of rt_deploy_vision.

#include "rt_deploy/vision/raw_depth_source.hpp"

#include "rt_deploy/vision/depth_ops.hpp"

#include <memory>
#include <string>
#include <utility>

#if defined(RT_DEPLOY_WITH_REALSENSE)
#    include "rt_deploy/core/clock.hpp"
#    include "rt_deploy/core/rate_probe.hpp"
#    include "rt_deploy/core/rt_thread.hpp"

#    include <librealsense2/rs.hpp>

#    include <atomic>
#    include <chrono>
#    include <cstddef>
#    include <cstdint>
#    include <cstdio>
#    include <optional>
#    include <string_view>
#    include <thread>
#    include <vector>
#endif

namespace rt_deploy::vision {

#if defined(RT_DEPLOY_WITH_REALSENSE)

namespace {

// Ceiling on the blocking wait. Long enough that a healthy 30 Hz stream never
// trips it, short enough that a wedged USB link surfaces in the log within a
// second instead of silently freezing the depth rate forever.
constexpr unsigned kFrameTimeoutMs = 1000U;

// Map the YAML thread block onto the core helper's struct. `name` must outlive
// the call — core::RtThreadConfig holds a string_view, not a string.
[[nodiscard]] core::RtThreadConfig to_rt(const ThreadRtConfig& t, std::string_view name) noexcept {
    core::RtThreadConfig c;
    c.fifo_priority = t.fifo_priority;
    c.cpu_affinity  = t.cpu_affinity;
    c.lock_memory   = t.lock_memory;
    c.name          = name;
    return c;
}

// Map librealsense's timestamp domain onto our own enum. Deliberately duplicated
// in realsense_stereo.cpp rather than hoisted into a shared header: the rs2 enum
// constants must stay inside the two TUs that already include the SDK.
//
// RS2_TIMESTAMP_DOMAIN_COUNT and anything a future SDK adds fall through to
// kUnavailable, which zeroes the transport-inclusive statistics rather than
// reporting them against a clock nobody identified.
[[nodiscard]] LatencyDomain to_latency_domain(rs2_timestamp_domain d) noexcept {
    switch (d) {
        case RS2_TIMESTAMP_DOMAIN_HARDWARE_CLOCK:
            return LatencyDomain::kHardware;
        case RS2_TIMESTAMP_DOMAIN_SYSTEM_TIME:
            return LatencyDomain::kSystemTime;
        case RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME:
            return LatencyDomain::kGlobalTime;
        case RS2_TIMESTAMP_DOMAIN_COUNT:
            break;
    }
    return LatencyDomain::kUnavailable;
}

}  // namespace

// ─── Impl ─────────────────────────────────────────────────────────────────────

struct RawDepthSource::Impl {
    Impl(VisionConfig cfg, core::TripleBuffer<g1::DepthFrame>& out) noexcept
        : config{std::move(cfg)}, inbox{out} {}

    VisionConfig                        config;
    core::TripleBuffer<g1::DepthFrame>& inbox;

    // Held in an optional because rs2::pipeline's constructor builds an
    // rs2::context and can throw: nothing that throws may run inside the
    // RawDepthSource constructor, so the device object is born in open(),
    // inside a try block, instead of as a plain member.
    std::optional<rs2::pipeline> pipeline;
    rs2::pipeline_profile        profile;

    StereoIntrinsics intr{};
    float            depth_scale = 0.001F;  // metres per Z16 count; read live in open()
    std::int32_t     src_w       = 0;       // negotiated capture width
    std::int32_t     src_h       = 0;       // negotiated capture height

    // Z16 → metres scratch at capture resolution. Sized once in open(), so the
    // capture loop never allocates.
    std::vector<float> depth_m;

    std::thread       thread;
    std::atomic<bool> running{false};

    // Capture-thread-private state. `seq` is the monotonic frame id this source
    // owns; the control loop reads it off the published frame to judge
    // staleness. `last_pub` drives the rate EMA.
    std::int64_t                seq = 0;
    core::MonotonicClock::Nanos last_pub{0};

    // Diagnostics — written only by the capture thread, read from anywhere.
    std::atomic<std::uint64_t> captured{0};
    std::atomic<std::uint64_t> published{0};
    std::atomic<std::uint64_t> overruns{0};
    std::atomic<std::uint64_t> errors{0};
    std::atomic<float>         last_ms{0.0F};
    std::atomic<float>         ema_hz{0.0F};

    // Latency distribution + drift for the 1 Hz log line, one tracker per clock
    // reference. Written ONLY by the capture thread (their single-writer
    // contract). ~8 KB each; Impl is heap-held, so no stack grows.
    //
    //   lat_arrival  from the CLOCK_MONOTONIC stamp taken the moment
    //                try_wait_for_frames() handed us the frameset. On THIS path
    //                that is a short interval by construction (there is no second
    //                stage — just Z16->metres and the reduce), which is exactly
    //                why the camera-stamp number below matters more here than it
    //                does on the cleaned path.
    //   lat_camera   from the D435i's own frame timestamp, projected onto
    //                CLOCK_MONOTONIC. Includes exposure, the ON-CHIP stereo
    //                match and USB transport — the whole reason `kRaw` looks
    //                cheap on CPU yet is not free in latency.
    LatencyTracker            lat_arrival;
    LatencyTracker            lat_camera;
    std::atomic<std::uint8_t> cam_domain{static_cast<std::uint8_t>(LatencyDomain::kUnavailable)};

    [[nodiscard]] core::Result<void, std::string> open();
    void                                          read_intrinsics();
    void                                          close() noexcept;
    void                                          run() noexcept;
    // `arrival_ns` is CLOCK_MONOTONIC when the driver handed us the frame;
    // `cam_mono_ns` is the camera's own stamp already projected onto the same
    // clock (0 = not reconcilable), and `cam_dom` labels which domain it came
    // from. All three are read in run(), inside its try block, because the
    // librealsense accessors throw and this function may not.
    void produce(const std::uint16_t* z16, std::size_t stride_px, std::int64_t arrival_ns,
                 std::int64_t cam_mono_ns, LatencyDomain cam_dom) noexcept;
};

core::Result<void, std::string> RawDepthSource::Impl::open() {
    using ResT = core::Result<void, std::string>;
    try {
        pipeline.emplace();

        rs2::config cfg;
        if (!config.serial.empty()) {
            cfg.enable_device(config.serial);
        }
        // The D435i's own on-chip stereo result — one Z16 stream, no IR pair,
        // and no GPU anywhere on this path.
        cfg.enable_stream(RS2_STREAM_DEPTH, config.width, config.height, RS2_FORMAT_Z16,
                          config.fps);

        profile = pipeline->start(cfg);

        rs2::device       dev    = profile.get_device();
        rs2::depth_sensor sensor = dev.first<rs2::depth_sensor>();

        // The opposite of the cleaned path: the on-chip block matcher genuinely
        // benefits from the projector's texture, so a fair raw baseline usually
        // wants `emitter: true`. Honour whatever the config asked for.
        if (sensor.supports(RS2_OPTION_EMITTER_ENABLED)) {
            sensor.set_option(RS2_OPTION_EMITTER_ENABLED, config.emitter ? 1.0F : 0.0F);
        }

        // Metres per Z16 count (RS2_OPTION_DEPTH_UNITS). 0.001 is only the
        // factory default — a High-Accuracy/High-Density preset or a re-flashed
        // calibration reports something else, and a wrong scale would silently
        // rescale every obstacle distance. Read it live.
        depth_scale = sensor.get_depth_scale();
        if (!(depth_scale > 0.0F)) {
            return ResT::Error{"D435i reported a non-positive depth scale"};
        }

        // Diagnostic, printed once at open: will this device hand us a
        // timestamp we can reconcile with CLOCK_MONOTONIC at all?
        //
        // RS2_OPTION_GLOBAL_TIME_ENABLED is what makes get_timestamp() report
        // RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME — the camera clock mapped onto host
        // system time by librealsense's own online linear fit — instead of the
        // raw hardware clock, whose epoch is unrelated to anything on the host
        // and therefore cannot yield a transport-inclusive latency. It defaults
        // ON for the D400 series; when it is off, the transport-inclusive
        // statistics read "unavailable" and this line is the explanation.
        //
        // Deliberately only REPORTED, never forced: silently reconfiguring the
        // device would perturb the very latency being measured.
        if (sensor.supports(RS2_OPTION_GLOBAL_TIME_ENABLED)) {
            const bool on = sensor.get_option(RS2_OPTION_GLOBAL_TIME_ENABLED) > 0.5F;
            std::fprintf(stderr,
                         "[raw_depth] global_time=%s -> camera-stamp (transport-inclusive) "
                         "latency %s\n",
                         on ? "ON" : "OFF", on ? "AVAILABLE" : "UNAVAILABLE (enable "
                                                "RS2_OPTION_GLOBAL_TIME_ENABLED)");
        } else {
            std::fprintf(stderr,
                         "[raw_depth] global_time unsupported by this device/firmware -> "
                         "camera-stamp (transport-inclusive) latency UNAVAILABLE\n");
        }

        read_intrinsics();
        if (src_w <= 0 || src_h <= 0) {
            return ResT::Error{"depth stream profile reported a non-positive resolution"};
        }

        // Last allocation on this path; everything after start() is in place.
        depth_m.assign(static_cast<std::size_t>(src_w) * static_cast<std::size_t>(src_h), 0.0F);
    } catch (const rs2::error& e) {
        return ResT::Error{std::string{"librealsense error ("} + e.get_failed_function()
                           + "): " + e.what()};
    } catch (const std::exception& e) {
        return ResT::Error{std::string{"RealSense depth open failed: "} + e.what()};
    }
    return ResT{};
}

void RawDepthSource::Impl::read_intrinsics() {
    const auto vsp = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    const rs2_intrinsics k = vsp.get_intrinsics();

    // Trust the negotiated profile over the config: it is what the frames will
    // actually carry, and the scratch is sized from it.
    src_w = static_cast<std::int32_t>(vsp.width());
    src_h = static_cast<std::int32_t>(vsp.height());

    intr.fx = k.fx;
    intr.fy = k.fy;
    intr.cx = k.ppx;
    intr.cy = k.ppy;
    // No baseline: the on-chip matcher already resolved the stereo geometry, so
    // nothing here evaluates fx*B/disp. Reporting a number no consumer uses
    // would only invite someone to trust it. (config.fx_override /
    // baseline_m_override are cleaned-path knobs for the same reason — they
    // feed the disparity conversion this path does not have.)
    intr.baseline_m  = 0.0F;
    intr.width       = src_w;
    intr.height      = src_h;
    intr.depth_scale = depth_scale;

    std::fprintf(stderr, "[raw_depth] %dx%d@%d  fx=%.1fpx  depth_units=%.6f m  emitter=%s\n",
                 static_cast<int>(src_w), static_cast<int>(src_h), static_cast<int>(config.fps),
                 static_cast<double>(intr.fx), static_cast<double>(depth_scale),
                 config.emitter ? "ON" : "OFF");
}

void RawDepthSource::Impl::close() noexcept {
    if (!pipeline.has_value()) {
        return;
    }
    try {
        pipeline->stop();
    } catch (const std::exception&) {
        // Already stopped, or the device was unplugged — nothing actionable at
        // shutdown, and stop() must not throw.
    }
    pipeline.reset();
}

// ─── Produce path (no allocation, no exception) ───────────────────────────────

void RawDepthSource::Impl::produce(const std::uint16_t* z16, std::size_t stride_px,
                                   std::int64_t arrival_ns, std::int64_t cam_mono_ns,
                                   LatencyDomain cam_dom) noexcept {
    const auto t0 = core::MonotonicClock::now();

    const auto w = static_cast<std::size_t>(src_w);
    const auto h = static_cast<std::size_t>(src_h);

    // Z16 → metres, RAW. Deliberately NOT sanitized here: the reducer below fuses
    // the invalid/clamp rule into the reduction, because Reduce::kAreaValid has to
    // know which samples the sensor actually measured. Once a plane is sanitized,
    // a "no return" and a genuine far reading are both clip_max and can no longer
    // be told apart.
    for (std::size_t y = 0; y < h; ++y) {
        const std::uint16_t* src = z16 + (y * stride_px);
        float*               dst = depth_m.data() + (y * w);
        for (std::size_t x = 0; x < w; ++x) {
            dst[x] = static_cast<float>(src[x]) * depth_scale;
        }
    }

    // Straight into the wait-free slot — no intermediate copy — using the same
    // shared reducer the cleaned path's CUDA kernel mirrors, so the two sources
    // are directly A/B comparable. The output extent is the compile-time slot
    // size, which start() has already checked the config agrees with.
    //
    // RAW METRES: no clip/log1p here. The g1::ObservationBuilder image group
    // encodes downstream, and doing it twice would be a silent, invisible bug.
    g1::DepthFrame& slot = inbox.producer_slot();
    downsample_depth_raw_m(depth_m.data(), src_w, src_w, src_h, slot.data.data(),
                           g1::DepthFrame::kCols, g1::DepthFrame::kRows, config.clip_min_m,
                           config.clip_max_m, config.downsample);
    seq += 1;
    slot.seq = seq;
    // Publish stamp, for the control loop's per-tick depth-age column. Same
    // clock the control thread times its ticks with, so the consumer can
    // subtract the two directly.
    const std::int64_t pub_ns = core::MonotonicClock::now().count();
    slot.stamp_ns             = pub_ns;
    inbox.publish();  // nothing may touch `slot` after this point
    published.fetch_add(1, std::memory_order_relaxed);

    const auto t1 = core::MonotonicClock::now();
    last_ms.store(static_cast<float>(static_cast<double>((t1 - t0).count()) * 1.0e-6),
                  std::memory_order_relaxed);

    // Publish-rate EMA, same smoothing as the cleaned path so the two sources'
    // rate readouts are comparable.
    if (last_pub.count() != 0) {
        const double dt_s = static_cast<double>((t1 - last_pub).count()) * 1.0e-9;
        if (dt_s > 0.0) {
            const float inst = static_cast<float>(1.0 / dt_s);
            const float prev = ema_hz.load(std::memory_order_relaxed);
            ema_hz.store(prev > 0.0F ? (0.9F * prev) + (0.1F * inst) : inst,
                         std::memory_order_relaxed);
        }
    }
    last_pub = t1;

    // ── Capture stamp -> availability ────────────────────────────────────────
    //
    // LAST in produce(), deliberately: `last_ms` and the EMA above measure only
    // the conversion and publish, so the ~6 ns spent here is outside both.
    //
    // Measured to `pub_ns` — the instant before the release CAS makes the slot
    // visible, and the SAME number the control loop reads as this frame's stamp.
    // "Capture -> available" here and the loop's "available -> consumed" then
    // abut exactly rather than overlapping by a clock read, so the two
    // instruments sum to the true end-to-end age with nothing double-counted.
    //
    // Two trackers because there are two honest answers, and neither may be
    // passed off as the other — see vision::LatencyDomain.
    // Both record() calls are allocation-free, lock-free and O(1).
    lat_arrival.record(pub_ns - arrival_ns, pub_ns);
    if (cam_mono_ns != 0) {
        lat_camera.record(pub_ns - cam_mono_ns, pub_ns);
    }
    // Latched every frame rather than once: a relaxed byte store is cheaper than
    // the branch that would avoid it, and the domain is what makes the
    // camera-stamp latency interpretable at all.
    cam_domain.store(static_cast<std::uint8_t>(cam_dom), std::memory_order_relaxed);
}

void RawDepthSource::Impl::run() noexcept {
    if (const int rc = core::configure_current_thread(to_rt(config.capture_rt, "rs-depth"));
        rc != 0) {
        std::fprintf(stderr, "[raw_depth] configure_current_thread rc=%d (continuing)\n", rc);
    }

    static core::RateProbe probe{"rs-depth"};

    // rate_hz == 0 means free-run: the camera's own cadence paces the loop and
    // wait_for_frames is the only blocking point. Otherwise PeriodicTicker
    // decimates down to rate_hz (asking for more than `fps` just free-runs at
    // fps and charges every cycle an overrun).
    std::optional<core::PeriodicTicker> ticker;
    if (config.rate_hz > 0.0F) {
        ticker.emplace(std::chrono::nanoseconds{
            static_cast<std::int64_t>(1.0e9 / static_cast<double>(config.rate_hz))});
    }

    auto last_log = core::MonotonicClock::now();

    while (running.load(std::memory_order_acquire)) {
        probe.tick();

        if (ticker.has_value()) {
            const auto tick = ticker->wait_next();
            // PeriodicTicker resyncs to "now" rather than spinning to catch up,
            // so a long cycle degrades the depth rate smoothly and never backs
            // up. Count it — the control loop is unaffected either way.
            if (tick.jitter >= ticker->period()) {
                overruns.fetch_add(1, std::memory_order_relaxed);
            }
        }

        bool produced = false;
        try {
            rs2::frameset fs;
            if (!pipeline->try_wait_for_frames(&fs, kFrameTimeoutMs)) {
                // A single miss is not fatal: the device usually recovers on
                // its own, and exiting here would take depth down for good.
                std::fprintf(stderr, "[raw_depth] wait_for_frames timed out after %u ms\n",
                             kFrameTimeoutMs);
                errors.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            // Arrival instant: taken the moment the frameset is in our
            // hands, before any accessor runs, so the arrival latency covers
            // everything WE do to the frame and nothing the driver did first.
            const std::int64_t arrival_ns = core::MonotonicClock::now().count();

            const rs2::depth_frame df = fs.get_depth_frame();
            if (!df) {
                errors.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            captured.fetch_add(1, std::memory_order_relaxed);

            // The negotiated profile is fixed for the session; a frame that
            // disagrees would overrun the scratch, so drop it rather than
            // trust it.
            if (df.get_width() != src_w || df.get_height() != src_h) {
                errors.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            // Transport-inclusive: the camera's own frame timestamp,
            // projected onto CLOCK_MONOTONIC only when its domain permits (see
            // vision::LatencyDomain). Both accessors are metadata reads — no
            // syscall, no allocation — but they THROW, which is why they run here
            // in the try block rather than inside noexcept produce().
            const LatencyDomain cam_dom = to_latency_domain(df.get_frame_timestamp_domain());
            const std::int64_t  cam_ns =
                camera_stamp_to_monotonic_ns(df.get_timestamp(), cam_dom);

            auto scope = probe.scope();
            // Stride is reported in bytes and is always Z16-aligned; convert to
            // samples so the row walk needs no byte→uint16 pointer cast.
            produce(static_cast<const std::uint16_t*>(df.get_data()),
                    static_cast<std::size_t>(df.get_stride_in_bytes()) / sizeof(std::uint16_t),
                    arrival_ns, cam_ns, cam_dom);
            produced = true;
        } catch (const rs2::error& e) {
            std::fprintf(stderr, "[raw_depth] librealsense error (%s): %s\n",
                         e.get_failed_function().c_str(), e.what());
            errors.fetch_add(1, std::memory_order_relaxed);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[raw_depth] frame error: %s\n", e.what());
            errors.fetch_add(1, std::memory_order_relaxed);
        }

        if (produced && config.profile_period_s > 0.0F) {
            const auto   now   = core::MonotonicClock::now();
            const double since = static_cast<double>((now - last_log).count()) * 1.0e-9;
            if (since >= static_cast<double>(config.profile_period_s)) {
                last_log = now;
                // Live view. summarize() walks 1024 buckets and does one
                // division; at the operator-chosen profile_period_s (seconds, not
                // frames) that is dwarfed by the fprintf it feeds, and it is on
                // the capture thread, never the control thread.
                //
                // BOTH latencies are printed, each with what it means, because
                // the arrival number alone would understate the real
                // camera-to-policy age by the on-chip match plus USB transport —
                // and on this path that is most of it.
                const auto lat = lat_arrival.summarize();
                const auto cam = lat_camera.summarize();
                const auto dom =
                    static_cast<LatencyDomain>(cam_domain.load(std::memory_order_relaxed));
                // %.*s over the string_view, not std::string{}.c_str(): building a
                // std::string here would ALLOCATE on the capture thread, and the
                // whole point of this path is that it never does.
                const std::string_view dom_name = to_string(dom);
                std::fprintf(stderr,
                             "[raw_depth] published=%llu  produce=%.2f ms  rate=%.1f Hz  "
                             "overruns=%llu  errors=%llu\n"
                             "[raw_depth]   lat(arrival)   p50/p99/max %.1f/%.1f/%.1f ms  "
                             "trend %+.2f ms/min  mean %.2f Hz  n=%llu\n"
                             "[raw_depth]   lat(cam:%.*s) p50/p99/max %.1f/%.1f/%.1f ms  "
                             "trend %+.2f ms/min  n=%llu  unreconciled=%llu\n",
                             static_cast<unsigned long long>(
                                 published.load(std::memory_order_relaxed)),
                             static_cast<double>(last_ms.load(std::memory_order_relaxed)),
                             static_cast<double>(ema_hz.load(std::memory_order_relaxed)),
                             static_cast<unsigned long long>(
                                 overruns.load(std::memory_order_relaxed)),
                             static_cast<unsigned long long>(
                                 errors.load(std::memory_order_relaxed)),
                             static_cast<double>(lat.p50_ms), static_cast<double>(lat.p99_ms),
                             static_cast<double>(lat.max_ms),
                             static_cast<double>(lat.slope_ms_per_min),
                             static_cast<double>(lat.mean_hz),
                             static_cast<unsigned long long>(lat.samples),
                             static_cast<int>(dom_name.size()), dom_name.data(),
                             static_cast<double>(cam.p50_ms), static_cast<double>(cam.p99_ms),
                             static_cast<double>(cam.max_ms),
                             static_cast<double>(cam.slope_ms_per_min),
                             static_cast<unsigned long long>(cam.samples),
                             static_cast<unsigned long long>(cam.unreconciled));
            }
        }
    }
}

// ─── Public wrapper ───────────────────────────────────────────────────────────

RawDepthSource::RawDepthSource(VisionConfig config, core::TripleBuffer<g1::DepthFrame>& inbox)
    : impl_{std::make_unique<Impl>(std::move(config), inbox)} {}

RawDepthSource::~RawDepthSource() {
    stop();
}

core::Result<void, std::string> RawDepthSource::start() {
    using ResT = core::Result<void, std::string>;
    if (impl_->running.load(std::memory_order_acquire)) {
        return ResT{};  // idempotent
    }

    // The sink is a compile-time 48x84 slot and produce() writes it directly, so
    // a disagreeing config would run off the end. make_depth_source() calls
    // VisionConfig::validate() first, but a direct caller may not — refuse here
    // rather than assume.
    if (impl_->config.out_width != g1::DepthFrame::kCols
        || impl_->config.out_height != g1::DepthFrame::kRows) {
        return ResT::Error{"out_width/out_height must be " + std::to_string(g1::DepthFrame::kCols)
                           + "x" + std::to_string(g1::DepthFrame::kRows)
                           + " to match g1::DepthFrame"};
    }

    if (auto r = impl_->open(); !r) {
        // The pipeline may already be streaming when open() bailed late.
        impl_->close();
        return r;
    }

    impl_->running.store(true, std::memory_order_release);
    try {
        impl_->thread = std::thread([this]() { impl_->run(); });
    } catch (const std::exception& e) {
        // Leave nothing running behind a failed start().
        impl_->running.store(false, std::memory_order_release);
        impl_->close();
        return ResT::Error{std::string{"failed to spawn rs-depth thread: "} + e.what()};
    }
    return ResT{};
}

void RawDepthSource::stop() noexcept {
    if (impl_ == nullptr) {
        return;
    }
    impl_->running.store(false, std::memory_order_release);
    // stop() is noexcept and runs from the destructor, so an escaping exception
    // would call std::terminate during teardown — losing the damping command and
    // any diagnostics with it. join() can throw system_error, and close() runs
    // librealsense, which throws freely. Swallow and log: by this point we are
    // shutting down and there is nothing left to recover.
    try {
        if (impl_->thread.joinable()) {
            impl_->thread.join();
        }
        // Release the device only once the thread that reads it is gone.
        impl_->close();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[raw_depth] teardown: %s\n", e.what());
    } catch (...) {
        std::fprintf(stderr, "[raw_depth] teardown: unknown exception\n");
    }
}

DepthStats RawDepthSource::stats() const noexcept {
    DepthStats s;
    s.captured  = impl_->captured.load(std::memory_order_relaxed);
    s.published = impl_->published.load(std::memory_order_relaxed);
    // `dropped` stays 0: there is no second stage here for a captured frame to
    // be superseded before. Frames decimated by rate_hz < fps are discarded
    // inside librealsense's queue, where we cannot count them.
    s.dropped  = 0;
    s.overruns = impl_->overruns.load(std::memory_order_relaxed);
    s.errors   = impl_->errors.load(std::memory_order_relaxed);
    s.last_ms  = impl_->last_ms.load(std::memory_order_relaxed);
    s.ema_hz   = impl_->ema_hz.load(std::memory_order_relaxed);
    return s;
}

StereoIntrinsics RawDepthSource::intrinsics() const noexcept {
    // Written once by open(), before the capture thread exists, and const for
    // the session.
    return impl_->intr;
}

#else  // !RT_DEPLOY_WITH_REALSENSE

// CMake only compiles this TU into rt_deploy_vision, which exists only with
// RT_DEPLOY_WITH_REALSENSE=ON and defines it PUBLIC.
#    error "raw_depth_source.cpp requires RT_DEPLOY_WITH_REALSENSE"

#endif  // RT_DEPLOY_WITH_REALSENSE

}  // namespace rt_deploy::vision
