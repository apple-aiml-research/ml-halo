//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// IDepthSource — the one thing a deploy app (via apps/g1/depth_setup.hpp) talks to.
//
// Both depth producers (learned-stereo `kCleaned` and on-chip `kRaw`) own their
// own threads and publish into the SAME sink: a wait-free
// core::TripleBuffer<g1::DepthFrame> that the g1 control loop drains with a
// single non-blocking try_acquire_latest() per tick. Selecting between them is
// a config flag, not a rebuild, and the control loop cannot tell them apart.
//
//   kCleaned:  [rs-capture] --TripleBuffer<StereoFrame>--> [depth-infer @30Hz]
//                  IR1/IR2                                   TRT + CUDA
//   kRaw:      [rs-depth @30Hz]
//                  Z16 -> metres -> downsample
//                                            ...both --> TripleBuffer<g1::DepthFrame>
//                                                              |
//                                                   [g1 control loop @50Hz]
//                                                    wait-free read, never blocks
//
// Neither producer ever touches the control thread: no shared mutex, no
// condition variable, no allocation after start(). If a producer stalls, dies,
// or is never started at all, the control loop keeps its last-known-good frame
// (and, before the first frame ever arrives, the observation builder's
// far-plane fallback). Depth latency degrades; control timing does not.

#include "rt_deploy/core/clock.hpp"
#include "rt_deploy/core/expected.hpp"
#include "rt_deploy/core/triple_buffer.hpp"
#include "rt_deploy/g1/depth_frame.hpp"
#include "rt_deploy/vision/camera_intrinsics.hpp"
#include "rt_deploy/vision/config.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <ratio>
#include <string>
#include <string_view>

namespace rt_deploy::vision {

// ─── Producer latency accounting ─────────────────────────────────────────────
//
// The producers' 1 Hz log lines report frame latency (source stamp -> publish)
// and whether it stays flat over the run. There are two different latencies,
// and only one of them can be measured without reconciling two clocks. They are
// reported separately and never substituted for one another:
//
//   ARRIVAL latency
//       publish time - the CLOCK_MONOTONIC stamp taken the instant the driver
//       handed the frame to us (StereoFrame::stamp_ns / the rs-depth loop's own
//       stamp). One clock, no reconciliation, always available. It covers
//       everything WE are responsible for: the capture->inference hand-off, the
//       network, the disparity->metres conversion, the reduce, and the memcpy
//       into the control loop's inbox. It EXCLUDES exposure, on-camera
//       processing and USB/uvc transport.
//
//   CAMERA latency (raw producer only)
//       publish time - the camera's OWN frame timestamp, projected onto
//       CLOCK_MONOTONIC. This is the end-to-end number, and it needs
//       (a) librealsense to report the stamp in a host-clock domain and
//       (b) a monotonic<->realtime offset. When either is missing nothing is
//       recorded and the domain is logged as unavailable, so a
//       transport-excluding number can never be mistaken for an end-to-end one.
//
// Everything below is fixed-cost per frame: a bucketed histogram plus five
// running sums. Nothing allocates, nothing sorts, nothing grows.

// Which clock librealsense measured a frame timestamp against. Mirrors
// rs2_timestamp_domain, kept as our own enum so this header stays free of the
// SDK (the rs2 -> LatencyDomain mapping lives in the two librealsense TUs).
enum class LatencyDomain : std::uint8_t {
    // No camera stamp travelled with the frame, or its domain is not comparable
    // with CLOCK_MONOTONIC. No camera latency is recorded.
    kUnavailable = 0,
    // RS2_TIMESTAMP_DOMAIN_HARDWARE_CLOCK — the camera's own clock. Arbitrary
    // epoch and unknown drift against the host, so it CANNOT be turned into a
    // host latency without a clock-offset estimator that nothing here runs. It
    // is reported by name rather than converted into a plausible-looking lie.
    kHardware = 1,
    // RS2_TIMESTAMP_DOMAIN_SYSTEM_TIME — host system clock, stamped by the
    // driver on frame ARRIVAL. Reconcilable, but being an arrival stamp it
    // excludes exposure and most of transport, much like our own stamp: treat
    // the camera latency in this domain as a cross-check, not as end-to-end.
    kSystemTime = 2,
    // RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME — the camera clock mapped onto the host
    // system clock by librealsense's own online linear fit
    // (RS2_OPTION_GLOBAL_TIME_ENABLED, on by default on the D400 series). The
    // only domain that yields a genuinely transport-inclusive latency. Its own
    // fit residual is tens to hundreds of
    // microseconds, and worst right after streaming starts.
    kGlobalTime = 3,
};

[[nodiscard]] constexpr std::string_view to_string(LatencyDomain d) noexcept {
    switch (d) {
        case LatencyDomain::kHardware:
            return "hw_clock(NOT reconciled: excludes transport)";
        case LatencyDomain::kSystemTime:
            return "system_time(driver arrival: excludes transport)";
        case LatencyDomain::kGlobalTime:
            return "global_time(camera stamp: INCLUDES transport)";
        case LatencyDomain::kUnavailable:
            break;
    }
    return "unavailable";
}

// Project a librealsense frame timestamp onto CLOCK_MONOTONIC nanoseconds.
// Returns 0 (== unavailable) for any domain that is not host-comparable.
//
// librealsense reports rs2_time_t in MILLISECONDS on the host SYSTEM clock (its
// time service is std::chrono::system_clock, i.e. CLOCK_REALTIME), while every
// timestamp in rt_deploy is CLOCK_MONOTONIC. The two need an offset, and the
// offset is sampled right here, once per frame, by reading both clocks back to
// back — two vDSO clock_gettime calls, tens of nanoseconds, no allocation, no
// lock, safe on a produce path.
//
// Per frame rather than once at start-up on purpose: CLOCK_REALTIME can be
// STEPPED by NTP at any moment, and a start-up offset would then be silently
// wrong for the rest of a multi-minute run. Sampling per frame confines a step to the one frame that straddles it
// (and shows up as a lone outlier, not a trend), and leaves slew as a ~ppm
// error. It costs one extra clock read per frame at 30 Hz.
//
// The subtraction is deliberately done in the MILLISECOND domain. Both stamps
// are ~1.7e12 ms since the Unix epoch; scaling that to nanoseconds first lands
// at ~1.7e18, past a double's 2^53 exact-integer range, and would quantise the
// result to ~256 ns. Differencing first keeps the magnitude small, so the error
// is just each stamp's own representation error (~0.25 us).
[[nodiscard]] inline std::int64_t camera_stamp_to_monotonic_ns(double        stamp_ms,
                                                               LatencyDomain domain) noexcept {
    if (domain != LatencyDomain::kSystemTime && domain != LatencyDomain::kGlobalTime) {
        return 0;
    }
    if (!(stamp_ms > 0.0)) {
        return 0;  // SDK handed us nothing usable
    }
    const std::int64_t mono_ns = core::MonotonicClock::now().count();
    const double       real_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::system_clock::now().time_since_epoch())
                               .count();
    const double age_ms = real_ms - stamp_ms;
    return mono_ns - static_cast<std::int64_t>(age_ms * 1.0e6);
}

// Streaming latency statistics for ONE depth producer, at fixed cost per frame.
//
// record() is the produce path: one bucket increment plus a handful of relaxed
// atomic loads/stores (measured at ~3 ns/call). No allocation, no sort, no
// growing vector, no lock — the same discipline as the DepthStats counters.
// summarize() does all the arithmetic and is explicitly NOT for the
// produce path (~0.06 us typical, ~0.6 us worst case).
//
// SINGLE WRITER: exactly one thread may call record() (the producer thread that
// owns the tracker); any number of threads may call summarize(). The members are
// relaxed atomics rather than plain fields so the cross-thread read in
// summarize() is not a data race. Only ONE ordering constraint exists in the
// whole class — a release/acquire pair on have_first_, taken once on the first
// sample, so that first_ns_ cannot be seen unset while n_ already counts. Every
// other field is relaxed and the set is therefore NOT a consistent snapshot; it
// is for logging, never for control decisions.
class LatencyTracker {
public:
    // 1024 buckets x 0.5 ms = 0..512 ms. 0.5 ms resolves the 33.3 ms frame
    // period ~67 ways (plenty to see a one-frame-period step), and 512 ms is far
    // beyond any usable depth latency, so on a healthy run the overflow bucket
    // stays empty. 8 KB of counters, allocated once with
    // the producer object.
    static constexpr std::size_t  kBucketCount = 1024;
    static constexpr std::int64_t kBucketNs    = 500'000;  // 0.5 ms
    static constexpr float        kBucketMs    = 0.5F;

    // A trend fitted over a couple of seconds is noise, not a trend. Below
    // either threshold summarize() reports exactly 0 ms/min. `samples` and
    // `mean_hz` are what separate "flat" from "not enough data yet": the fitted
    // window is (samples - 1) / mean_hz seconds.
    static constexpr std::uint64_t kMinSlopeSamples = 64;
    // Samples dropped from the slope regression at the start of a run. 60 is
    // two seconds at 30 Hz — comfortably past the first-inference transient
    // (TRT autotune + CUDA-graph capture, ~1.3 s measured on an Orin NX)
    // without discarding real steady-state signal.
    static constexpr std::uint64_t kSlopeWarmupSamples = 60;
    static constexpr double        kMinSlopeSpanMin = 0.25;  // 15 s

    // Produce path. Call exactly once per publish, from the producer thread.
    //   lat_ns      source stamp -> publish. MAY be negative when a reconciled
    //               camera stamp is bad; those are counted, never hidden.
    //   publish_ns  CLOCK_MONOTONIC at publish — the abscissa of the trend fit
    //               and the basis of mean_hz.
    void record(std::int64_t lat_ns, std::int64_t publish_ns) noexcept {
        const std::uint64_t n = n_.load(std::memory_order_relaxed);
        if (n == 0U) {
            first_ns_.store(publish_ns, std::memory_order_relaxed);
            // The ONE release store in this class, and only on the first sample.
            // It pairs with the acquire in summarize() so that a reader which
            // sees the flag set is guaranteed to see first_ns_ too. Without it,
            // an all-relaxed reader could observe n_ > 0 with first_ns_ still
            // unset and compute a span of "however long this machine has been
            // up", i.e. mean_hz ~ 0 for a producer running perfectly at 30 Hz.
            // A flag rather than a `first_ns_ != 0` sentinel because a valid
            // CLOCK_MONOTONIC reading of exactly 0 is not impossible, and a
            // statistic that silently switches off at one clock value is the kind
            // of thing nobody ever finds again.
            have_first_.store(true, std::memory_order_release);
        }
        n_.store(n + 1U, std::memory_order_relaxed);
        last_ns_.store(publish_ns, std::memory_order_relaxed);

        // Bucket straight from integer nanoseconds: exact, and it keeps the
        // produce path free of float rounding entirely.
        if (lat_ns < 0) {
            // A negative end-to-end latency is physically impossible — the frame
            // cannot be published before it was captured. It means the clock
            // reconciliation is wrong, so it is counted apart instead of folded
            // into bucket 0, where it would masquerade as a 0.5 ms latency.
            unreconciled_.fetch_add(1, std::memory_order_relaxed);
        } else if (const std::int64_t idx = lat_ns / kBucketNs;
                   idx < static_cast<std::int64_t>(kBucketCount)) {
            buckets_[static_cast<std::size_t>(idx)].fetch_add(1, std::memory_order_relaxed);
        } else {
            over_span_.fetch_add(1, std::memory_order_relaxed);
        }

        // Running max in nanoseconds, so the produce path does no float compare.
        // A plain compare-then-store is enough because there is exactly one
        // writer — unlike core::JitterHistogram, which is written by many
        // threads and therefore needs a CAS loop.
        if (lat_ns > max_ns_.load(std::memory_order_relaxed)) {
            max_ns_.store(lat_ns, std::memory_order_relaxed);
        }

        // WHY A SLOPE, and not just p99.
        //
        // The failure the trend exists to catch is a queue silently filling:
        // the inference stage consuming frames slower than the camera produces
        // them, or a driver-side buffer backing up. That shows up as latency
        // RAMPING LINEARLY with time — every extra frame sitting in the backlog
        // adds one frame period to the age of whatever we publish next.
        //
        // Percentiles cannot see it. A run that starts at 20 ms and ends at
        // 300 ms and a run that sits at 40 ms and spikes to 300 ms now and then
        // have the SAME p99 and a similar max, yet only the first one is broken:
        // one is a backlog that will keep growing, the other is bounded jitter.
        // A least-squares fit of latency against elapsed time separates them
        // directly — ~0 ms/min is stationary, a persistently positive slope is a
        // backlog — for five running sums, O(1) and allocation-free.
        //
        // t is measured from the FIRST sample, not from an arbitrary epoch: it
        // keeps the magnitudes small, which is what keeps the
        // n*sum_t2 - sum_t^2 cancellation in summarize() well conditioned.
        //
        // WARM-UP IS EXCLUDED FROM THE FIT. The first inference on a TensorRT
        // engine pays cuDNN autotune and CUDA-graph capture and can be a full
        // second — measured at 1275 ms on an Orin NX against a 55 ms steady
        // state. Left in, that single point drags a 60 s least-squares fit to
        // roughly -18 ms/min on a pipeline that is behaving perfectly. It is a known one-off transient, not the queue growth this
        // slope exists to detect, so the first kSlopeWarmupSamples are recorded
        // in the histogram and the max (where the operator should still see
        // them) but kept out of the regression.
        if (n + 1U <= kSlopeWarmupSamples) {
            return;
        }
        const double t_min =
            static_cast<double>(publish_ns - first_ns_.load(std::memory_order_relaxed))
            * kNsToMinutes;
        const double l_ms = static_cast<double>(lat_ns) * 1.0e-6;
        fit_n_.store(fit_n_.load(std::memory_order_relaxed) + 1U, std::memory_order_relaxed);
        sum_t_.store(sum_t_.load(std::memory_order_relaxed) + t_min, std::memory_order_relaxed);
        sum_t2_.store(sum_t2_.load(std::memory_order_relaxed) + (t_min * t_min),
                      std::memory_order_relaxed);
        sum_l_.store(sum_l_.load(std::memory_order_relaxed) + l_ms, std::memory_order_relaxed);
        sum_tl_.store(sum_tl_.load(std::memory_order_relaxed) + (t_min * l_ms),
                      std::memory_order_relaxed);
    }

    struct Summary {
        std::uint64_t samples      = 0;
        std::uint64_t unreconciled = 0;  // lat < 0: the clock mapping is broken
        std::uint64_t over_span    = 0;  // lat >= 512 ms: off the histogram
        float         p50_ms       = 0.0F;
        float         p99_ms       = 0.0F;
        float         max_ms       = 0.0F;
        float         slope_ms_per_min = 0.0F;
        float         mean_hz          = 0.0F;
    };

    // Off the produce path: walks 1024 buckets and does the least-squares
    // division. Cheap enough for a 1 Hz log line, not for a per-frame one.
    //
    // Every load is relaxed, so a caller racing the very first record() can see
    // a torn set — n_ already bumped while first_ns_/max_ns_ are still at their
    // initial values. That is the same "not a consistent snapshot" contract the
    // DepthStats counters have always had, and it cannot mislead here because
    // both cases are guarded below rather than divided through.
    [[nodiscard]] Summary summarize() const noexcept {
        Summary s;
        s.samples      = n_.load(std::memory_order_relaxed);
        s.unreconciled = unreconciled_.load(std::memory_order_relaxed);
        s.over_span    = over_span_.load(std::memory_order_relaxed);
        if (s.samples == 0U) {
            return s;
        }
        // kInitMax means no max has become visible yet; reporting it raw would
        // print about -9.2e6 ms.
        const std::int64_t max_ns = max_ns_.load(std::memory_order_relaxed);
        s.max_ms = max_ns == kInitMax ? 0.0F
                                      : static_cast<float>(static_cast<double>(max_ns) * 1.0e-6);
        s.p50_ms = percentile_ms(s.samples, 0.50, s.max_ms);
        s.p99_ms = percentile_ms(s.samples, 0.99, s.max_ms);

        // Acquire, pairing with record()'s one release store: it is what makes
        // first_ns_ (stored relaxed) guaranteed-visible here. Until it is set
        // there is no span to measure against, so mean_hz and the trend stay 0
        // rather than being computed from a half-published state.
        const bool         have_first = have_first_.load(std::memory_order_acquire);
        const std::int64_t span_ns =
            have_first ? last_ns_.load(std::memory_order_relaxed)
                             - first_ns_.load(std::memory_order_relaxed)
                       : 0;
        if (s.samples >= 2U && span_ns > 0) {
            // (n-1) intervals over the span the publishes actually cover, so a
            // producer that started late is not charged for its warm-up. This is
            // the whole-run mean the EMA cannot give: ema_hz forgets, by design.
            s.mean_hz = static_cast<float>(static_cast<double>(s.samples - 1U)
                                           / (static_cast<double>(span_ns) * 1.0e-9));
        }

        const double span_min = static_cast<double>(span_ns) * kNsToMinutes;
        const std::uint64_t fit_n = fit_n_.load(std::memory_order_relaxed);
        if (fit_n >= kMinSlopeSamples && span_min >= kMinSlopeSpanMin) {
            // MUST be fit_n, not s.samples: the five sums below accumulate only
            // post-warm-up samples, and mixing a larger n with them would bias
            // every term of the regression.
            const double n      = static_cast<double>(fit_n);
            const double sum_t  = sum_t_.load(std::memory_order_relaxed);
            const double sum_t2 = sum_t2_.load(std::memory_order_relaxed);
            const double sum_l  = sum_l_.load(std::memory_order_relaxed);
            const double sum_tl = sum_tl_.load(std::memory_order_relaxed);
            const double denom  = (n * sum_t2) - (sum_t * sum_t);
            // Guarded rather than assumed non-zero: with every sample at the
            // same t the denominator is a rounding residue, and dividing by it
            // would print a spectacular slope for a run that never moved.
            if (denom > 0.0) {
                s.slope_ms_per_min = static_cast<float>(((n * sum_tl) - (sum_t * sum_l)) / denom);
            }
        }
        return s;
    }

private:
    static constexpr double       kNsToMinutes = 1.0 / 60.0e9;
    static constexpr std::int64_t kInitMax     = (std::numeric_limits<std::int64_t>::min)();

    // Cumulative walk reporting the bucket's UPPER edge. core::JitterHistogram
    // reports the LOWER edge; this number is read as a latency bound, so it must
    // never read BETTER than reality.
    [[nodiscard]] float percentile_ms(std::uint64_t total, double q, float max_ms) const noexcept {
        // ceil, so q=0.99 over 100 samples targets the 99th sample, not the
        // 98th. total >= 1 here, so target >= 1.
        const auto target = static_cast<std::uint64_t>(std::ceil(q * static_cast<double>(total)));
        std::uint64_t acc = unreconciled_.load(std::memory_order_relaxed);
        if (acc >= target) {
            // The quantile lands among the negative (unreconciled) samples. The
            // histogram cannot express those, so return a negative marker rather
            // than pretend the latency was sub-millisecond.
            return -1.0F;
        }
        for (std::size_t i = 0; i < kBucketCount; ++i) {
            acc += buckets_[i].load(std::memory_order_relaxed);
            if (acc >= target) {
                return static_cast<float>(i + 1U) * kBucketMs;
            }
        }
        // Landed in the overflow bucket: the running max is the only real number
        // available for it, and it is an upper bound — the safe direction.
        return max_ms;
    }

    // A mutex here would make the publish blockable and defeat the whole
    // wait-free hand-off, and std::atomic silently falls back to one for any
    // type the platform cannot do losslessly. Prove it instead of hoping.
    static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
                  "LatencyTracker counters must be lock-free on the produce path");
    static_assert(std::atomic<std::int64_t>::is_always_lock_free,
                  "LatencyTracker max must be lock-free on the produce path");
    static_assert(std::atomic<double>::is_always_lock_free,
                  "LatencyTracker least-squares sums must be lock-free on the produce path");

    alignas(64) std::array<std::atomic<std::uint64_t>, kBucketCount> buckets_{};
    alignas(64) std::atomic<std::uint64_t> n_{0};
    std::atomic<std::uint64_t> unreconciled_{0};
    std::atomic<std::uint64_t> over_span_{0};
    std::atomic<std::int64_t>  max_ns_{kInitMax};
    std::atomic<std::int64_t>  first_ns_{0};
    std::atomic<std::int64_t>  last_ns_{0};
    // Publication flag for first_ns_; see record()/summarize(). Not a counter.
    std::atomic<bool> have_first_{false};
    // Least-squares accumulators, latency [ms] against elapsed time [min].
    // Samples that actually entered the regression (n_ minus the warm-up).
    std::atomic<std::uint64_t> fit_n_{0};
    std::atomic<double> sum_t_{0.0};
    std::atomic<double> sum_t2_{0.0};
    std::atomic<double> sum_l_{0.0};
    std::atomic<double> sum_tl_{0.0};
};

// Diagnostics. Read from any thread; each field is an independent relaxed
// atomic load, so the set is not a consistent snapshot — it is for logging,
// not for control decisions.
struct DepthStats {
    std::uint64_t captured  = 0;     // frames delivered by the camera
    std::uint64_t published = 0;     // g1::DepthFrame handoffs to the control loop
    std::uint64_t dropped   = 0;     // captured pairs superseded before inference
    std::uint64_t overruns  = 0;     // cycles that missed the rate_hz deadline
    std::uint64_t errors    = 0;     // non-fatal inference failures
    float         last_ms   = 0.0F;  // most recent produce (infer) wall time
    float         ema_hz    = 0.0F;  // smoothed publish rate
};

class IDepthSource {
public:
    virtual ~IDepthSource() = default;

    IDepthSource()                               = default;
    IDepthSource(const IDepthSource&)            = delete;
    IDepthSource& operator=(const IDepthSource&) = delete;
    IDepthSource(IDepthSource&&)                 = delete;
    IDepthSource& operator=(IDepthSource&&)      = delete;

    // Acquire the device, warm the GPU, and spawn the producer thread(s).
    // Off hot path. On error nothing is left running.
    [[nodiscard]] virtual core::Result<void, std::string> start() = 0;

    // Signal the producer thread(s) to exit and join them. Idempotent; the
    // destructor calls it. Safe to call even if start() failed.
    virtual void stop() noexcept = 0;

    [[nodiscard]] virtual DepthStats       stats() const noexcept      = 0;
    [[nodiscard]] virtual StereoIntrinsics intrinsics() const noexcept = 0;
    [[nodiscard]] virtual std::string_view name() const noexcept       = 0;
};

// Build the producer named by `cfg.source` and bind it to `inbox`.
//
// `inbox` must outlive the returned source, and must have exactly one consumer
// (the control loop's depth inbox; see apps/g1/depth_setup.hpp).
//
// Returns an error (never a partially-started source) when the config is
// invalid, the build lacks the backend the config asks for (RealSense for
// either source, TensorRT+CUDA for kCleaned), or the device/plan cannot be
// opened. Callers decide whether a depth failure is fatal.
[[nodiscard]] core::Result<std::unique_ptr<IDepthSource>>
make_depth_source(const VisionConfig& cfg, core::TripleBuffer<g1::DepthFrame>& inbox);

// True when this build can actually construct `s`. Lets the app print a useful
// diagnostic instead of failing deep inside the factory.
[[nodiscard]] bool depth_source_available(DepthSource s) noexcept;

}  // namespace rt_deploy::vision
