//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// MonotonicClock — deterministic, phase-locked wake-ups.
//
// On Linux, uses clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, ...) so wake
// jitter is bounded by kernel tick resolution, not by sleep-duration drift.
// On macOS/Windows dev hosts, falls back to std::this_thread::sleep_until so
// the code builds for simulation; those hosts are not RT targets.

#include <chrono>
#include <cstdint>
#include <thread>

#if defined(__linux__)
#    include <cerrno>  // EINTR — used by the clock_nanosleep resume loop below
#    include <time.h>
#endif

namespace rt_deploy::core {

class MonotonicClock {
public:
    using Nanos = std::chrono::nanoseconds;

    // Return the current CLOCK_MONOTONIC timestamp in nanoseconds since an
    // unspecified epoch. noexcept, hot-path safe.
    [[nodiscard]] static Nanos now() noexcept {
#if defined(__linux__)
        struct timespec ts{};
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return Nanos{static_cast<int64_t>(ts.tv_sec) * 1'000'000'000LL + ts.tv_nsec};
#else
        using Clock = std::chrono::steady_clock;
        return std::chrono::duration_cast<Nanos>(Clock::now().time_since_epoch());
#endif
    }

    // Sleep until the given absolute monotonic deadline. Returns the actual
    // wake-up timestamp (for jitter measurement).
    [[nodiscard]] static Nanos sleep_until(Nanos deadline) noexcept {
#if defined(__linux__)
        struct timespec ts {};
        ts.tv_sec  = static_cast<time_t>(deadline.count() / 1'000'000'000LL);
        ts.tv_nsec = static_cast<long>(deadline.count() % 1'000'000'000LL);
        // Loop on EINTR — signals (e.g. profiler) may otherwise cut sleep short.
        int rc = 0;
        do {
            rc = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr);
        } while (rc == EINTR);
#else
        std::this_thread::sleep_until(
            std::chrono::steady_clock::time_point{deadline});
#endif
        return now();
    }
};

// Drives a fixed-period loop with phase-locked wake-ups. Construct once at
// loop start with the desired period, then call wait_next() every iteration.
// Returns the wake-up timestamp and the absolute scheduled deadline so
// callers can record jitter.
class PeriodicTicker {
public:
    using Nanos = MonotonicClock::Nanos;

    explicit PeriodicTicker(Nanos period) noexcept
        : period_{period}, next_{MonotonicClock::now() + period} {}

    struct Tick {
        Nanos scheduled;  // the deadline the control loop was supposed to wake on
        Nanos actual;     // actual wake-up time
        Nanos jitter;     // actual - scheduled (can be negative only if clock drifts)
    };

    [[nodiscard]] Tick wait_next() noexcept {
        const Nanos scheduled = next_;
        const Nanos actual    = MonotonicClock::sleep_until(scheduled);
        next_ += period_;
        // If we overran the period by >= one tick, resync to "now" so we don't
        // spin trying to catch up on missed ticks (which would compound jitter).
        if (actual > next_) {
            next_ = actual + period_;
        }
        return Tick{scheduled, actual, actual - scheduled};
    }

    [[nodiscard]] Nanos period() const noexcept { return period_; }

private:
    Nanos period_;
    Nanos next_;
};

}  // namespace rt_deploy::core
