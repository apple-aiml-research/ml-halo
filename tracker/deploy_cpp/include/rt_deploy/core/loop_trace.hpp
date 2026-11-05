//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// LoopTrace — per-iteration control-loop timing, streamed to CSV.
//
// Unlike JitterHistogram / RateProbe (which bucket samples and only ever report
// percentiles), LoopTrace keeps EVERY iteration. That is what makes the trace
// overlayable on an external recording: each row carries an absolute wall-clock
// timestamp, so the timing curve can be aligned with a third-person video of
// the robot frame-by-frame.
//
// Non-blocking by construction:
//   - record() runs on the RT control thread and does exactly one non-blocking
//     SPSC push of a trivially-copyable POD. No allocation, no lock, no
//     syscall, no I/O, no formatting.
//   - A background writer thread drains the ring and does all formatting and
//     file I/O. Construct LoopTrace from a NON-RT thread (i.e. before the
//     control loop calls configure_current_thread) so the writer never
//     inherits SCHED_FIFO and can never preempt the control thread.
//   - If the ring ever fills (writer starved), record() drops the sample and
//     bumps a counter rather than blocking. Drops are reported in the CSV
//     trailer, so a trace is never silently incomplete.
//
// All durations are CLOCK_MONOTONIC nanoseconds. One CLOCK_REALTIME /
// CLOCK_MONOTONIC pair is captured at create() and written into the header, so
// t_mono can be mapped onto wall clock (and therefore onto video timecode).
//
// Usage:
//     auto trace = core::LoopTrace::create("loop_cpp.csv", {{"impl", "cpp"}});
//     ...
//     core::LoopSample smp{};
//     { auto ts = core::PhaseStamp{&smp.obs_ns};   build_obs(); }
//     { auto ts = core::PhaseStamp{&smp.infer_ns}; infer();     }
//     smp.loop_ns = ...;
//     (void)trace->record(smp);
//     ...
//     trace->close();   // drains, writes the trailer, joins the writer

#include "rt_deploy/core/clock.hpp"
#include "rt_deploy/core/spsc_ring.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace rt_deploy::core {

// One control iteration. Trivially copyable so the SPSC ring can move it with a
// plain memcpy. Field order matches the CSV column order.
struct LoopSample {
    std::int64_t iter      = 0;  // control-tick index; never reset
    std::int64_t t_mono_ns = 0;  // wake-up timestamp of this iteration
    std::int64_t period_ns = 0;  // t_mono_ns - previous t_mono_ns (0 on the first)
    std::int64_t loop_ns   = 0;  // work done this iteration, excluding the wait
    std::int64_t obs_ns    = 0;  // state read + observation build
    std::int64_t infer_ns  = 0;  // policy inference (0 on zero-order-hold ticks)
    std::int64_t act_ns    = 0;  // action decode + command publish (+ sim step)
    std::int64_t jitter_ns = 0;  // period_ns minus the nominal control period
    std::array<char, 12> state{};  // control-state name, NUL-terminated

    // ── Depth hand-off freshness ────────────────────────────────────────────
    // The evidence that the control thread never BLOCKS on the depth producer:
    // it publishes a command on its own period regardless, and simply records
    // how old the frame it happened to read was. A per-tick record is what makes
    // that provable — an end-of-run "max stale ticks" counter cannot show
    // WHETHER a stall coincided with a period overrun, and inferring
    // non-blocking behaviour from the absence of overruns is not proof.
    //
    // Filled by g1::ControlLoop from its wait-free inbox read. The defaults are
    // the "no depth at all" values, so a proprio-only run is visibly (not
    // ambiguously) depth-free: depth_seq stays -1 for the whole trace.
    //
    // Named _ns here / emitted as depth_age_ms in the CSV, matching the
    // loop_ns → loop_ms convention every other duration in this struct follows.
    std::int64_t depth_seq    = -1;  // seq of the frame used this tick (-1 = none)
    std::int32_t depth_stale  = 0;   // control ticks since `seq` last advanced
    std::int64_t depth_age_ns = -1;  // tick wake-up minus the frame's publish stamp;
                                     // -1 == unknown (no frame, or an unstamped
                                     // producer). Read depth_seq first.

    // Copy a short state label in. Truncates silently; a bounded byte copy, so
    // it stays hot-path safe (no allocation, no strlen on unbounded input).
    void set_state(std::string_view s) noexcept {
        const std::size_t n = std::min(s.size(), state.size() - 1);
        for (std::size_t i = 0; i < n; ++i) {
            state[i] = s[i];
        }
        state[n] = '\0';
    }
};

// record() hands a LoopSample to the SPSC ring, which relocates it with a
// placement-new of a trivially-copyable type — i.e. a memcpy. Asserted rather
// than assumed: a member with a non-trivial copy (a std::string state label,
// say) would silently turn the RT thread's one ring push into an allocation.
static_assert(std::is_trivially_copyable_v<LoopSample>,
              "LoopSample travels through the SPSC ring as raw bytes");
static_assert(std::is_standard_layout_v<LoopSample>,
              "LoopSample must stay POD so its field order documents the CSV");

// RAII stopwatch that ACCUMULATES its elapsed nanoseconds into `out` on scope
// exit. Accumulating (rather than assigning) lets one logical phase be measured
// across several disjoint blocks of a loop body — e.g. obs_ns covering both the
// state read near the top of the tick and the observation build further down.
// Pass nullptr to disable: costs one branch and no clock read.
class PhaseStamp {
public:
    explicit PhaseStamp(std::int64_t* out) noexcept
        : out_{out},
          start_{out != nullptr ? MonotonicClock::now() : MonotonicClock::Nanos{}} {}

    ~PhaseStamp() noexcept {
        if (out_ == nullptr) {
            return;
        }
        *out_ += (MonotonicClock::now() - start_).count();
    }

    PhaseStamp(const PhaseStamp&)            = delete;
    PhaseStamp& operator=(const PhaseStamp&) = delete;
    PhaseStamp(PhaseStamp&& o) noexcept : out_{o.out_}, start_{o.start_} {
        o.out_ = nullptr;
    }
    PhaseStamp& operator=(PhaseStamp&&) = delete;

private:
    std::int64_t*         out_;
    MonotonicClock::Nanos start_;
};

class LoopTrace {
public:
    // 4096 samples = 82 s of backlog at 50 Hz. The writer drains every
    // flush_period (default 200 ms, i.e. ~10 samples), so this is pure slack
    // against a page-fault or scheduler stall on the writer thread.
    static constexpr std::size_t kRingCapacity = 4096;

    // Leading `# key=value` comment lines, emitted verbatim into the header.
    using MetaList = std::vector<std::pair<std::string, std::string>>;

    // Open `path`, write the header, start the writer thread. Returns nullptr
    // if the file cannot be opened. Call from a non-RT thread.
    [[nodiscard]] static std::unique_ptr<LoopTrace> create(
        const std::string& path, const MetaList& meta,
        std::chrono::milliseconds flush_period = std::chrono::milliseconds{200});

    ~LoopTrace();

    LoopTrace(const LoopTrace&)            = delete;
    LoopTrace& operator=(const LoopTrace&) = delete;

    // Hot path. One non-blocking SPSC push. Returns false if the sample was
    // dropped because the ring was full (never blocks, never throws).
    [[nodiscard]] bool record(const LoopSample& s) noexcept {
        if (!ring_->push(s)) {
            dropped_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        return true;
    }

    // Drain, write the trailer, join the writer, close the file. Idempotent;
    // also called by the destructor. record() stays safe (and drops) after this.
    void close() noexcept;

    [[nodiscard]] std::uint64_t dropped() const noexcept {
        return dropped_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t written() const noexcept {
        return written_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::int64_t t0_mono_ns() const noexcept { return t0_mono_ns_; }
    [[nodiscard]] std::int64_t t0_wall_ns() const noexcept { return t0_wall_ns_; }
    [[nodiscard]] const std::string& path() const noexcept { return path_; }

private:
    LoopTrace(std::string path, std::FILE* out, std::int64_t t0_mono_ns,
              std::int64_t t0_wall_ns) noexcept;

    void        run_writer(std::chrono::milliseconds period) noexcept;
    std::size_t drain() noexcept;

    std::string                   path_;
    std::FILE*                    out_{nullptr};
    std::int64_t                  t0_mono_ns_{0};
    std::int64_t                  t0_wall_ns_{0};
    std::unique_ptr<SpscRing<LoopSample, kRingCapacity>> ring_;
    std::atomic<std::uint64_t>    dropped_{0};
    std::atomic<std::uint64_t>    written_{0};
    std::atomic<bool>             stop_{false};
    std::atomic<bool>             closed_{false};
    // Makes the writer's inter-flush wait interruptible, so close() returns
    // promptly instead of blocking for up to one full flush period on the
    // shutdown path (which runs right after a safety latch or Ctrl-C).
    std::mutex                    mu_;
    std::condition_variable       cv_;
    std::thread                   writer_;
};

}  // namespace rt_deploy::core
