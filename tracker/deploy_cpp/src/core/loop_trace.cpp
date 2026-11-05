//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#include "rt_deploy/core/loop_trace.hpp"

#include <ctime>

namespace rt_deploy::core {

namespace {

// CLOCK_REALTIME in nanoseconds since the Unix epoch. Sampled exactly once, at
// create(), and paired with a MonotonicClock reading so every row's monotonic
// timestamp can be converted back to wall clock (and thus to video timecode)
// without ever re-reading a clock that can be stepped by NTP mid-run.
[[nodiscard]] std::int64_t realtime_now_ns() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

// Local-time ISO 8601 with UTC offset, e.g. 2026-08-17T14:22:05+0200. Local
// rather than UTC because the point of this field is eyeballing the trace
// against a camera's on-screen clock. Called once on the constructing thread,
// so the non-reentrant std::localtime is safe here (and portable, unlike
// localtime_r).
[[nodiscard]] std::string iso_local(std::int64_t wall_ns) {
    const auto  secs = static_cast<std::time_t>(wall_ns / 1'000'000'000LL);
    const std::tm* tm = std::localtime(&secs);
    if (tm == nullptr) {
        return "unknown";
    }
    std::array<char, 40> buf{};
    if (std::strftime(buf.data(), buf.size(), "%Y-%m-%dT%H:%M:%S%z", tm) == 0) {
        return "unknown";
    }
    return std::string{buf.data()};
}

}  // namespace

std::unique_ptr<LoopTrace> LoopTrace::create(const std::string& path,
                                              const MetaList&    meta,
                                              std::chrono::milliseconds flush_period) {
    std::FILE* out = std::fopen(path.c_str(), "wb");
    if (out == nullptr) {
        return nullptr;
    }

    // Sample the clock pair as close together as possible.
    const std::int64_t t0_wall = realtime_now_ns();
    const std::int64_t t0_mono = MonotonicClock::now().count();

    // v2 == v1 plus the three trailing depth columns. The version is bumped
    // rather than left at v1 because a reader that indexes columns POSITIONALLY
    // must be able to tell the two apart; readers that go by column name (e.g.
    // pandas) load either.
    std::fprintf(out, "# rt_deploy loop trace v2\n");
    for (const auto& [key, value] : meta) {
        std::fprintf(out, "# %s=%s\n", key.c_str(), value.c_str());
    }
    std::fprintf(out, "# t0_mono_ns=%lld\n", static_cast<long long>(t0_mono));
    std::fprintf(out, "# t0_wall_ns=%lld\n", static_cast<long long>(t0_wall));
    std::fprintf(out, "# t0_wall_iso=%s\n", iso_local(t0_wall).c_str());
    std::fprintf(out,
                 "# t_rel_s: seconds since t0_mono_ns (monotonic)\n"
                 "# t_wall_s: Unix epoch seconds = t0_wall_ns/1e9 + t_rel_s "
                 "(use this to sync with video)\n"
                 "# period_ms: interval between consecutive iteration wake-ups\n"
                 "# loop_ms: work done in the iteration, EXCLUDING the wait for "
                 "the next period\n"
                 "# obs_ms/infer_ms/act_ms: state read + observation build / "
                 "policy inference / action decode + command publish\n"
                 "# jitter_ms: measured period minus the nominal control period\n"
                 "# depth_seq: producer seq of the depth frame this tick used "
                 "(-1 = no depth inbox, or none published yet)\n"
                 "# depth_stale: control ticks since depth_seq last advanced "
                 "(0 == a new frame arrived this tick)\n"
                 "# depth_age_ms: this tick's wake-up minus the frame's publish "
                 "stamp; -1 == unknown (unstamped producer or no frame)\n");
    std::fprintf(out,
                 "iter,t_rel_s,t_wall_s,period_ms,loop_ms,obs_ms,infer_ms,"
                 "act_ms,jitter_ms,state,depth_seq,depth_stale,depth_age_ms\n");
    std::fflush(out);

    // Private ctor — can't use make_unique.
    auto trace = std::unique_ptr<LoopTrace>{new LoopTrace{path, out, t0_mono, t0_wall}};
    trace->writer_ =
        std::thread{[t = trace.get(), flush_period]() { t->run_writer(flush_period); }};
    return trace;
}

LoopTrace::LoopTrace(std::string path, std::FILE* out, std::int64_t t0_mono_ns,
                       std::int64_t t0_wall_ns) noexcept
    : path_{std::move(path)},
      out_{out},
      t0_mono_ns_{t0_mono_ns},
      t0_wall_ns_{t0_wall_ns},
      ring_{std::make_unique<SpscRing<LoopSample, kRingCapacity>>()} {}

LoopTrace::~LoopTrace() { close(); }

std::size_t LoopTrace::drain() noexcept {
    if (out_ == nullptr) {
        return 0;
    }
    const double t0_wall_s = static_cast<double>(t0_wall_ns_) / 1.0e9;
    std::size_t  n         = 0;
    LoopSample   s{};
    while (ring_->pop(s)) {
        const double t_rel_s =
            static_cast<double>(s.t_mono_ns - t0_mono_ns_) / 1.0e9;
        auto ms = [](std::int64_t ns) noexcept {
            return static_cast<double>(ns) / 1.0e6;
        };
        // -1 is a SENTINEL, not a duration: converting it would print -0.0000
        // and read like a real, near-zero age. Emitted verbatim so a reader can
        // test `depth_age_ms < 0` for "unknown". A frame published just AFTER
        // this tick's wake-up can legitimately give a small negative age too,
        // which is why depth_seq (-1) is the authoritative "no depth" marker.
        const double depth_age_ms =
            (s.depth_age_ns < 0) ? -1.0 : ms(s.depth_age_ns);
        std::fprintf(out_,
                     "%lld,%.6f,%.6f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%s,"
                     "%lld,%d,%.4f\n",
                     static_cast<long long>(s.iter), t_rel_s,
                     t0_wall_s + t_rel_s, ms(s.period_ns), ms(s.loop_ns),
                     ms(s.obs_ns), ms(s.infer_ns), ms(s.act_ns),
                     ms(s.jitter_ns), s.state.data(),
                     static_cast<long long>(s.depth_seq),
                     static_cast<int>(s.depth_stale), depth_age_ms);
        ++n;
    }
    if (n > 0) {
        written_.fetch_add(n, std::memory_order_relaxed);
    }
    return n;
}

void LoopTrace::run_writer(std::chrono::milliseconds period) noexcept {
    while (true) {
        {
            std::unique_lock lock{mu_};
            // Interruptible wait: close() sets stop_ under mu_ and notifies, so
            // shutdown doesn't have to wait out the remaining flush period.
            cv_.wait_for(lock, period,
                         [this] { return stop_.load(std::memory_order_acquire); });
        }
        const bool stopping = stop_.load(std::memory_order_acquire);
        if (drain() > 0) {
            // Flush every batch so a Ctrl-C, safety latch, or crash still
            // leaves a usable trace on disk (only the last <period worth of
            // rows can be lost).
            std::fflush(out_);
        }
        if (stopping) {
            return;
        }
    }
}

void LoopTrace::close() noexcept {
    bool expected = false;
    if (!closed_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return;  // already closed
    }
    {
        // Set the flag under the mutex so the writer cannot miss the wakeup
        // between evaluating its predicate and going back to sleep.
        const std::scoped_lock lock{mu_};
        stop_.store(true, std::memory_order_release);
    }
    cv_.notify_all();
    if (writer_.joinable()) {
        writer_.join();
    }
    // Writer is joined, so this thread now has exclusive access to out_.
    (void)drain();
    if (out_ != nullptr) {
        std::fprintf(out_, "# samples_written=%llu\n",
                     static_cast<unsigned long long>(written()));
        std::fprintf(out_, "# samples_dropped=%llu\n",
                     static_cast<unsigned long long>(dropped()));
        std::fflush(out_);
        std::fclose(out_);
        out_ = nullptr;
    }
}

}  // namespace rt_deploy::core
