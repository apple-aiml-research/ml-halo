//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// RateProbe + RateMonitor — measure each thread's iteration rate and
// per-iteration compute time at runtime. Gated by a global enable flag so
// the cost is one relaxed atomic load + branch when disabled.
//
// Usage:
//   1) In main thread (e.g. run_control_loop), call once:
//          rt_deploy::core::set_profiling_enabled(true);
//      then declare:
//          rt_deploy::core::RateMonitor monitor;     // prints every ~2 s
//
//   2) In every thread you want to probe, declare a static probe:
//          static rt_deploy::core::RateProbe probe{"control_loop"};
//      and at each iteration:
//          probe.tick();                  // counts the iteration (always)
//          { auto scope = probe.scope();  // measures compute time
//            // ... do work ...
//          }
//
//      `tick()` is a single relaxed atomic increment; `scope()` brackets
//      the work and feeds the elapsed time into a JitterHistogram. Both
//      are no-ops when set_profiling_enabled(false).

#include "rt_deploy/core/clock.hpp"
#include "rt_deploy/core/jitter_histogram.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace rt_deploy::core {

inline std::atomic<bool>& profiling_enabled_flag() noexcept {
    static std::atomic<bool> flag{false};
    return flag;
}
inline void set_profiling_enabled(bool on) noexcept {
    profiling_enabled_flag().store(on, std::memory_order_release);
}
[[nodiscard]] inline bool profiling_enabled() noexcept {
    return profiling_enabled_flag().load(std::memory_order_acquire);
}

class RateProbe {
public:
    explicit RateProbe(std::string name) : name_{std::move(name)} {
        const std::scoped_lock lock{reg_mu()};
        reg().push_back(this);
    }

    ~RateProbe() {
        const std::scoped_lock lock{reg_mu()};
        auto& r = reg();
        r.erase(std::remove(r.begin(), r.end(), this), r.end());
    }

    RateProbe(const RateProbe&)            = delete;
    RateProbe& operator=(const RateProbe&) = delete;

    // Bump the iteration counter. No-op when profiling is disabled.
    void tick() noexcept {
        if (!profiling_enabled()) return;
        count_.fetch_add(1, std::memory_order_relaxed);
    }

    // RAII scope: measures elapsed wall-clock time between construction and
    // destruction; feeds it into the per-probe JitterHistogram. No-op when
    // profiling is disabled.
    class Scope {
    public:
        explicit Scope(RateProbe& p) noexcept
            : probe_{profiling_enabled() ? &p : nullptr},
              start_{probe_ != nullptr ? MonotonicClock::now() : MonotonicClock::Nanos{}} {}

        ~Scope() noexcept {
            if (probe_ == nullptr) return;
            const auto end     = MonotonicClock::now();
            const auto elapsed = (end - start_).count();
            probe_->compute_.record(elapsed);
        }

        Scope(const Scope&)            = delete;
        Scope& operator=(const Scope&) = delete;
        Scope(Scope&& o) noexcept : probe_{o.probe_}, start_{o.start_} { o.probe_ = nullptr; }
        Scope& operator=(Scope&& o) noexcept {
            if (this != &o) {
                probe_   = o.probe_;
                start_   = o.start_;
                o.probe_ = nullptr;
            }
            return *this;
        }

    private:
        RateProbe*          probe_{nullptr};
        MonotonicClock::Nanos start_{};
    };

    [[nodiscard]] Scope scope() noexcept { return Scope{*this}; }

    [[nodiscard]] std::uint64_t count() const noexcept {
        return count_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    [[nodiscard]] const JitterHistogram& compute_hist() const noexcept { return compute_; }

    // Registry — used by RateMonitor to enumerate live probes.
    static std::mutex& reg_mu() noexcept {
        static std::mutex m;
        return m;
    }
    static std::vector<RateProbe*>& reg() noexcept {
        static std::vector<RateProbe*> r;
        return r;
    }

private:
    std::string                 name_;
    std::atomic<std::uint64_t>  count_{0};
    JitterHistogram             compute_{};
};

// Background thread that periodically prints, for each registered probe:
//   <name>=<rate Hz>  compute(p50/p99/max)=<a>/<b>/<c> µs
//
// Only spawns its worker thread when profiling is currently enabled at
// construction time. RAII-joined on dtor.
class RateMonitor {
public:
    explicit RateMonitor(std::chrono::milliseconds period
                            = std::chrono::milliseconds{2000}) {
        if (!profiling_enabled()) return;
        thread_ = std::thread{[this, period]() { run(period); }};
    }

    ~RateMonitor() {
        stop_.store(true, std::memory_order_release);
        if (thread_.joinable()) thread_.join();
    }

    RateMonitor(const RateMonitor&)            = delete;
    RateMonitor& operator=(const RateMonitor&) = delete;

    // Print one final, full-history summary. Safe to call from any thread.
    static void print_final_summary() {
        if (!profiling_enabled()) return;
        const std::scoped_lock lock{RateProbe::reg_mu()};
        std::fprintf(stderr, "[rate] final summary:\n");
        for (auto* p : RateProbe::reg()) {
            const auto s = p->compute_hist().summarize();
            std::fprintf(stderr,
                         "  %s: ticks=%llu  compute(min/p50/p99/p999/max)="
                         "%lld/%lld/%lld/%lld/%lld µs\n",
                         p->name().c_str(),
                         static_cast<unsigned long long>(p->count()),
                         static_cast<long long>(s.min_ns / 1000),
                         static_cast<long long>(s.p50_us),
                         static_cast<long long>(s.p99_us),
                         static_cast<long long>(s.p999_us),
                         static_cast<long long>(s.max_ns / 1000));
        }
    }

private:
    void run(std::chrono::milliseconds period) noexcept {
        using clock = std::chrono::steady_clock;
        auto last_time = clock::now();
        std::vector<std::pair<std::string, std::uint64_t>> last_counts;

        while (!stop_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(period);
            if (stop_.load(std::memory_order_acquire)) break;

            const auto   now  = clock::now();
            const double dt_s = std::chrono::duration<double>(now - last_time).count();
            last_time         = now;

            // Snapshot under lock; release before printing.
            struct Sample {
                std::string             name;
                std::uint64_t           count;
                JitterHistogram::Summary summary;
            };
            std::vector<Sample> snap;
            {
                const std::scoped_lock lock{RateProbe::reg_mu()};
                snap.reserve(RateProbe::reg().size());
                for (auto* p : RateProbe::reg()) {
                    snap.push_back({p->name(), p->count(), p->compute_hist().summarize()});
                }
            }

            std::fprintf(stderr, "[rate]");
            for (auto& s : snap) {
                std::uint64_t prev = 0;
                for (auto& [pname, pcount] : last_counts) {
                    if (pname == s.name) {
                        prev = pcount;
                        break;
                    }
                }
                const double hz = (dt_s > 0.0)
                                      ? static_cast<double>(s.count - prev) / dt_s
                                      : 0.0;
                std::fprintf(stderr,
                             "  %s=%.1fHz(p50=%lld p99=%lld µs)",
                             s.name.c_str(), hz,
                             static_cast<long long>(s.summary.p50_us),
                             static_cast<long long>(s.summary.p99_us));
            }
            std::fprintf(stderr, "\n");

            last_counts.clear();
            last_counts.reserve(snap.size());
            for (auto& s : snap) {
                last_counts.emplace_back(s.name, s.count);
            }
        }
    }

    std::atomic<bool> stop_{false};
    std::thread       thread_;
};

}  // namespace rt_deploy::core
