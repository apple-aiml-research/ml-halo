//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// Bucketed microsecond histogram for loop-period / inference-latency tracking.
//
// - Fixed number of fixed-width buckets (default 0..2000 µs in 10 µs bins).
// - record() is noexcept, lock-free, and hot-path safe.
// - Snapshot/summary helpers are NOT hot-path safe (they iterate + sort).

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ostream>
#include <vector>

namespace rt_deploy::core {

class JitterHistogram {
public:
    static constexpr std::size_t kBucketCount = 200;  // covers 0..2000 µs @ 10 µs
    static constexpr std::int64_t kBucketUs   = 10;

    void record(std::int64_t nanos) noexcept {
        // Fold negatives (clock jitter, preempt gap) into bucket 0.
        const std::int64_t us     = nanos / 1000;
        const std::int64_t b_idx  = us / kBucketUs;
        const std::size_t  bucket = (b_idx < 0)                                ? 0
                                    : (b_idx >= static_cast<std::int64_t>(kBucketCount))
                                        ? kBucketCount - 1
                                        : static_cast<std::size_t>(b_idx);
        buckets_[bucket].fetch_add(1, std::memory_order_relaxed);
        samples_.fetch_add(1, std::memory_order_relaxed);

        // Track min/max with relaxed CAS loops (off hot critical path for most ticks).
        std::int64_t cur_min = min_ns_.load(std::memory_order_relaxed);
        while (nanos < cur_min
               && !min_ns_.compare_exchange_weak(cur_min, nanos, std::memory_order_relaxed)) {
        }
        std::int64_t cur_max = max_ns_.load(std::memory_order_relaxed);
        while (nanos > cur_max
               && !max_ns_.compare_exchange_weak(cur_max, nanos, std::memory_order_relaxed)) {
        }
    }

    [[nodiscard]] std::uint64_t samples() const noexcept {
        return samples_.load(std::memory_order_relaxed);
    }

    struct Summary {
        std::uint64_t samples = 0;
        std::int64_t  min_ns  = 0;
        std::int64_t  max_ns  = 0;
        std::int64_t  p50_us  = 0;
        std::int64_t  p99_us  = 0;
        std::int64_t  p999_us = 0;
    };

    // NOT hot-path safe. Call off the control thread.
    [[nodiscard]] Summary summarize() const {
        Summary s;
        s.samples = samples();
        s.min_ns  = min_ns_.load(std::memory_order_relaxed);
        s.max_ns  = max_ns_.load(std::memory_order_relaxed);
        if (s.samples == 0) {
            s.min_ns = 0;
            s.max_ns = 0;
            return s;
        }
        s.p50_us  = percentile_us(0.50);
        s.p99_us  = percentile_us(0.99);
        s.p999_us = percentile_us(0.999);
        return s;
    }

    void dump_csv(std::ostream& out) const;

    void reset() noexcept {
        for (auto& b : buckets_) {
            b.store(0, std::memory_order_relaxed);
        }
        samples_.store(0, std::memory_order_relaxed);
        min_ns_.store(kInitMin, std::memory_order_relaxed);
        max_ns_.store(kInitMax, std::memory_order_relaxed);
    }

private:
    [[nodiscard]] std::int64_t percentile_us(double q) const;

    static constexpr std::int64_t kInitMin = (std::numeric_limits<std::int64_t>::max)();
    static constexpr std::int64_t kInitMax = (std::numeric_limits<std::int64_t>::min)();

    alignas(64) std::array<std::atomic<std::uint64_t>, kBucketCount> buckets_{};
    alignas(64) std::atomic<std::uint64_t> samples_{0};
    std::atomic<std::int64_t> min_ns_{kInitMin};
    std::atomic<std::int64_t> max_ns_{kInitMax};
};

}  // namespace rt_deploy::core
