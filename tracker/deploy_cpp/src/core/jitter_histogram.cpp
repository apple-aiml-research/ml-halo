//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#include "rt_deploy/core/jitter_histogram.hpp"

namespace rt_deploy::core {

std::int64_t JitterHistogram::percentile_us(double q) const {
    const std::uint64_t total = samples();
    if (total == 0) {
        return 0;
    }
    const std::uint64_t target = static_cast<std::uint64_t>(q * static_cast<double>(total));
    std::uint64_t       acc    = 0;
    for (std::size_t i = 0; i < kBucketCount; ++i) {
        acc += buckets_[i].load(std::memory_order_relaxed);
        if (acc >= target) {
            return static_cast<std::int64_t>(i) * kBucketUs;
        }
    }
    return static_cast<std::int64_t>(kBucketCount) * kBucketUs;
}

void JitterHistogram::dump_csv(std::ostream& out) const {
    out << "bucket_start_us,bucket_end_us,count\n";
    for (std::size_t i = 0; i < kBucketCount; ++i) {
        const auto start = static_cast<std::int64_t>(i) * kBucketUs;
        const auto end   = start + kBucketUs;
        out << start << ',' << end << ',' << buckets_[i].load(std::memory_order_relaxed) << '\n';
    }
}

}  // namespace rt_deploy::core
