//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#include "rt_deploy/core/clock.hpp"

// All MonotonicClock methods are header-inline and noexcept. This TU exists
// only to give the static library a non-empty object file for the clock
// component, which helps link ordering on some toolchains.
namespace rt_deploy::core {
[[maybe_unused]] static constexpr int kClockTuAnchor = 0;
}  // namespace rt_deploy::core
