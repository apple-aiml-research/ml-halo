//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// Strict base-10 int parse for CLI values. Header-only and dependency-free;
// used by the app glue in apps/g1/common.hpp.

#include <cerrno>
#include <climits>
#include <cstdlib>

namespace rt_deploy::core {

// The whole string must be an integer that fits in int: an optional leading
// '+' or '-', then digits, nothing else (no whitespace, no suffix). atoi()
// would turn a typo like "5O" or "abc" into 5 / 0 and silently change
// behaviour. A leading '+' is accepted ("+5" == 5), matching strtol. On
// failure `out` is left untouched.
[[nodiscard]] inline bool parse_int_strict(const char* s, int& out) noexcept {
    // strtol would skip leading whitespace; require a sign or digit first.
    if (s == nullptr || !(*s == '-' || *s == '+' || (*s >= '0' && *s <= '9'))) {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    const long v = std::strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v < INT_MIN || v > INT_MAX) {
        return false;
    }
    out = static_cast<int>(v);
    return true;
}

}  // namespace rt_deploy::core
