//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// CLI value-taking + the --engine choice. Header-only and dependency-free (like
// parse_int.hpp), shared by the app glue in apps/g1/common.hpp and
// apps/g1/depth_setup.hpp.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace rt_deploy::core {

// A flag's value may not look like another flag. Without this, a forgotten
// value silently eats the NEXT flag: `--trace_csv --no_send` writes a trace file
// named "--no_send" and actuates the robot, because --no_send never reached its
// parser. Only "--" is refused, so a negative number ("-1") stays a value.
[[nodiscard]] inline bool looks_like_flag(std::string_view v) noexcept {
    return v.size() >= 2 && v[0] == '-' && v[1] == '-';
}

namespace detail {
inline void report_missing_value(std::string_view flag, const char* got) noexcept {
    if (got != nullptr) {
        std::fprintf(stderr, "error: missing value for %.*s (got %s)\n",
                     static_cast<int>(flag.size()), flag.data(), got);
    } else {
        std::fprintf(stderr, "error: missing value for %.*s\n",
                     static_cast<int>(flag.size()), flag.data());
    }
}
}  // namespace detail

// argv form. `argv[i]` is the flag; on success returns its value and advances
// `i` onto it. On failure (end of argv, or the next token is a flag) prints
// "missing value for --X [(got --Y)]", returns nullptr and leaves `i` alone.
[[nodiscard]] inline const char* take_value(int argc, char* const* argv, int& i) noexcept {
    if (i + 1 >= argc) {
        detail::report_missing_value(argv[i], nullptr);
        return nullptr;
    }
    if (looks_like_flag(argv[i + 1])) {
        detail::report_missing_value(argv[i], argv[i + 1]);
        return nullptr;
    }
    return argv[++i];
}

// Leftover-argument-vector form (the per-app extras after parse_common_args).
[[nodiscard]] inline const std::string* take_value(const std::vector<std::string>& args,
                                                   std::size_t&                    i) noexcept {
    if (i + 1 >= args.size()) {
        detail::report_missing_value(args[i], nullptr);
        return nullptr;
    }
    if (looks_like_flag(args[i + 1])) {
        detail::report_missing_value(args[i], args[i + 1].c_str());
        return nullptr;
    }
    return &args[++i];
}

// --engine auto|trt|ort. parse_engine_choice() maps anything unknown to kAuto,
// so the app must reject an unknown spelling with is_engine_choice() first —
// otherwise `--engine onnx` quietly runs whatever auto picks.
enum class PreferredEngine : std::uint8_t { kAuto, kTrt, kOrt };

[[nodiscard]] inline bool is_engine_choice(std::string_view s) noexcept {
    return s == "auto" || s == "trt" || s == "ort";
}

[[nodiscard]] inline PreferredEngine parse_engine_choice(std::string_view s) noexcept {
    if (s == "trt") return PreferredEngine::kTrt;
    if (s == "ort") return PreferredEngine::kOrt;
    return PreferredEngine::kAuto;
}

}  // namespace rt_deploy::core
