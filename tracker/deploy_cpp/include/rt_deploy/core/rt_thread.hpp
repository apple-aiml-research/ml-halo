//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// RT thread configuration helpers.
//
// On Linux: mlockall, sched_setaffinity, SCHED_FIFO priority, thread naming.
// On macOS/Windows: inert no-ops so the code runs on dev hosts. Failures return
// non-zero status codes; callers decide whether to abort or warn.

#include <cstdint>
#include <string_view>
#include <thread>

namespace rt_deploy::core {

struct RtThreadConfig {
    int          fifo_priority = 0;        // 1..99 on Linux; 0 disables SCHED_FIFO
    int          cpu_affinity  = -1;       // -1 = no pinning
    std::string_view name{};               // sets pthread name (<=15 chars on Linux)
    bool         lock_memory   = false;    // calls mlockall(MCL_CURRENT|MCL_FUTURE)
};

// Apply all configured attributes to the calling thread. Returns 0 on success,
// or a platform-specific errno-style code on failure. On non-Linux hosts this
// is a no-op that always returns 0.
[[nodiscard]] int configure_current_thread(const RtThreadConfig& cfg) noexcept;

// Lock all current + future pages into RAM. Should be called once at process
// start before any real-time thread spins up. Returns 0 on success.
[[nodiscard]] int lock_process_memory() noexcept;

// Pin the calling thread to a specific CPU core. cpu=-1 is a no-op.
[[nodiscard]] int pin_current_thread(int cpu) noexcept;

// Set SCHED_FIFO with the given priority on the calling thread.
[[nodiscard]] int set_fifo_priority(int priority) noexcept;

// Set the pthread name of the calling thread (truncated to 15 chars on Linux).
[[nodiscard]] int set_current_thread_name(std::string_view name) noexcept;

}  // namespace rt_deploy::core
