//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#include "rt_deploy/core/rt_thread.hpp"

#include <algorithm>
#include <array>
#include <cstring>

#if defined(__linux__)
#    include <cerrno>  // errno — inspected after a failed mlockall()
#    include <pthread.h>
#    include <sched.h>
#    include <sys/mman.h>
#    include <unistd.h>
#endif

namespace rt_deploy::core {

int lock_process_memory() noexcept {
#if defined(__linux__)
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        return errno != 0 ? errno : -1;
    }
    return 0;
#else
    return 0;  // not supported on this host; return success so dev hosts can proceed
#endif
}

int pin_current_thread([[maybe_unused]] int cpu) noexcept {
#if defined(__linux__)
    if (cpu < 0) {
        return 0;
    }
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    const int rc = pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
    return rc;
#else
    return 0;
#endif
}

int set_fifo_priority([[maybe_unused]] int priority) noexcept {
#if defined(__linux__)
    if (priority <= 0) {
        return 0;
    }
    sched_param p{};
    p.sched_priority = priority;
    return pthread_setschedparam(pthread_self(), SCHED_FIFO, &p);
#else
    return 0;
#endif
}

int set_current_thread_name([[maybe_unused]] std::string_view name) noexcept {
#if defined(__linux__)
    std::array<char, 16> buf{};
    const std::size_t n = std::min<std::size_t>(name.size(), buf.size() - 1);
    std::memcpy(buf.data(), name.data(), n);
    buf[n] = '\0';
    return pthread_setname_np(pthread_self(), buf.data());
#else
    return 0;
#endif
}

int configure_current_thread(const RtThreadConfig& cfg) noexcept {
    if (cfg.lock_memory) {
        if (const int rc = lock_process_memory(); rc != 0) {
            return rc;
        }
    }
    if (const int rc = pin_current_thread(cfg.cpu_affinity); rc != 0) {
        return rc;
    }
    if (!cfg.name.empty()) {
        if (const int rc = set_current_thread_name(cfg.name); rc != 0) {
            return rc;
        }
    }
    if (const int rc = set_fifo_priority(cfg.fifo_priority); rc != 0) {
        return rc;
    }
    return 0;
}

}  // namespace rt_deploy::core
