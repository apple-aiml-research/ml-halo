//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#pragma once

// Lock-free single-producer single-consumer ring buffer.
//
// - Capacity is a compile-time power-of-two (wraps via mask, no modulo).
// - Each side's index sits on its own cache line to avoid false sharing.
// - push()/pop() are noexcept, allocation-free, and hot-path safe.
// - Designed for trivially-copyable payloads. Use std::move for non-trivial
//   types; the implementation uses placement-new + in-place destruction.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>

namespace rt_deploy::core {

#if defined(__cpp_lib_hardware_interference_size) && __cpp_lib_hardware_interference_size >= 201703L
constexpr std::size_t kCacheLine = std::hardware_destructive_interference_size;
#else
constexpr std::size_t kCacheLine = 64;
#endif

template <typename T, std::size_t Capacity>
class SpscRing {
    static_assert(Capacity >= 2 && (Capacity & (Capacity - 1)) == 0,
                  "Capacity must be a power of two >= 2");

public:
    SpscRing() noexcept = default;

    SpscRing(const SpscRing&)            = delete;
    SpscRing& operator=(const SpscRing&) = delete;
    SpscRing(SpscRing&&)                 = delete;
    SpscRing& operator=(SpscRing&&)      = delete;

    ~SpscRing() noexcept {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            while (pop_discard()) {
            }
        }
    }

    // Non-blocking push. Returns false if the ring is full. noexcept.
    template <typename U>
    [[nodiscard]] bool push(U&& value) noexcept(std::is_nothrow_constructible_v<T, U&&>) {
        const auto head = head_.load(std::memory_order_relaxed);
        const auto next = head + 1;
        if (next - tail_.load(std::memory_order_acquire) > Capacity) {
            return false;
        }
        ::new (slot(head)) T(std::forward<U>(value));
        head_.store(next, std::memory_order_release);
        return true;
    }

    // Non-blocking pop. Returns false if empty. On success, moves the value
    // into `out` and destroys the slot.
    [[nodiscard]] bool pop(T& out) noexcept(std::is_nothrow_move_assignable_v<T>) {
        const auto tail = tail_.load(std::memory_order_relaxed);
        if (head_.load(std::memory_order_acquire) == tail) {
            return false;
        }
        T* p = slot(tail);
        out  = std::move(*p);
        p->~T();
        tail_.store(tail + 1, std::memory_order_release);
        return true;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        const auto h = head_.load(std::memory_order_acquire);
        const auto t = tail_.load(std::memory_order_acquire);
        return static_cast<std::size_t>(h - t);
    }

    [[nodiscard]] bool empty() const noexcept { return size() == 0; }
    [[nodiscard]] static constexpr std::size_t capacity() noexcept { return Capacity; }

private:
    bool pop_discard() noexcept {
        T tmp;
        return pop(tmp);
    }

    T* slot(std::uint64_t idx) noexcept {
        return std::launder(reinterpret_cast<T*>(&storage_[idx & (Capacity - 1)]));
    }

    using Cell = std::aligned_storage_t<sizeof(T), alignof(T)>;

    alignas(kCacheLine) std::atomic<std::uint64_t> head_{0};
    alignas(kCacheLine) std::atomic<std::uint64_t> tail_{0};
    alignas(kCacheLine) Cell storage_[Capacity]{};
};

}  // namespace rt_deploy::core
